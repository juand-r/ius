#!/usr/bin/env python3
"""Entity coverage evaluation for summarization quality assessment."""

import json
import logging
import os
import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any
import spacy
from tqdm import tqdm
from unidecode import unidecode
from rapidfuzz import fuzz

from ius.utils import call_llm
from ius.exceptions import ValidationError

# Set up logging (only if not already configured)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_spacy_model():
    """Load spaCy model for entity extraction."""
    try:
        nlp = spacy.load("en_core_web_lg")
        return nlp
    except OSError:
        raise ValidationError(
            "spaCy English model not found. Please install it with: "
            "python -m spacy download en_core_web_lg"
        )

def extract_entities_with_spacy(text: str, nlp) -> List[str]:
    """Extract named entities from text using spaCy."""
    doc = nlp(text)
    
    # Focus on key entity types for relevance
    relevant_labels = {"PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "FAC", "WORK_OF_ART"}
    
    entities = []
    for ent in doc.ents:
        if ent.label_ in relevant_labels:
            # Clean up entity text (strip whitespace, handle newlines)
            clean_entity = ent.text.strip().replace('\n', ' ')
            if clean_entity and len(clean_entity) > 1:  # Filter out single characters
                entities.append(clean_entity)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_entities = []
    for entity in entities:
        if entity not in seen:
            seen.add(entity)
            unique_entities.append(entity)
    
    return unique_entities

def get_dataset_paths(dataset_name: str) -> Tuple[Path, Path]:
    """Get source dataset and chunk directory paths for a dataset."""
    dataset_mappings = {
        "bmds": ("datasets/bmds", "outputs/chunks/bmds_fixed_size2_8000"),
        "true-detective": ("datasets/true-detective", "outputs/chunks/true-detective_fixed_size_2000"),
        "squality": ("datasets/squality", "outputs/chunks/squality_fixed_size_8000"),
        "detectiveqa": ("datasets/detectiveqa", "outputs/chunks/detectiveqa_fixed_size_8000")
    }
    
    if dataset_name not in dataset_mappings:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {list(dataset_mappings.keys())}")
    
    source_path, chunk_path = dataset_mappings[dataset_name]
    return Path(source_path), Path(chunk_path)

def extract_dataset_name(input_path: str) -> str:
    """Extract dataset name from input path."""
    input_path = input_path.rstrip('/')
    last_dir = os.path.basename(input_path)
    dataset_name = last_dir.split("_")[0]
    return dataset_name

def get_cache_subdir(add_reveal: bool, reveal_only: bool) -> str:
    """Determine cache subdirectory based on reveal flags."""
    if reveal_only:
        return "source-reveal-only"
    elif add_reveal:  
        return "source-add-reveal"
    else:
        return "source-prereveal"

def get_source_entities(item_id: str, dataset_name: str, nlp, force_extract: bool = False, model: str = "gpt-5-mini", add_reveal: bool = False, reveal_only: bool = False) -> Tuple[List[str], List[str], Dict[str, List[str]], str]:
    """Get or extract entities from source dataset item."""
    
    # Check cache first
    cache_subdir = get_cache_subdir(add_reveal, reveal_only)
    cache_file = Path(f"outputs/eval/intrinsic/source-processed/entities/{cache_subdir}/{dataset_name}/items/{item_id}.json")
    
    if cache_file.exists() and not force_extract:
        logger.debug(f"Loading cached entities for {item_id}")
        with open(cache_file, 'r') as f:
            cached_data = json.load(f)
        
        original_entities = cached_data.get("original_entities", [])
        deduplicated_entities = cached_data.get("deduplicated_entities", [])
        grouping_info = cached_data.get("grouping_info", {})
        deduplication_response = cached_data.get("deduplication_response", "")
        
        logger.info(f"Loaded {len(original_entities)} original/{len(deduplicated_entities)} deduplicated cached entities for {item_id}")
        return original_entities, deduplicated_entities, grouping_info, deduplication_response
    
    # Extract from source dataset
    source_path, _ = get_dataset_paths(dataset_name)
    source_file = source_path / "items" / f"{item_id}.json"
    
    if not source_file.exists():
        raise FileNotFoundError(f"Source file not found: {source_file}")
    
    logger.info(f"Extracting entities from source: {item_id}")
    with open(source_file, 'r') as f:
        source_data = json.load(f)
    
    # Extract entities from content (in documents[0]['content'])
    documents = source_data.get("documents", [])
    if not documents:
        logger.warning(f"No documents found in source file: {source_file}")
        return [], [], {}, ""
    
    content = documents[0].get("content", "")
    if not content:
        logger.warning(f"No content found in documents[0] of source file: {source_file}")
        return [], [], {}, ""
    
    # Check for conflicting flags
    if add_reveal and reveal_only:
        raise ValueError("Cannot use both --add-reveal and --reveal-only flags together")
    
    # Add reveal text if requested
    if add_reveal:
        reveal_text = ""
        try:
            metadata = documents[0].get("metadata", {})
            
            if dataset_name == "bmds":
                # For bmds: documents[0]['metadata']['detection']['reveal_segment']
                reveal_text = metadata.get("detection", {}).get("reveal_segment", "")
            elif dataset_name == "true-detective":
                # For true-detective: documents[0]['metadata']['original_metadata']['reveal_text']
                reveal_text = metadata.get("original_metadata", {}).get("reveal_text", "")
            elif dataset_name == "detectiveqa":
                # For detectiveqa: documents[0]['metadata']['detection']['reveal_segment']
                reveal_text = metadata.get("detection", {}).get("reveal_segment", "")
            else:
                logger.warning(f"Unknown dataset '{dataset_name}' - cannot extract reveal text")
                raise ValueError(f"Unknown dataset '{dataset_name}' - cannot extract reveal text")
            
            if reveal_text:
                content = content + "\n\n" + reveal_text
                logger.debug(f"Added reveal text to {item_id} (length: {len(reveal_text)} chars)")
            else:
                logger.warning(f"No reveal text found for {item_id}")
                raise ValueError(f"No reveal text found for {item_id}")
                
        except Exception as e:
            logger.warning(f"Failed to extract reveal text for {item_id}: {e}")
            raise ValueError(f"Failed to extract reveal text for {item_id}: {e}")
    
    # Use only reveal text if requested  
    if reveal_only:
        reveal_text = ""
        try:
            metadata = documents[0].get("metadata", {})
            
            if dataset_name == "bmds":
                # For bmds: documents[0]['metadata']['detection']['reveal_segment']
                reveal_text = metadata.get("detection", {}).get("reveal_segment", "")
            elif dataset_name == "true-detective":
                # For true-detective: documents[0]['metadata']['original_metadata']['reveal_text']
                reveal_text = metadata.get("original_metadata", {}).get("reveal_text", "")
            elif dataset_name == "detectiveqa":
                # For detectiveqa: documents[0]['metadata']['detection']['reveal_segment']
                reveal_text = metadata.get("detection", {}).get("reveal_segment", "")
            else:
                logger.warning(f"Unknown dataset '{dataset_name}' - cannot extract reveal text")
                raise ValueError(f"Unknown dataset '{dataset_name}' - cannot extract reveal text")
            
            if reveal_text:
                content = reveal_text
                logger.debug(f"Using only reveal text for {item_id} (length: {len(reveal_text)} chars)")
            else:
                logger.warning(f"No reveal text found for {item_id}")
                raise ValueError(f"No reveal text found for {item_id}")
                
        except Exception as e:
            logger.warning(f"Failed to extract reveal text for {item_id}: {e}")
            raise ValueError(f"Failed to extract reveal text for {item_id}: {e}")

    logger.debug(f"Content: {content}")

    original_entities = extract_entities_with_spacy(content, nlp)
    logger.info(f"Extracted {len(original_entities)} original entities, now deduplicating...")
    
    # Deduplicate entities
    deduplicated_entities, grouping_info, deduplication_response = deduplicate_entities_with_llm(original_entities, model, content)
    logger.info(f"Deduplicated to {len(deduplicated_entities)} canonical entities")
    
    # Save to cache
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_data = {
        "item_id": item_id,
        "original_entities": original_entities,
        "deduplicated_entities": deduplicated_entities,
        "grouping_info": grouping_info,
        "deduplication_model": model,
        "deduplication_response": deduplication_response,
        "extraction_metadata": {
            "method": "spacy",
            "model": "en_core_web_lg",
            "entity_types": ["PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "FAC", "WORK_OF_ART"],
            "timestamp": datetime.now().isoformat(),
            "total_original_entities": len(original_entities),
            "total_deduplicated_entities": len(deduplicated_entities),
            "content_processing": {
                "add_reveal": add_reveal,
                "reveal_only": reveal_only,
                "cache_subdir": cache_subdir
            }
        }
    }
    
    with open(cache_file, 'w') as f:
        json.dump(cache_data, f, indent=2)
    
    logger.info(f"Cached {len(original_entities)} original/{len(deduplicated_entities)} deduplicated entities for {item_id}")
    return original_entities, deduplicated_entities, grouping_info, deduplication_response

def select_summary_text(summaries: List[str], range_spec: str) -> Tuple[str, List[int]]:
    """Select summary text based on range specification (same logic as whodunit)."""
    
    if not summaries:
        return "", []
    
    total_summaries = len(summaries)
    
    if range_spec == "all":
        selected_indices = list(range(total_summaries))
        selected_text = "\n\n".join(summaries)
    elif range_spec == "last":
        selected_indices = [total_summaries - 1]
        selected_text = summaries[-1]
    elif range_spec == "penultimate":
        if total_summaries < 2:
            selected_indices = [0]
            selected_text = summaries[0]
        else:
            selected_indices = [total_summaries - 2]
            selected_text = summaries[-2]
    elif range_spec == "all-but-last":
        if total_summaries <= 1:
            selected_indices = [0]
            selected_text = summaries[0]
        else:
            selected_indices = list(range(total_summaries - 1))
            selected_text = "\n\n".join(summaries[:-1])
    elif "-" in range_spec:
        # Handle "N-M" format
        try:
            start, end = map(int, range_spec.split("-"))
            start_idx = max(0, start - 1)  # Convert to 0-based
            end_idx = min(total_summaries, end)
            selected_indices = list(range(start_idx, end_idx))
            selected_text = "\n\n".join(summaries[start_idx:end_idx])
        except ValueError:
            raise ValueError(f"Invalid range format: {range_spec}")
    elif range_spec.isdigit():
        # Handle "N" format
        try:
            n = int(range_spec)
            if n <= 0 or n > total_summaries:
                raise ValueError(f"Range {n} out of bounds (1-{total_summaries})")
            selected_indices = [n - 1]  # Convert to 0-based
            selected_text = summaries[n - 1]
        except ValueError:
            raise ValueError(f"Invalid range format: {range_spec}")
    else:
        raise ValueError(f"Unknown range specification: {range_spec}")
    
    return selected_text, selected_indices

def deduplicate_entities_with_llm(entities: List[str], model: str = "gpt-5-mini", context_text: str = "") -> Tuple[List[str], Dict[str, List[str]], str]:
    """
    Use LLM to group entity variations that refer to the same person/entity.
    Uses OpenAI structured outputs to guarantee valid JSON.
    
    Returns:
        - deduplicated_entities: List of canonical entity names  
        - grouping_info: Dict mapping canonical name -> list of original variations
        - raw_llm_response: For debugging
    """
    if not entities:
        return [], {}, ""
    
    entity_list = ", ".join(entities)
    
    # Define JSON schema for structured outputs
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "entity_grouping",
            "schema": {
                "type": "object",
                "properties": {
                    "groups": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "canonical_name": {
                                    "type": "string",
                                    "description": "The most complete/formal name to use as the canonical version"
                                },
                                "variations": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "All variations of this entity, including the canonical name"
                                }
                            },
                            "required": ["canonical_name", "variations"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["groups"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
    
    try:
        response = call_llm(
            text="",  # Not used in template substitution
            model=model,
            system_and_user_prompt={
                "system": "You are an expert at identifying when different entity names refer to the same person, place, or organization in detective stories.",
                "user": "Here is some context from a document:\n\n{context_text}\n\n---\n\nHere is a list of entities extracted from this document: {entity_list}\n\nBe VERY CONSERVATIVE. Only group entities if they are:\n1. Obvious spelling variations (e.g., \"L'Hommedieu\" and \"L'Hommedieus\")\n2. Clear abbreviations (e.g., \"Dr. Smith\" and \"Smith\" when context shows they're the same)\n3. Formal vs informal versions of the same name (e.g., \"Robert\" and \"Bob\" when context confirms same person)\n\nDO NOT group entities that could be different people or group different 'aliases' of people. When in doubt, keep them separate. Use the context to verify they actually refer to the same entity.\n\nChoose the most complete/formal name as the canonical version. Every entity from the input list must appear exactly once in the variations array."
            },
            template_vars={
                "entity_list": entity_list,
                "context_text": context_text[:2000] if context_text else "No additional context available."  # Limit context to avoid token limits
            },
            temperature=1.0,  # gpt-5-mini only supports default temperature
            max_completion_tokens=10000,
            response_format=response_format
        )
        
        raw_response = response.get("response", "").strip()
        
        # Parse JSON response (should be guaranteed valid with structured outputs)
        import json
        try:
            result = json.loads(raw_response)
            groups = result.get("groups", [])
            
            deduplicated_entities = []
            grouping_info = {}
            
            for group in groups:
                canonical = group.get("canonical_name", "")
                variations = group.get("variations", [])
                
                if canonical and variations:
                    deduplicated_entities.append(canonical)
                    grouping_info[canonical] = variations
                    
            return deduplicated_entities, grouping_info, raw_response
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response for entity deduplication: {e}")
            logger.error(f"Raw response: {raw_response}")
            # Fallback: return original entities ungrouped
            return entities, {entity: [entity] for entity in entities}, raw_response
            
    except Exception as e:
        logger.error(f"LLM call failed for entity deduplication: {e}")
        # Fallback: return original entities ungrouped
        return entities, {entity: [entity] for entity in entities}, str(e)


def normalize_entity(entity: str) -> str:
    """
    Normalize entity text for string matching.
    
    Applies:
    - Lowercase conversion
    - Transliteration (é → e)  
    - Punctuation removal
    - Determiner removal (the, a, la, le, il, el)
    - Possessive removal ('s)
    - Expansion (& → and)
    - Whitespace normalization
    """
    # Convert to lowercase and transliterate
    normalized = unidecode(entity.lower())
    
    # Remove possessive 's at the end
    normalized = re.sub(r"'s$", "", normalized)
    
    # Expand common abbreviations
    normalized = normalized.replace("&", "and")
    normalized = normalized.replace(" and ", " and ")  # Normalize spacing
    
    # Remove determiners at the beginning
    determiners = ["the ", "a ", "an ", "la ", "le ", "il ", "el ", "los ", "las "]
    for det in determiners:
        if normalized.startswith(det):
            normalized = normalized[len(det):]
            break
    
    # Remove punctuation and extra whitespace
    normalized = re.sub(r'[^\w\s]', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized

def match_entities_hybrid(summary_entities: List[str], source_entities: List[str], 
                         model: str = "gpt-5-mini", edit_distance_threshold: float = 0.2) -> Tuple[List[Tuple[str, str]], List[str], Dict[str, str]]:
    """
    Hybrid entity matching: fast string matching first, then batch LLM for leftovers.
    
    Args:
        summary_entities: Entities from summary
        source_entities: Entities from source
        model: LLM model for batch matching of leftovers
        edit_distance_threshold: Threshold for normalized edit distance (0-1)
        
    Returns:
        - intersection: List of (summary_entity, matched_source_entity) tuples
        - summary_only: List of unmatched summary entities
        - raw_responses: Dict mapping summary_entity -> raw LLM response for debugging
    """
    intersection = []
    summary_only = []
    raw_responses = {}
    
    # Phase 1: Fast string matching
    logger.info(f"Phase 1: Fast string matching for {len(summary_entities)} entities...")
    
    # Normalize source entities once
    normalized_source = {normalize_entity(entity): entity for entity in source_entities}
    normalized_source_list = list(normalized_source.keys())
    
    leftovers = []
    
    for summary_entity in summary_entities:
        normalized_summary = normalize_entity(summary_entity)
        
        # Try exact match first
        if normalized_summary in normalized_source:
            matched_source = normalized_source[normalized_summary]
            intersection.append((summary_entity, matched_source))
            raw_responses[summary_entity] = f"FAST_MATCH: {matched_source}"
            continue
        
        # Try fuzzy matching with edit distance
        best_match = None
        best_score = 0
        
        for norm_source in normalized_source_list:
            # Use normalized edit distance (0 = identical, 1 = completely different)
            similarity = fuzz.ratio(normalized_summary, norm_source) / 100.0
            
            if similarity >= (1 - edit_distance_threshold) and similarity > best_score:
                best_score = similarity
                best_match = norm_source
        
        if best_match:
            matched_source = normalized_source[best_match]
            intersection.append((summary_entity, matched_source))
            raw_responses[summary_entity] = f"FUZZY_MATCH: {matched_source} (similarity: {best_score:.3f})"
        else:
            leftovers.append(summary_entity)
    
    logger.info(f"Phase 1 complete: {len(intersection)} fast matches, {len(leftovers)} leftovers for LLM")
    
    # Phase 2: Batch LLM for remaining entities
    if leftovers:
        logger.info(f"Phase 2: Batch LLM call for {len(leftovers)} leftover entities...")
        
        leftover_matches, batch_raw_response = match_entities_batch_llm(leftovers, source_entities, model)
        
        # Process batch results
        for summary_entity, matched_source in leftover_matches.items():
            if matched_source:
                intersection.append((summary_entity, matched_source))
            else:
                summary_only.append(summary_entity)
            raw_responses[summary_entity] = batch_raw_response
    
    logger.info(f"Hybrid matching complete: {len(intersection)} total matches, {len(summary_only)} unmatched")
    
    return intersection, summary_only, raw_responses

def match_entities_batch_llm(summary_entities: List[str], source_entities: List[str], 
                            model: str = "gpt-5-mini") -> Tuple[Dict[str, Optional[str]], str]:
    """
    Batch LLM call to match multiple summary entities against source entities.
    
    Returns:
        - matches: Dict mapping summary_entity -> matched_source_entity (or None)
        - raw_response: Raw LLM response for debugging
    """
    if not summary_entities:
        return {}, ""
    
    source_list = ", ".join(source_entities)
    summary_list = ", ".join(summary_entities)
    
    # Define structured output schema
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "entity_batch_matching",
            "schema": {
                "type": "object",
                "properties": {
                    "matches": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "summary_entity": {"type": "string"},
                                "matched_source_entity": {
                                    "type": ["string", "null"],
                                    "description": "The matching source entity name verbatim, or null if no match"
                                }
                            },
                            "required": ["summary_entity", "matched_source_entity"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["matches"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
    
    try:
        response = call_llm(
            text="",
            model=model,
            system_and_user_prompt={
                "system": "You are an expert at determining if entity names refer to the same entity, even with slight variations in spelling or formatting.",
                "user": "I need you to match each of these summary entities: {summary_entities}\n\nAgainst this list of source entities: {source_entities}\n\nFor each summary entity, determine if it matches any source entity (allowing for slight spelling differences, titles like Dr./Mr., abbreviations, etc.). If you find a match, return the source entity name verbatim. If no match, return null.\n\nReturn results as a JSON array with objects containing 'summary_entity' and 'matched_source_entity' fields."
            },
            template_vars={
                "summary_entities": summary_list,
                "source_entities": source_list
            },
            temperature=1.0,
            max_completion_tokens=10000,
            response_format=response_format
        )
        
        raw_response = response.get("response", "").strip()
        
        # Parse structured response
        result = json.loads(raw_response)
        matches = {}
        
        for match_obj in result.get("matches", []):
            summary_entity = match_obj.get("summary_entity")
            matched_source = match_obj.get("matched_source_entity")
            
            if summary_entity:
                # Validate that matched source is in our source list or is null
                if matched_source and matched_source not in source_entities:
                    logger.warning(f"LLM returned invalid source entity: '{matched_source}' for '{summary_entity}'")
                    matched_source = None
                
                matches[summary_entity] = matched_source
        
        # Ensure all summary entities are in the response
        for entity in summary_entities:
            if entity not in matches:
                logger.warning(f"Missing response for entity: {entity}")
                matches[entity] = None
        
        return matches, raw_response
        
    except Exception as e:
        logger.error(f"Batch LLM call failed for entity matching: {e}")
        # Fallback: return no matches
        return {entity: None for entity in summary_entities}, f"ERROR: {str(e)}"

def match_entity_with_llm(summary_entity: str, source_entities: List[str], 
                         model: str = "gpt-5-mini") -> Tuple[Optional[str], str]:
    """Use LLM to match a summary entity against source entities."""
    
    source_list = ", ".join(source_entities)
    
    try:
        response = call_llm(
            text="",  # Not used in template substitution
            model=model,
            system_and_user_prompt={
                "system": "You are an expert at determining if entity names refer to the same entity, even with slight variations in spelling or formatting.",
                "user": "Does the following {entity} match any of the entities in this list: {source_entity_list}? Note the entity be written slightly differently (e.g., Dr. Stone vs Mr. Stone vs John Stone; or Jacob vs 'Jacob the priest', or obvious spelling mistakes, etc.). If you find a match, return the matching entity from the list verbatim. If not, just say No. Do not say anything else."
            },
            template_vars={
                "entity": summary_entity,
                "source_entity_list": source_list
            },
            temperature=1.0,
            max_completion_tokens=10000
        )
        
        raw_response = response.get("response", "").strip()
        
        # Strict parsing: exact "No" means no match
        if raw_response == "No":
            return None, raw_response
        elif raw_response in source_entities:
            return raw_response, raw_response
        else:
            # LLM returned something else - treat as no match but log
            logger.warning(f"Unexpected LLM response for {summary_entity}: '{raw_response}'")
            return None, raw_response
            
    except Exception as e:
        logger.error(f"LLM call failed for entity matching: {e}")
        return None, f"ERROR: {str(e)}"

def calculate_entity_metrics(intersection: List[Tuple[str, str]], 
                           summary_only: List[str], 
                           source_only: List[str]) -> Dict[str, float]:
    """Calculate entity coverage metrics."""
    
    num_intersection = len(intersection)
    num_summary_only = len(summary_only) 
    num_source_only = len(source_only)
    num_source_total = num_intersection + num_source_only
    num_summary_total = num_intersection + num_summary_only
    
    # Jaccard similarity: intersection / union
    union_size = num_intersection + num_summary_only + num_source_only
    jaccard = num_intersection / union_size if union_size > 0 else 0.0
    
    # Recall: intersection / source_total
    recall = num_intersection / num_source_total if num_source_total > 0 else 0.0
    
    # Precision: intersection / summary_total  
    precision = num_intersection / num_summary_total if num_summary_total > 0 else 0.0
    
    return {
        "jaccard_similarity": jaccard,
        "recall": recall,
        "precision": precision,
        "num_source_entities": num_source_total,
        "num_summary_entities": num_summary_total,
        "num_matched_entities": num_intersection
    }

def generate_output_hash(hash_parameters: Dict[str, Any]) -> str:
    """Generate 6-character hash for output directory naming."""
    # Sort parameters for consistent hashing
    param_str = json.dumps(hash_parameters, sort_keys=True)
    return hashlib.md5(param_str.encode()).hexdigest()[:6]

def run_entity_coverage_evaluation(
    input_path: str,
    range_spec: str = "penultimate",
    model: str = "gpt-5-mini",
    prompt_name: str = "default-entity-matching",
    output_dir: Optional[str] = None,
    overwrite: bool = False,
    stop_after: Optional[int] = None,
    add_reveal: bool = False,
    reveal_only: bool = False,
    item_id: Optional[str] = None
) -> str:
    """
    Run entity coverage evaluation on summaries.
    
    Args:
        input_path: Path to summary collection (must start with "outputs/summaries")
        range_spec: Which summary parts to use ("penultimate", "all-but-last", etc.)
        model: LLM model for entity matching
        prompt_name: Name of matching prompt
        output_dir: Optional custom output directory name
        overwrite: Whether to overwrite existing results
        stop_after: Optional limit on number of items to process (for testing)
        add_reveal: Whether to append reveal text to source documents
        item_id: Optional specific item ID to process (if None, processes all items)
        
    Returns:
        Path to output directory
    """
    
    # Validation
    if not input_path.startswith("outputs/summaries"):
        raise ValueError("Can only be used to evaluate summaries, input must be in 'outputs/summaries'")
    
    input_path_obj = Path(input_path)
    if not input_path_obj.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    
    # Load spaCy model
    nlp = load_spacy_model()
    logger.info("Loaded spaCy model for entity extraction")
    
    # Extract dataset name
    dataset_name = extract_dataset_name(input_path)
    logger.info(f"Detected dataset: {dataset_name}")
    
    # Generate hash parameters (always needed for collection metadata)
    hash_parameters = {
        "extraction_method": "spacy",
        "extraction_model": "en_core_web_lg",
        "model": model,
        "prompt_name": prompt_name,
        "range_spec": range_spec,
        "temperature": 1.0,
        "max_completion_tokens": 10000,
        "add_reveal": add_reveal,
        "reveal_only": reveal_only
    }
    
    # Generate output directory name if not provided
    if output_dir is None:
        hash_value = generate_output_hash(hash_parameters)
        input_basename = os.path.basename(input_path.rstrip('/'))
        output_dir = f"{input_basename}_entity_coverage_{hash_value}"
    else:
        hash_value = generate_output_hash(hash_parameters)
    
    # Create output directory
    output_path = Path(f"outputs/eval/intrinsic/entity-coverage/{output_dir}")
    output_path.mkdir(parents=True, exist_ok=True)
    items_path = output_path / "items"
    items_path.mkdir(exist_ok=True)
    
    # Load input collection
    collection_file = input_path_obj / "collection.json"
    if not collection_file.exists():
        raise FileNotFoundError(f"Collection file not found: {collection_file}")
    
    with open(collection_file, 'r') as f:
        input_collection = json.load(f)
    
    # Extract summarization metadata from collection
    summarization_model = None
    summarization_prompt_name = None
    try:
        collection_metadata = input_collection.get("summarization_info", {}).get("collection_metadata", {})
        summarization_model = collection_metadata.get("model")
        summarization_prompt_name = collection_metadata.get("prompt_name")
    except Exception as e:
        logger.debug(f"Could not extract summarization metadata from collection: {e}")
    
    # Get list of items to process
    items_dir = input_path_obj / "items"
    if not items_dir.exists():
        raise FileNotFoundError(f"Items directory not found: {items_dir}")
    
    item_files = list(items_dir.glob("*.json"))
    if not item_files:
        raise ValueError(f"No JSON files found in {items_dir}")
    
    # Filter to specific item if requested
    if item_id is not None:
        item_files = [f for f in item_files if f.stem == item_id]
        if not item_files:
            raise ValueError(f"Item '{item_id}' not found in collection")
        logger.info(f"Processing single item: {item_id}")
    else:
        logger.info(f"Found {len(item_files)} items to process")
    
    # Initialize collection metadata
    collection_metadata = {
        "evaluation_function": "run_entity_coverage_evaluation",
        "content_type": "entity_coverage_analysis",
        "input_type": "summaries",
        "model": model,
        "prompt_name": prompt_name,
        "range_spec": range_spec,
        "temperature": 1.0,
        "max_completion_tokens": 10000,
        "extraction_method": "spacy",
        "extraction_model": "en_core_web_lg",
        "prompts_used": {
            "entity_deduplication": {
                "system": "You are an expert at identifying when different entity names refer to the same person, place, or organization in detective stories.",
                "user": "Here is some context from a document:\n\n{context_text}\n\n---\n\nHere is a list of entities extracted from this document: {entity_list}\n\nBe VERY CONSERVATIVE. Only group entities if they are:\n1. Obvious spelling variations (e.g., \"L'Hommedieu\" and \"L'Hommedieus\")\n2. Clear abbreviations (e.g., \"Dr. Smith\" and \"Smith\" when context shows they're the same)\n3. Formal vs informal versions of the same name (e.g., \"Robert\" and \"Bob\" when context confirms same person)\n\nDO NOT group entities that could be different people or group different 'aliases' of people. When in doubt, keep them separate. Use the context to verify they actually refer to the same entity.\n\nChoose the most complete/formal name as the canonical version. Every entity from the input list must appear exactly once in the variations array."
            },
            "entity_matching": {
                "system": "You are an expert at determining if entity names refer to the same entity, even with slight variations in spelling or formatting.",  
                "user": "I need you to match each of these summary entities: {summary_entities}\n\nAgainst this list of source entities: {source_entities}\n\nFor each summary entity, determine if it matches any source entity (allowing for slight spelling differences, titles like Dr./Mr., abbreviations, etc.). If you find a match, return the source entity name verbatim. If no match, return null.\n\nReturn results as a JSON array with objects containing 'summary_entity' and 'matched_source_entity' fields."
            }
        },
        "source_collection": input_path,
        "source_dataset": f"datasets/{dataset_name}",
        "command_run": f"python -m ius entity-coverage --input {input_path} --range {range_spec} --model {model}",
        "hash_parameters": hash_parameters,
        "hash_note": "Directory name contains 6-char MD5 hash of these parameters",
        "hash_value": hash_value
    }
    
    # Process items
    processed_items = []
    total_cost = 0.0
    total_tokens = 0
    total_matching_calls = 0
    total_fast_matches = 0
    total_entities_processed = 0
    successful_items = 0
    failed_items = 0
    skipped_items = 0
    
    # Apply stop_after limit if specified
    if stop_after is not None:
        original_count = len(item_files)
        item_files = item_files[:stop_after]
        logger.info(f"Limiting processing to first {stop_after} items (out of {original_count} available)")
    
    for item_file in tqdm(item_files, desc="Processing items"):
        item_id = item_file.stem
        output_item_file = items_path / f"{item_id}.json"
        
        # Skip if already processed and not overwriting
        if output_item_file.exists() and not overwrite:
            logger.debug(f"Skipping {item_id} (already processed)")
            skipped_items += 1
            continue
        
        try:
            # Load summary item
            with open(item_file, 'r') as f:
                summary_data = json.load(f)
            
            # Summaries are in documents[0]['summaries']
            documents = summary_data.get("documents", [])
            if not documents:
                logger.warning(f"No documents found for {item_id}")
                failed_items += 1
                continue
                
            summaries = documents[0].get("summaries", [])
            if not summaries:
                logger.warning(f"No summaries found for {item_id}")
                failed_items += 1
                continue
            
            # Extract summarization metadata from item
            optional_summary_length = None
            strategy_function = None
            summary_content_type = None
            step_k_inputs = None
            
            try:
                item_metadata = documents[0].get("metadata", {}).get("item_experiment_metadata", {})
                
                # From template_vars
                template_vars = item_metadata.get("template_vars", {})
                optional_summary_length = template_vars.get("optional_summary_length")
                
                # From item_experiment_metadata
                strategy_function = item_metadata.get("strategy_function")
                summary_content_type = item_metadata.get("summary_content_type")  
                step_k_inputs = item_metadata.get("step_k_inputs")
                
            except Exception as e:
                logger.debug(f"Could not extract summarization metadata for {item_id}: {e}")
            
            # Select summary text based on range
            selected_text, selected_indices = select_summary_text(summaries, range_spec)
            
            if not selected_text.strip():
                logger.warning(f"No text selected for {item_id} with range {range_spec}")
                failed_items += 1
                continue
            
            # Extract and deduplicate entities from summary
            original_summary_entities = extract_entities_with_spacy(selected_text, nlp)
            logger.info(f"Extracted {len(original_summary_entities)} original summary entities, now deduplicating...")
            summary_entities, summary_grouping_info, summary_dedup_response = deduplicate_entities_with_llm(original_summary_entities, model, selected_text)
            logger.info(f"Deduplicated to {len(summary_entities)} canonical summary entities")
            
            # Get source entities (Step 1 - with caching and deduplication)
            original_source_entities, source_entities, source_grouping_info, source_dedup_response = get_source_entities(item_id, dataset_name, nlp, model=model, add_reveal=add_reveal, reveal_only=reveal_only)
            
            if not source_entities:
                logger.warning(f"No source entities found for {item_id}")
                failed_items += 1
                continue
            
            # Step 2: Entity matching (using hybrid approach)
            matching_usage = {"total_input_tokens": 0, "total_output_tokens": 0, "total_tokens": 0, "total_cost": 0.0}
            
            logger.info(f"Matching {len(summary_entities)} summary entities against {len(source_entities)} source entities for {item_id}")
            
            # Use hybrid matching approach
            intersection, summary_only, raw_responses = match_entities_hybrid(
                summary_entities, source_entities, model, edit_distance_threshold=0.2
            )
            
            # Convert raw_responses dict to list format for backward compatibility
            matching_responses = []
            for summary_entity in summary_entities:
                raw_response = raw_responses.get(summary_entity, "No response recorded")
                
                # Determine if entity was matched
                matched_entity = None
                for sum_ent, src_ent in intersection:
                    if sum_ent == summary_entity:
                        matched_entity = src_ent
                        break
                
                matching_responses.append({
                    "summary_entity": summary_entity,
                    "raw_response": raw_response,
                    "parsed_match": matched_entity
                })
            
            # Count LLM calls: only count entities that went to batch LLM
            llm_leftover_count = len([r for r in raw_responses.values() if not r.startswith("FAST_MATCH") and not r.startswith("FUZZY_MATCH")])
            total_matching_calls += 1 if llm_leftover_count > 0 else 0  # One batch call for all leftovers
            
            # Estimate cost: much lower now due to batching
            estimated_cost = 0.01 if llm_leftover_count > 0 else 0.0  # One batch call cost
            matching_usage["total_cost"] += estimated_cost
            
            # Log efficiency improvement
            fast_matches = len([r for r in raw_responses.values() if r.startswith("FAST_MATCH") or r.startswith("FUZZY_MATCH")])
            total_entities = len(summary_entities)
            fast_match_fraction = fast_matches / total_entities if total_entities > 0 else 0.0
            
            logger.info(f"Efficiency: {fast_matches}/{total_entities} entities matched via fast string methods, {llm_leftover_count} needed LLM")
            print(f"📊 String matching efficiency for {item_id}: {fast_match_fraction:.1%} ({fast_matches}/{total_entities}) matched without LLM")
            
            # Update overall efficiency tracking
            total_fast_matches += fast_matches
            total_entities_processed += total_entities
            
            # Calculate source-only entities
            matched_source_entities = {match for _, match in intersection}
            source_only = [entity for entity in source_entities if entity not in matched_source_entities]
            
            # Calculate metrics
            metrics = calculate_entity_metrics(intersection, summary_only, source_only)
            
            # Create item result
            item_result = {
                "item_metadata": {
                    "item_id": item_id,
                    "input_type": "summaries",
                    "selected_range": range_spec,
                    "selected_indices": selected_indices,
                    "selected_text_length": len(selected_text),
                    "total_chunks": len(summaries),
                    "optional_summary_length": optional_summary_length,
                    "strategy_function": strategy_function,
                    "summary_content_type": summary_content_type,
                    "step_k_inputs": step_k_inputs,
                    "summarization_model": summarization_model,
                    "summarization_prompt_name": summarization_prompt_name,
                    "evaluation_timestamp": datetime.now().isoformat()
                },
                "evaluation_metadata": {
                    "prompt_name": prompt_name,
                    "model": model,
                    "temperature": 1.0,
                    "max_completion_tokens": 10000,
                    "extraction_method": "spacy",
                    "extraction_model": "en_core_web_lg",
                    "prompts_used": {
                        "entity_deduplication": {
                            "system": "You are an expert at identifying when different entity names refer to the same person, place, or organization in detective stories.",
                            "user": "Here is some context from a document:\n\n{context_text}\n\n---\n\nHere is a list of entities extracted from this document: {entity_list}\n\nBe VERY CONSERVATIVE. Only group entities if they are:\n1. Obvious spelling variations (e.g., \"L'Hommedieu\" and \"L'Hommedieus\")\n2. Clear abbreviations (e.g., \"Dr. Smith\" and \"Smith\" when context shows they're the same)\n3. Formal vs informal versions of the same name (e.g., \"Robert\" and \"Bob\" when context confirms same person)\n\nDO NOT group entities that could be different people or group different 'aliases' of people. When in doubt, keep them separate. Use the context to verify they actually refer to the same entity.\n\nChoose the most complete/formal name as the canonical version. Every entity from the input list must appear exactly once in the variations array."
                        },
                        "entity_matching": {
                            "system": "You are an expert at determining if entity names refer to the same entity, even with slight variations in spelling or formatting.",
                            "user": "I need you to match each of these summary entities: {summary_entities}\n\nAgainst this list of source entities: {source_entities}\n\nFor each summary entity, determine if it matches any source entity (allowing for slight spelling differences, titles like Dr./Mr., abbreviations, etc.). If you find a match, return the source entity name verbatim. If no match, return null.\n\nReturn results as a JSON array with objects containing 'summary_entity' and 'matched_source_entity' fields."
                        }
                    },
                    "command_run": f"python -m ius entity-coverage --input {input_path} --range {range_spec} --model {model}",
                    "processing_time": 0.0,  # TODO: Track actual time
                    "extraction_time": 0.0,
                    "matching_time": 0.0,
                    "usage": {
                        "input_tokens": matching_usage.get("total_input_tokens", 0),
                        "output_tokens": matching_usage.get("total_output_tokens", 0),
                        "total_tokens": matching_usage.get("total_tokens", 0),
                        "input_cost": 0.0,  # TODO: Calculate actual costs
                        "output_cost": 0.0,
                        "total_cost": matching_usage.get("total_cost", 0.0)
                    }
                },
                "source_entities": source_entities,
                "summary_entities": summary_entities,
                "original_source_entities": original_source_entities,
                "original_summary_entities": original_summary_entities,
                "deduplication_info": {
                    "source_grouping": source_grouping_info,
                    "summary_grouping": summary_grouping_info,
                    "source_deduplication_response": source_dedup_response,
                    "summary_deduplication_response": summary_dedup_response
                },
                "entity_analysis": {
                    "intersection": intersection,
                    "summary_only": summary_only,
                    "source_only": source_only,
                    "metrics": metrics
                },
                "matching_metadata": {
                    "matching_calls": len(summary_entities),
                    "matching_usage": matching_usage,
                    "matching_timestamp": datetime.now().isoformat(),
                    "raw_matching_responses": matching_responses
                }
            }
            
            # Save item result
            with open(output_item_file, 'w') as f:
                json.dump(item_result, f, indent=2)
            
            processed_items.append(item_id)
            successful_items += 1
            total_cost += matching_usage["total_cost"]
            
            logger.info(f"Processed {item_id}: {metrics['num_matched_entities']}/{metrics['num_source_entities']} entities matched (recall: {metrics['recall']:.3f})")
            
        except Exception as e:
            logger.error(f"Failed to process {item_id}: {e}")
            failed_items += 1
            continue
    
    # Create/update collection.json
    collection_data = {
        "entity_coverage_evaluation_info": {
            "collection_metadata": collection_metadata,
            "timestamp": datetime.now().isoformat(),
            "items_processed": processed_items,
            "processing_stats": {
                "total_items": len(item_files),
                "successful_items": successful_items,
                "skipped_items": skipped_items,
                "failed_items": failed_items,
                "total_cost": total_cost,
                "total_tokens": total_tokens,
                "total_matching_calls": total_matching_calls
            }
        }
    }
    
    collection_output_file = output_path / "collection.json"
    with open(collection_output_file, 'w') as f:
        json.dump(collection_data, f, indent=2)
    
    logger.info(f"Entity coverage evaluation complete!")
    logger.info(f"Processed {successful_items}/{len(item_files)} items successfully")
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"Total matching calls: {total_matching_calls}")
    logger.info(f"Estimated total cost: ${total_cost:.3f}")
    
    # Print overall string matching efficiency
    if total_entities_processed > 0:
        overall_fast_match_fraction = total_fast_matches / total_entities_processed
        print(f"\n🎯 OVERALL STRING MATCHING EFFICIENCY: {overall_fast_match_fraction:.1%} ({total_fast_matches}/{total_entities_processed}) entities matched without LLM")
        print(f"💰 API Cost Reduction: ~{(1 - total_matching_calls/total_entities_processed)*100:.1f}% fewer LLM calls compared to individual matching")
    
    return str(output_path)

if __name__ == "__main__":
    # Simple test interface - CLI will be implemented separately
    import sys
    if len(sys.argv) < 2:
        print("Usage: python entity_coverage.py <input_path> [range_spec] [model]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    range_spec = sys.argv[2] if len(sys.argv) > 2 else "penultimate"
    model = sys.argv[3] if len(sys.argv) > 3 else "gpt-5-mini"
    
    try:
        output_dir = run_entity_coverage_evaluation(input_path, range_spec, model)
        print(f"Results saved to: {output_dir}")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)