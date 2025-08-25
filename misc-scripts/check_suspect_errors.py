#!/usr/bin/env python3
"""
Standalone script to check for suspect errors/hallucinations in BMDS stories using GPT-5.

This script:
1. Reads story_id and "Gold suspects pre-reveal" from Google Sheets
2. Loads corresponding BMDS chunks JSON files 
3. Concatenates all chunks except the last one
4. Prompts GPT-5 to identify errors in the suspect list
5. Saves results to CSV
"""

import json
import csv
import pandas as pd
from pathlib import Path
import logging
import sys
import os

# Add the project root to Python path to import ius modules
sys.path.insert(0, str(Path(__file__).parent))

from ius.utils import call_llm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Google Sheet configuration
SHEET_ID = "1awnPbTUjIfVOqqhd8vWXQm8iwPXRMXJ4D1-MWfwLNwM"
GID = "0"

# BMDS chunks directory
BMDS_CHUNKS_DIR = Path("outputs/chunks/bmds_fixed_size2_8000/items")

def load_google_sheet_data():
    """Load story_id and suspects data from Google Sheets."""
    try:
        csv_url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"
        logger.info("Loading data from Google Sheets...")
        
        df = pd.read_csv(csv_url, keep_default_na=False, na_values=[''])
        
        # Convert potential NaN values to empty strings and filter out empty rows
        df['story_id'] = df['story_id'].astype(str).str.strip()
        df['Gold suspects pre-reveal'] = df['Gold suspects pre-reveal'].astype(str).str.strip()
        
        # Filter out rows where story_id or suspects are empty/nan
        df = df[(df['story_id'] != '') & (df['story_id'] != 'nan') & 
                (df['Gold suspects pre-reveal'] != '') & (df['Gold suspects pre-reveal'] != 'nan')]
        
        logger.info(f"✅ Loaded {len(df)} valid rows from Google Sheets")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load data from Google Sheets: {e}")
        return None

def load_bmds_chunks(story_id):
    """Load and concatenate BMDS chunks for a story (all but the last chunk)."""
    try:
        json_file = BMDS_CHUNKS_DIR / f"{story_id}.json"
        
        if not json_file.exists():
            logger.warning(f"BMDS chunks file not found: {json_file}")
            return None
            
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Get chunks from documents
        documents = data.get('documents', [])
        if not documents:
            logger.warning(f"No documents found in {story_id}")
            return None
            
        # Concatenate all chunks except the last one
        all_chunks = []
        for doc in documents:
            chunks = doc.get('chunks', [])
            if chunks:
                # Add all but the last chunk
                all_chunks.extend(chunks[:-1])
        
        if not all_chunks:
            logger.warning(f"No chunks found (or only one chunk) in {story_id}")
            return None
            
        # Concatenate the chunk texts
        story_text = '\n\n'.join(all_chunks)
        logger.debug(f"Concatenated {len(all_chunks)} chunks for {story_id} ({len(story_text)} characters)")
        
        return story_text
        
    except Exception as e:
        logger.error(f"Failed to load chunks for {story_id}: {e}")
        return None

def clean_csv_content(text):
    """Clean text content to prevent CSV parsing issues."""
    if not text:
        return text
    
    # Convert to string if not already
    text = str(text)
    
    # Prevent formula injection by prefixing with space if starts with =, +, -, @
    if text.startswith(('=', '+', '-', '@')):
        text = ' ' + text
    
    # Replace problematic characters that might cause encoding issues
    # Replace smart quotes and other Unicode characters that might cause issues
    replacements = {
        '"': '"',  # Left double quotation mark
        '"': '"',  # Right double quotation mark
        ''': "'",  # Left single quotation mark  
        ''': "'",  # Right single quotation mark
        '—': '-',  # Em dash
        '–': '-',  # En dash
        '…': '...',  # Horizontal ellipsis
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove any null bytes that might cause issues
    text = text.replace('\x00', '')
    
    return text

def check_suspects_with_gpt5(suspects, story_text):
    """Use GP:iT-5 to check for errors in the suspect list."""
    try:
        # Use template-based prompts like the whodunit code
        user_template = """You will help me figure out if there are any errors or hallucinations with this list of suspects:

{suspects}

in this story:

{story_text}

ONLY respond with a short list of important mistakes like the wrong first name:"""

        # Call GPT-5 with the specified parameters
        result = call_llm(
            text="",  # Empty text since we're using system_and_user_prompt
            model="gpt-5",
            system_and_user_prompt={
                "system": "You are an expert at identifying errors and inconsistencies in text.",
                "user": user_template
            },
            template_vars={
                "suspects": suspects,
                "story_text": story_text
            },
            temperature=1.0,
            max_completion_tokens=100000
        )
        
        if result.get('error'):
            logger.error(f"GPT-5 call failed: {result['error']}")
            return None
            
        response_text = result.get('response', '').strip()
        logger.debug(f"GPT-5 response length: {len(response_text)} characters")
        
        return response_text
        
    except Exception as e:
        logger.error(f"Failed to call GPT-5: {e}")
        return None

def main():
    """Main function."""
    logger.info("🔍 Starting suspect error checking script")
    
    # Check if BMDS chunks directory exists
    if not BMDS_CHUNKS_DIR.exists():
        logger.error(f"BMDS chunks directory not found: {BMDS_CHUNKS_DIR}")
        return 1
    
    # Load data from Google Sheets
    df = load_google_sheet_data()
    if df is None or df.empty:
        logger.error("No data loaded from Google Sheets")
        return 1
    
    # Prepare results
    results = []
    total_items = len(df)
    processed = 0
    skipped = 0
    
    logger.info(f"📊 Processing {total_items} stories...")
    
    for idx, row in df.iterrows():
        story_id = str(row['story_id']).strip()
        suspects = str(row['Gold suspects pre-reveal']).strip()
        
        logger.info(f"[{idx+1}/{total_items}] Processing {story_id}...")
        
        # Load BMDS chunks
        story_text = load_bmds_chunks(story_id)
        if story_text is None:
            logger.warning(f"⏭️  Skipping {story_id} - could not load chunks")
            skipped += 1
            continue
        
        # Check suspects with GPT-5
        gpt5_response = check_suspects_with_gpt5(suspects, story_text)
        print(gpt5_response)
        if gpt5_response is None:
            logger.warning(f"⏭️  Skipping {story_id} - GPT-5 call failed")
            skipped += 1
            continue
        
        # Clean the response to prevent CSV issues
        cleaned_response = clean_csv_content(gpt5_response)
        
        # Add to results
        results.append({
            'story_id': story_id,
            'gpt_5_response': cleaned_response
        })
        
        processed += 1
        logger.info(f"✅ Processed {story_id} ({processed}/{total_items})")
    
    # Save results to CSV
    if results:
        output_file = "suspect_error_check_results.csv"
        with open(output_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
            fieldnames = ['story_id', 'gpt_5_response']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        
        logger.info(f"💾 Results saved to {output_file}")
        logger.info(f"📊 Final stats: {processed} processed, {skipped} skipped, {len(results)} saved")
    else:
        logger.warning("No results to save")
    
    return 0

if __name__ == "__main__":
    exit(main())