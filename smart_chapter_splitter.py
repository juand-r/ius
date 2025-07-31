#!/usr/bin/env python3
"""
Smart Chapter Splitter with LLM-assisted pattern detection.

Uses table of contents to generate custom detection logic for each book,
with confidence-based output directories.
"""

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import openai
import os


class SmartChapterSplitter:
    """Smart chapter splitter with LLM-assisted detection."""
    
    def __init__(self, output_base_dir: Path = Path("data-source/smart_chapters")):
        self.output_base_dir = output_base_dir
        self.confidence_dirs = {
            'confident': output_base_dir / 'chapters-confident',
            'likely': output_base_dir / 'chapters-likely', 
            'unsure': output_base_dir / 'chapters-unsure',
            'unknown': output_base_dir / 'chapters-unknown'
        }
        self.codes_dir = output_base_dir / 'codes'
        
        # Create directories
        for dir_path in self.confidence_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        self.codes_dir.mkdir(parents=True, exist_ok=True)
            
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = openai.OpenAI(api_key=api_key)
    
    def extract_table_of_contents(self, text: str) -> Optional[str]:
        """
        Extract table of contents from book text.
        
        Looks for common TOC patterns and extracts relevant section.
        """
        lines = text.split('\n')
        
        # Look for common TOC indicators
        toc_indicators = [
            'contents',
            'table of contents', 
            'chapter',
            'part one',
            'part i',
            'prologue'
        ]
        
        toc_start = None
        toc_end = None
        
        # Find TOC start
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            if any(indicator in line_lower for indicator in toc_indicators):
                if len(line.strip()) < 50:  # TOC headers are usually short
                    toc_start = i
                    break
        
        if toc_start is None:
            return None
            
        # Find TOC end (look for start of actual content)
        content_indicators = [
            'chapter 1',
            'ONE',
            'one',
            'CHAPTER 1'
            'chapter one', 
            'prologue',
            'introduction',
            'preface'
        ]
        
        # Look for where actual content starts (usually 50+ lines after TOC start)
        for i in range(toc_start + 10, min(len(lines), toc_start + 200)):
            line = lines[i].strip()
            
            # Look for paragraph-like content (long lines)
            if len(line) > 100 and '.' in line:
                # This might be actual content
                toc_end = i
                break
                
            # Or look for explicit content indicators
            line_lower = line.lower()
            if any(indicator in line_lower for indicator in content_indicators):
                if len(line) > 50:  # Actual content, not just TOC entry
                    toc_end = i
                    break
        
        if toc_end is None:
            toc_end = min(len(lines), toc_start + 100)  # Default to 100 lines
            
        toc_text = '\n'.join(lines[toc_start:toc_end])
        return toc_text.strip() if toc_text.strip() else None
    
    def generate_detection_code(self, book_name: str, book_beginning: str) -> Optional[str]:
        """
        Use LLM to generate custom chapter detection code for this book.
        """
        prompt = f"""I need to split a book called "{book_name}" into chapters. Here are the first 100 lines of the book:

```
{book_beginning}
```

Please write Python code that will detect chapter boundaries in this specific book. The code should:

1. Take the full book text as input (variable name: `text`)
2. Return a list of tuples: [(start_line, chapter_title), ...]
3. Be specific to the patterns I can see in this book's structure
4. Avoid matching index entries, bibliography, or other non-chapter content
5. Focus on the main narrative chapters
6. Note splits are likely to happen on newlines, but they might not.

Return ONLY the Python function code, no explanations. The function should be named `detect_chapters(text: str) -> List[Tuple[int, str]]`.

IMPORTANT: the splits should not remove any text from the book, only split it into chapters. 
Do not remove the first part of the book, or the last part of the book.

MANDATORY REQUIREMENTS:
1. Your first chapter MUST start at line 0 (the very beginning of the book)
2. Your last chapter MUST end at the last line of the book  
3. NO content should be skipped or lost between chapters
4. If a chapter begins with e.g., "2. Independence Day", then this string should be the start of the chapter

The function should preserve EVERY line of the original book across all chapters.

Example format:
```python
def detect_chapters(text: str) -> List[Tuple[int, str]]:
    lines = text.split('\\n')
    chapters = []
    
    # ALWAYS start with line 0 to preserve front matter, e.g.:
    chapters.append((0, "Front Matter"))  # Everything before first real chapter
    
    # Then find your actual chapters
    for i, line in enumerate(lines):
        if matches_chapter_pattern(line):
            chapters.append((i, line.strip()))
    
    return chapters
```

CRITICAL: Your function MUST return (0, "Front Matter") as the first chapter to preserve the beginning of the book."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a Python code generator specialized in text processing. Return only clean, executable Python functions."},
                    {"role": "user", "content": prompt}
                ],
            )
            
            code = response.choices[0].message.content.strip()
            
            # Clean up the code (remove markdown markers if present)
            if code.startswith('```python'):
                code = code[9:]
            if code.endswith('```'):
                code = code[:-3]
                
            return code.strip()
            
        except Exception as e:
            print(f"❌ Error generating detection code: {e}")
            return None
    
    def execute_detection_code(self, code: str, text: str) -> Optional[List[Tuple[int, str]]]:
        """
        Safely execute the generated detection code.
        """
        try:
            # Create a safe execution environment
            namespace = {
                're': re,
                'List': List,
                'Tuple': Tuple
            }
            
            # Execute the code
            exec(code, namespace)
            
            # Call the function
            if 'detect_chapters' in namespace:
                result = namespace['detect_chapters'](text)
                return result
            else:
                print("❌ Generated code doesn't contain detect_chapters function")
                return None
                
        except Exception as e:
            print(f"❌ Error executing detection code: {e}")
            return None
    
    def assess_chapter_quality(self, chapters: List[Tuple[int, str]], text: str) -> str:
        """
        Assess the quality of chapter detection and return confidence level.
        
        Returns: 'confident', 'likely', 'unsure', or 'unknown'
        """
        if not chapters or len(chapters) < 2:
            return 'unknown'
        
        lines = text.split('\n')
        chapter_lengths = []
        
        # CRITICAL: Check for content preservation
        if chapters[0][0] != 0:
            print(f"   ⚠️  WARNING: First chapter doesn't start at line 0 (starts at {chapters[0][0]})")
            print(f"   ⚠️  This means {chapters[0][0]} lines of content will be LOST!")
            return 'unknown'  # Automatic failure for content loss
        
        # Calculate chapter lengths and check for gaps
        total_lines_covered = 0
        for i, (start_line, title) in enumerate(chapters):
            end_line = chapters[i + 1][0] if i + 1 < len(chapters) else len(lines)
            chapter_text = '\n'.join(lines[start_line:end_line])
            chapter_lengths.append(len(chapter_text.split()))
            total_lines_covered += (end_line - start_line)
        
        # Verify all content is preserved
        if total_lines_covered != len(lines):
            print(f"   ⚠️  WARNING: Content loss detected!")
            print(f"   ⚠️  Original: {len(lines)} lines, Covered: {total_lines_covered} lines")
            return 'unknown'
        
        # Quality checks
        checks = {
            'reasonable_count': 3 <= len(chapters) <= 100,
            'reasonable_lengths': all(500 <= length <= 50000 for length in chapter_lengths),
            'consistent_lengths': max(chapter_lengths) / min(chapter_lengths) < 20,
            'no_tiny_chapters': all(length > 100 for length in chapter_lengths),
            'titles_look_good': all(len(title.strip()) > 0 and len(title) < 100 for _, title in chapters),
            'content_preserved': True  # We already validated this above
        }
        
        passed_checks = sum(checks.values())
        
        # Determine confidence
        if passed_checks >= 5:
            return 'confident'
        elif passed_checks >= 4:
            return 'likely'
        elif passed_checks >= 3:
            return 'unsure'
        else:
            return 'unknown'
    
    def split_into_chapters(self, chapters: List[Tuple[int, str]], text: str, book_name: str, confidence: str) -> bool:
        """
        Split text into chapter files and save to appropriate confidence directory.
        """
        lines = text.split('\n')
        output_dir = self.confidence_dirs[confidence] / book_name
        output_dir.mkdir(exist_ok=True)
        
        if confidence == 'unknown':
            # Save as single file for manual review
            with open(output_dir / f"{book_name}_full.txt", 'w', encoding='utf-8') as f:
                f.write(text)
            return True
        
        # Save chapters
        for i, (start_line, title) in enumerate(chapters):
            end_line = chapters[i + 1][0] if i + 1 < len(chapters) else len(lines)
            chapter_text = '\n'.join(lines[start_line:end_line])
            
            # Clean title for filename
            clean_title = re.sub(r'[^\w\s-]', '', title).strip()
            clean_title = re.sub(r'\s+', '_', clean_title)
            
            filename = f"chapter_{i:02d}_{clean_title}.txt"
            
            with open(output_dir / filename, 'w', encoding='utf-8') as f:
                f.write(chapter_text)
        
        # Save metadata
        metadata = {
            'book_name': book_name,
            'confidence': confidence,
            'num_chapters': len(chapters),
            'chapter_titles': [title for _, title in chapters],
            'detection_method': 'llm_generated'
        }
        
        with open(output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        return True
    
    def process_book(self, book_path: Path) -> Dict[str, Any]:
        """
        Process a single book with the smart chapter splitting approach.
        """
        book_name = book_path.stem
        print(f"\n📖 Processing: {book_name}")
        
        try:
            # Read book
            with open(book_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            print(f"   📄 Book length: {len(text):,} chars")
            
            # Generate detection code (use first 100 lines)
            lines = text.split('\n')
            book_beginning = '\n'.join(lines[:100])
            detection_code = self.generate_detection_code(book_name, book_beginning)
            if not detection_code:
                print("   ❌ Could not generate detection code")
                return {'book': book_name, 'status': 'failed', 'reason': 'no_code'}
            
            print("   🤖 Generated custom detection code")
            
            # Save the generated code
            code_filename = f"{book_name}_detection.py"
            code_path = self.codes_dir / code_filename
            with open(code_path, 'w', encoding='utf-8') as f:
                f.write(f"# Generated detection code for: {book_name}\n")
                f.write(f"# Book: {book_name}\n\n")
                f.write(detection_code)
            print(f"   💾 Saved detection code to: {code_filename}")
            
            # Execute detection
            chapters = self.execute_detection_code(detection_code, text)
            if not chapters:
                print("   ❌ Detection code failed to find chapters")
                return {'book': book_name, 'status': 'failed', 'reason': 'detection_failed'}
            
            print(f"   📚 Found {len(chapters)} chapters")
            
            # Assess quality
            confidence = self.assess_chapter_quality(chapters, text)
            print(f"   🎯 Confidence: {confidence}")
            
            # Split and save
            success = self.split_into_chapters(chapters, text, book_name, confidence)
            
            if success:
                return {
                    'book': book_name,
                    'status': 'success',
                    'confidence': confidence,
                    'num_chapters': len(chapters),
                    'book_beginning_length': len(book_beginning)
                }
            else:
                return {'book': book_name, 'status': 'failed', 'reason': 'save_failed'}
                
        except Exception as e:
            print(f"   ❌ Error processing {book_name}: {e}")
            return {'book': book_name, 'status': 'error', 'reason': str(e)}


def main():
    parser = argparse.ArgumentParser(description="Smart chapter splitter with LLM assistance")
    parser.add_argument("book_path", help="Path to the book text file")
    parser.add_argument("--output-dir", default="data-source/smart_chapters", help="Output directory")
    
    args = parser.parse_args()
    
    book_path = Path(args.book_path)
    if not book_path.exists():
        print(f"❌ Book file not found: {book_path}")
        return 1
    
    splitter = SmartChapterSplitter(Path(args.output_dir))
    result = splitter.process_book(book_path)
    
    print(f"\n📊 Result: {result}")
    
    return 0


if __name__ == "__main__":
    exit(main())