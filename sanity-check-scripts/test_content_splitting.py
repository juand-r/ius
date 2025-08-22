#!/usr/bin/env python3
"""
Sanity check script to compare DetectiveQA dataset versions:
- datasets/detectiveqaGG: newer version without line numbers, with reveal separated
- datasets/detectiveqa: older version with line numbers, everything in content
"""

import json
import re
from pathlib import Path

def test_content_splitting():
    """Compare the two dataset versions to verify content splitting correctness."""
    
    old_dir = Path("datasets/detectiveqa/items")
    new_dir = Path("datasets/detectiveqaGG/items")
    
    if not old_dir.exists():
        print(f"❌ Old dataset directory not found: {old_dir}")
        return
    
    if not new_dir.exists():
        print(f"❌ New dataset directory not found: {new_dir}")
        return
    
    old_files = list(old_dir.glob("*.json"))
    new_files = list(new_dir.glob("*.json"))
    
    print(f"📚 Found {len(old_files)} old files and {len(new_files)} new files")
    
    # Get common files
    old_ids = {f.stem for f in old_files}
    new_ids = {f.stem for f in new_files}
    common_ids = old_ids.intersection(new_ids)
    
    print(f"🔍 Testing {len(common_ids)} common files...")
    
    issues = []
    successes = 0
    
    for novel_id in sorted(common_ids):
        try:
            # Load both versions
            with open(old_dir / f"{novel_id}.json", 'r') as f:
                old_data = json.load(f)
            
            with open(new_dir / f"{novel_id}.json", 'r') as f:
                new_data = json.load(f)
            
            # Get answer position from new version
            answer_position = new_data['documents'][0]['metadata']['questions'][0]['answer_position']
            
            # Get content from both versions
            old_content = old_data['documents'][0]['content']
            new_content = new_data['documents'][0]['content']
            
            # Split old content and find the line starting with [answer_position]
            old_lines = old_content.split('\n')
            target_prefix = f"[{answer_position}]"
            orig_line = None
            
            for line in old_lines:
                if line.startswith(target_prefix):
                    orig_line = line
                    break
            
            if orig_line is None:
                issues.append({
                    'novel_id': novel_id,
                    'error': f"Could not find line starting with '{target_prefix}' in old content"
                })
                continue
            
            # Get reveal segment from new version
            reveal_segment = new_data['documents'][0]['metadata']['detection']['reveal_segment']
            reveal_lines = reveal_segment.split('\n')
            
            # Test 1: Compare orig_line with first line of reveal_segment
            orig_line_cleaned = re.sub(r'^\[\d+\]\s*', '', orig_line)
            first_reveal_line = reveal_lines[0] if reveal_lines else ""
            
            test1_pass = (first_reveal_line == orig_line_cleaned)
            
            # Test 2: Compare lines at answer_position-2 index in both datasets
            test2_pass = True
            test2_error = None
            
            if answer_position >= 2:  # Make sure we don't go negative
                # Find line [answer_position-1] in old content
                target_prefix_minus1 = f"[{answer_position-1}]"
                orig_line_minus1 = None
                
                for line in old_lines:
                    if line.startswith(target_prefix_minus1):
                        orig_line_minus1 = line
                        break
                
                if orig_line_minus1 is None:
                    test2_pass = False
                    test2_error = f"Could not find line starting with '{target_prefix_minus1}' in old content"
                else:
                    # Get corresponding line from new content at index answer_position-2
                    new_lines = new_content.split('\n')
                    if answer_position - 2 >= len(new_lines):
                        test2_pass = False
                        test2_error = f"answer_position-2 ({answer_position-2}) exceeds new content lines ({len(new_lines)})"
                    else:
                        new_line_minus1 = new_lines[answer_position - 2]
                        orig_line_minus1_cleaned = re.sub(r'^\[\d+\]\s*', '', orig_line_minus1)
                        test2_pass = (new_line_minus1 == orig_line_minus1_cleaned)
                        
                        if not test2_pass:
                            test2_error = f"Line mismatch at position {answer_position-1}"
            else:
                test2_pass = False
                test2_error = f"answer_position ({answer_position}) < 2, cannot test previous line"
            
            # Test 3: Check that all clue_position values are < answer_position
            test3_pass = True
            test3_error = None
            
            questions = new_data['documents'][0]['metadata']['questions']
            for i, question in enumerate(questions):
                clue_positions = question.get('clue_position', [])
                if clue_positions:  # Only check if clue_position exists and is not empty
                    invalid_clues = [pos for pos in clue_positions if pos >= answer_position]
                    if invalid_clues:
                        test3_pass = False
                        test3_error = f"Question {i+1} has clue_position {invalid_clues} >= answer_position ({answer_position})"
                        break
            
            # Overall result
            if test1_pass and test2_pass and test3_pass:
                successes += 1
                print(f"✅ Novel {novel_id}: All tests pass")
            else:
                error_details = []
                if not test1_pass:
                    error_details.append(f"Test 1 failed: reveal_segment first line != orig_line[{answer_position}]")
                if not test2_pass:
                    error_details.append(f"Test 2 failed: {test2_error}")
                if not test3_pass:
                    error_details.append(f"Test 3 failed: {test3_error}")
                
                issues.append({
                    'novel_id': novel_id,
                    'error': 'Test failure',
                    'details': error_details,
                    'answer_position': answer_position,
                    'test1_pass': test1_pass,
                    'test2_pass': test2_pass,
                    'test3_pass': test3_pass,
                    'orig_line_preview': orig_line_cleaned[:100] + ('...' if len(orig_line_cleaned) > 100 else ''),
                    'reveal_first_preview': first_reveal_line[:100] + ('...' if len(first_reveal_line) > 100 else ''),
                })
                print(f"❌ Novel {novel_id}: {'; '.join(error_details)}")
                
        except Exception as e:
            issues.append({
                'novel_id': novel_id,
                'error': f"Exception during processing: {str(e)}"
            })
            print(f"❌ Novel {novel_id}: {str(e)}")
    
    # Summary report
    print(f"\n📊 SUMMARY:")
    print(f"Successfully verified: {successes}")
    print(f"Issues found: {len(issues)}")
    
    if issues:
        print(f"\n🔍 DETAILED ISSUES:")
        for issue in issues[:5]:  # Show first 5 issues
            novel_id = issue['novel_id']
            error = issue['error']
            
            if error == 'Test failure':
                print(f"\nNovel {novel_id} (answer_position: {issue['answer_position']}):")
                for detail in issue['details']:
                    print(f"  {detail}")
                if not issue['test1_pass']:
                    print(f"    Original line: {issue['orig_line_preview']}")
                    print(f"    Reveal first:  {issue['reveal_first_preview']}")
            else:
                print(f"\nNovel {novel_id}: {error}")
        
        if len(issues) > 5:
            print(f"\n... and {len(issues) - 5} more issues")
    
    return successes, len(issues)

if __name__ == "__main__":
    test_content_splitting()