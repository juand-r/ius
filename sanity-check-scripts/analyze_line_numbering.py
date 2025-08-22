#!/usr/bin/env python3
"""
Script to analyze line numbering scheme in DetectiveQA dataset.
Checks if the [number] format follows expected patterns.
"""

import json
import re
from pathlib import Path

def analyze_line_numbering(content):
    """
    Analyze the line numbering scheme in content.
    
    Args:
        content: Text content to analyze
        
    Returns:
        dict: Analysis results with issues found
    """
    lines = content.split('\n')
    issues = []
    
    # Pattern to match [number] at the start of a line
    line_number_pattern = r'^\[(\d+)\]'
    
    expected_number = 1
    lines_with_numbers = 0
    lines_without_numbers = 0
    
    for line_idx, line in enumerate(lines):
        match = re.match(line_number_pattern, line)
        
        if match:
            lines_with_numbers += 1
            actual_number = int(match.group(1))
            
            # Check if number matches expected sequence
            if actual_number != expected_number:
                issues.append({
                    'type': 'sequence_gap',
                    'line_idx': line_idx,
                    'expected': expected_number,
                    'actual': actual_number,
                    'line_preview': line[:100] + ('...' if len(line) > 100 else '')
                })
            
            # Check if line index + 1 equals the number in brackets
            if line_idx + 1 != actual_number:
                issues.append({
                    'type': 'index_mismatch',
                    'line_idx': line_idx,
                    'zero_based_idx': line_idx,
                    'one_based_idx': line_idx + 1,
                    'bracket_number': actual_number,
                    'line_preview': line[:100] + ('...' if len(line) > 100 else '')
                })
            
            expected_number = actual_number + 1
        else:
            lines_without_numbers += 1
            # Only report if this is near the beginning (first 10 lines) or if it looks like it should have a number
            if line_idx < 10 or (line.strip() and not line.startswith(' ') and len(line.strip()) > 20):
                issues.append({
                    'type': 'missing_number',
                    'line_idx': line_idx,
                    'line_preview': line[:100] + ('...' if len(line) > 100 else '')
                })
    
    # Check if first numbered line starts with [1]
    first_numbered_line = None
    for line_idx, line in enumerate(lines):
        match = re.match(line_number_pattern, line)
        if match:
            first_numbered_line = int(match.group(1))
            if first_numbered_line != 1:
                issues.append({
                    'type': 'first_line_not_one',
                    'line_idx': line_idx,
                    'first_number': first_numbered_line,
                    'line_preview': line[:100] + ('...' if len(line) > 100 else '')
                })
            break
    
    return {
        'total_lines': len(lines),
        'lines_with_numbers': lines_with_numbers,
        'lines_without_numbers': lines_without_numbers,
        'first_numbered_line': first_numbered_line,
        'expected_last_number': expected_number - 1,
        'issues': issues
    }

def main():
    """Analyze line numbering in all DetectiveQA items."""
    print("🔍 Analyzing line numbering scheme in DetectiveQA dataset...")
    
    items_dir = Path("datasets/detectiveqa/items")
    if not items_dir.exists():
        print(f"❌ Directory not found: {items_dir}")
        return
    
    json_files = list(items_dir.glob("*.json"))
    print(f"📚 Found {len(json_files)} novels to analyze...")
    
    all_issues = []
    novels_with_issues = 0
    total_novels = 0
    
    for json_file in sorted(json_files):
        novel_id = json_file.stem
        total_novels += 1
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            content = data['documents'][0]['content']
            title = data['documents'][0]['metadata'].get('title', 'Unknown')
            
            # Analyze line numbering
            analysis = analyze_line_numbering(content)
            
            if analysis['issues']:
                novels_with_issues += 1
                print(f"⚠️  Novel {novel_id} ({title}): {len(analysis['issues'])} issues")
                
                # Show first few issues
                for i, issue in enumerate(analysis['issues'][:3]):
                    if issue['type'] == 'sequence_gap':
                        print(f"    - Gap in sequence at line {issue['line_idx']}: expected [{issue['expected']}] but found [{issue['actual']}]")
                    elif issue['type'] == 'index_mismatch':
                        print(f"    - Index mismatch at line {issue['line_idx']}: line index {issue['zero_based_idx']} should be [{issue['one_based_idx']}] but found [{issue['bracket_number']}]")
                    elif issue['type'] == 'missing_number':
                        print(f"    - Missing number at line {issue['line_idx']}: {issue['line_preview']}")
                    elif issue['type'] == 'first_line_not_one':
                        print(f"    - First line not [1] at line {issue['line_idx']}: starts with [{issue['first_number']}]")
                
                if len(analysis['issues']) > 3:
                    print(f"    ... and {len(analysis['issues']) - 3} more issues")
                
                all_issues.append({
                    'novel_id': novel_id,
                    'title': title,
                    'analysis': analysis
                })
            else:
                print(f"✅ Novel {novel_id} ({title}): Line numbering is correct")
                
        except Exception as e:
            print(f"❌ Error processing {novel_id}: {e}")
    
    # Summary
    print(f"\n📊 ANALYSIS SUMMARY:")
    print(f"Total novels analyzed: {total_novels}")
    print(f"Novels with line numbering issues: {novels_with_issues}")
    print(f"Novels with correct line numbering: {total_novels - novels_with_issues}")
    
    if novels_with_issues > 0:
        print(f"\n🔍 ISSUE BREAKDOWN:")
        issue_types = {}
        for novel_issues in all_issues:
            for issue in novel_issues['analysis']['issues']:
                issue_type = issue['type']
                issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
        
        for issue_type, count in issue_types.items():
            print(f"  {issue_type}: {count} occurrences")
        
        # Detailed report for the worst offenders
        if len(all_issues) > 0:
            print(f"\n📝 DETAILED REPORT FOR WORST CASES:")
            worst_cases = sorted(all_issues, key=lambda x: len(x['analysis']['issues']), reverse=True)[:3]
            
            for case in worst_cases:
                novel_id = case['novel_id']
                title = case['title']
                issues = case['analysis']['issues']
                stats = case['analysis']
                
                print(f"\nNovel {novel_id} - {title}:")
                print(f"  Total lines: {stats['total_lines']}")
                print(f"  Lines with numbers: {stats['lines_with_numbers']}")
                print(f"  Lines without numbers: {stats['lines_without_numbers']}")
                print(f"  First numbered line: [{stats['first_numbered_line']}]")
                print(f"  Issues: {len(issues)}")
                
                # Show a few representative issues
                for issue in issues[:5]:
                    if issue['type'] == 'sequence_gap':
                        print(f"    - Line {issue['line_idx']}: sequence gap [{issue['expected']}] → [{issue['actual']}]")
                    elif issue['type'] == 'index_mismatch':
                        print(f"    - Line {issue['line_idx']}: index mismatch (should be [{issue['one_based_idx']}], found [{issue['bracket_number']}])")
                    elif issue['type'] == 'missing_number':
                        print(f"    - Line {issue['line_idx']}: missing number")
                    elif issue['type'] == 'first_line_not_one':
                        print(f"    - Line {issue['line_idx']}: starts with [{issue['first_number']}] not [1]")

if __name__ == "__main__":
    main()