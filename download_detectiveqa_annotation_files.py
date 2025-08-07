#!/usr/bin/env python3
"""
Download all annotation JSON files from DetectiveQA dataset directories.
This bypasses the streaming dataset corruption by downloading files directly.
"""

import json
import os
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

def download_annotation_files():
    """Download all JSON files from anno_data_en directories."""
    
    repo_id = "Phospheneser/DetectiveQA"
    output_dir = Path("data-source/detectiveqa-annotations")
    
    # Create output directories
    output_dir.mkdir(exist_ok=True)
    aisup_dir = output_dir / "AIsup_anno"
    human_dir = output_dir / "human_anno"
    aisup_dir.mkdir(exist_ok=True)
    human_dir.mkdir(exist_ok=True)
    
    print("🔍 Downloading DetectiveQA annotation files...")
    print(f"📁 Output directory: {output_dir}")
    
    try:
        # List all files in the repository
        print("📋 Fetching file list...")
        all_files = list_repo_files(repo_id, repo_type="dataset")
        
        # Filter for annotation JSON files
        aisup_files = [f for f in all_files if f.startswith("anno_data_en/AIsup_anno/") and f.endswith(".json")]
        human_files = [f for f in all_files if f.startswith("anno_data_en/human_anno/") and f.endswith(".json")]
        
        print(f"📊 Found {len(aisup_files)} AIsup_anno files")
        print(f"📊 Found {len(human_files)} human_anno files")
        print(f"📊 Total: {len(aisup_files) + len(human_files)} annotation files")
        
        downloaded_count = 0
        failed_count = 0
        
        # Download AIsup_anno files
        print(f"\n📥 Downloading AIsup_anno files...")
        for file_path in tqdm(aisup_files, desc="AIsup_anno"):
            filename = Path(file_path).name
            local_path = aisup_dir / filename
            
            # Skip if already exists
            if local_path.exists():
                continue
            
            try:
                # Download the file
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=file_path,
                    repo_type="dataset",
                    local_dir=output_dir,
                    local_dir_use_symlinks=False
                )
                
                # Move to flat structure
                if Path(downloaded_path).exists():
                    final_path = aisup_dir / filename
                    if not final_path.exists():
                        Path(downloaded_path).rename(final_path)
                
                downloaded_count += 1
                
            except Exception as e:
                print(f"❌ Failed to download {filename}: {e}")
                failed_count += 1
        
        # Download human_anno files
        print(f"\n📥 Downloading human_anno files...")
        for file_path in tqdm(human_files, desc="human_anno"):
            filename = Path(file_path).name
            local_path = human_dir / filename
            
            # Skip if already exists
            if local_path.exists():
                continue
            
            try:
                # Download the file
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=file_path,
                    repo_type="dataset",
                    local_dir=output_dir,
                    local_dir_use_symlinks=False
                )
                
                # Move to flat structure
                if Path(downloaded_path).exists():
                    final_path = human_dir / filename
                    if not final_path.exists():
                        Path(downloaded_path).rename(final_path)
                
                downloaded_count += 1
                
            except Exception as e:
                print(f"❌ Failed to download {filename}: {e}")
                failed_count += 1
        
        # Clean up nested directories
        nested_anno_dir = output_dir / "anno_data_en"
        if nested_anno_dir.exists():
            import shutil
            shutil.rmtree(nested_anno_dir)
        
        print(f"\n🎉 Download complete!")
        print(f"✅ Successfully downloaded: {downloaded_count} files")
        print(f"❌ Failed downloads: {failed_count}")
        
        # Analyze downloaded files
        aisup_files_local = list(aisup_dir.glob("*.json"))
        human_files_local = list(human_dir.glob("*.json"))
        
        print(f"\n📊 Final counts:")
        print(f"   AIsup_anno: {len(aisup_files_local)} files")
        print(f"   human_anno: {len(human_files_local)} files")
        print(f"   Total: {len(aisup_files_local) + len(human_files_local)} files")
        
        # Test loading a few files to check for corruption
        print(f"\n🧪 Testing file integrity...")
        test_files = aisup_files_local[:5] + human_files_local[:5]
        corrupted_files = []
        
        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"✅ {test_file.name}: OK ({len(data.get('questions', []))} questions)")
            except Exception as e:
                print(f"❌ {test_file.name}: CORRUPTED - {e}")
                corrupted_files.append(test_file.name)
        
        if corrupted_files:
            print(f"\n⚠️  Found {len(corrupted_files)} corrupted files in sample")
        else:
            print(f"\n✅ All tested files are valid JSON!")
        
        return len(aisup_files_local) + len(human_files_local), failed_count
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 0, 0

if __name__ == "__main__":
    total_files, failed_files = download_annotation_files()