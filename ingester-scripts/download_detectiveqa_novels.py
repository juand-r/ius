#!/usr/bin/env python3
"""
Download all novel files from DetectiveQA dataset novel_data_en directory.
"""

from huggingface_hub import hf_hub_download, list_repo_files
import os
from pathlib import Path
from tqdm import tqdm

def download_all_novels():
    """Download all novel files from the DetectiveQA dataset."""
    
    repo_id = "Phospheneser/DetectiveQA"
    novel_dir = "novel_data_en"
    local_dir = Path("data-source/detectiveqa")
    
    # Create local directory
    local_dir.mkdir(exist_ok=True)
    
    print(f"📚 Downloading DetectiveQA novels to {local_dir}/")
    
    try:
        # List all files in the repository
        print("🔍 Fetching file list...")
        all_files = list_repo_files(repo_id, repo_type="dataset")
        
        # Filter for novel files in novel_data_en directory
        novel_files = [f for f in all_files if f.startswith(f"{novel_dir}/") and f.endswith(".txt")]
        
        print(f"📋 Found {len(novel_files)} novel files to download")
        
        # Download each file
        for file_path in tqdm(novel_files, desc="Downloading"):
            filename = Path(file_path).name
            local_path = local_dir / filename
            
            # Skip if already exists
            if local_path.exists():
                print(f"⏭️  Skipping {filename} (already exists)")
                continue
            
            try:
                # Download the file
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=file_path,
                    repo_type="dataset",
                    local_dir=local_dir,
                    local_dir_use_symlinks=False
                )
                
                # Move from nested structure to flat structure
                if Path(downloaded_path).exists():
                    final_path = local_dir / filename
                    if not final_path.exists():
                        Path(downloaded_path).rename(final_path)
                    
                    # Clean up nested directory
                    nested_dir = local_dir / novel_dir
                    if nested_dir.exists() and not any(nested_dir.iterdir()):
                        nested_dir.rmdir()
                
                print(f"✅ Downloaded: {filename}")
                
            except Exception as e:
                print(f"❌ Failed to download {filename}: {e}")
        
        print(f"\n🎉 Download complete! Files saved to {local_dir}/")
        
        # Show summary
        downloaded_files = list(local_dir.glob("*.txt"))
        print(f"📊 Total files downloaded: {len(downloaded_files)}")
        
        # Show file sizes
        total_size = sum(f.stat().st_size for f in downloaded_files)
        print(f"💾 Total size: {total_size / (1024*1024):.1f} MB")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have huggingface_hub installed: pip install huggingface_hub")

if __name__ == "__main__":
    download_all_novels()
