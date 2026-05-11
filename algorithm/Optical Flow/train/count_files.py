import os
from pathlib import Path

def count_files(dir_path, extensions):
    count = 0
    for root, dirs, files in os.walk(dir_path):
        for f in files:
            if any(f.endswith(ext) for ext in extensions):
                count += 1
    return count

def main():
    input_dir = Path('input')
    
    # Folders to check
    folders = ['asphalt', 'Belgian', 'Forest', 'Gravel', 'Via Appia Antica', 'cement']
    
    total_images = 0
    print("=== Image Counts ===")
    for folder in folders:
        path = input_dir / folder
        if path.exists():
            cnt = count_files(path, ['.jpg', '.png', '.jfif'])
            print(f"{folder}: {cnt} images")
            total_images += cnt
            
    # Also count images in input root
    root_images = count_files(input_dir, ['.jpg', '.png', '.jfif']) - count_files(input_dir / 'raw', ['.jpg', '.png', '.jfif']) - count_files(input_dir / 'raw_candidates', ['.jpg', '.png', '.jfif'])
    # Subtract files in subfolders
    for folder in folders:
        path = input_dir / folder
        if path.exists():
            root_images -= count_files(path, ['.jpg', '.png', '.jfif'])
            
    print(f"Root (Backgrounds): {root_images} images")
    total_images += root_images
    
    print(f"\nTotal Images (excluding raw pools): {total_images}")
    
    print("\n=== Label Counts ===")
    labels_dir = input_dir / 'road' / 'labels'
    total_labels = count_files(labels_dir, ['.txt'])
    print(f"Total Labels (.txt): {total_labels}")
    
    # Break down labels
    for folder in folders:
        path = labels_dir / folder
        if path.exists():
            cnt = count_files(path, ['.txt'])
            print(f"Labels in {folder}: {cnt}")

if __name__ == "__main__":
    main()
