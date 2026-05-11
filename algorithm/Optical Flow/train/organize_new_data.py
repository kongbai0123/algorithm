import os
import shutil
from pathlib import Path

def main():
    labels_dir = Path('input/road/labels/asphalt')
    raw_dir = Path('input/raw')
    output_images_dir = Path('input/asphalt')
    
    output_images_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning labels in {labels_dir}...")
    label_files = list(labels_dir.glob('*.txt'))
    
    copied_count = 0
    
    for label_file in label_files:
        name = label_file.stem
        # Filter for new files (they contain a hyphen)
        if '-' in name:
            img_name = f"{name}.jpg"
            src_img = raw_dir / img_name
            dst_img = output_images_dir / img_name
            
            if src_img.exists():
                shutil.copy(src_img, dst_img)
                copied_count += 1
                print(f"Copied {img_name} to {output_images_dir}")
            else:
                print(f"Warning: Image {src_img} not found!")
                
    print(f"\nDone! Copied {copied_count} images to {output_images_dir}.")

if __name__ == "__main__":
    main()
