import csv
import cv2
import re
from pathlib import Path
import numpy as np

def main():
    base_dir = Path('train')
    img_dir = base_dir / 'prepared/images'
    out_csv = base_dir / 'metadata.csv'

    records = []
    fieldnames = ['filename', 'split', 'scene_id', 'source_id', 'material', 'lighting', 'shadow_level', 'hard_negative_tags', 'notes']

    for split in ['train', 'val', 'test']:
        split_dir = img_dir / split
        if not split_dir.exists():
            print(f"Directory {split_dir} does not exist, skipping.")
            continue
            
        # Support multiple image extensions
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp', '*.JPG', '*.JPEG', '*.PNG']:
            for img_path in split_dir.glob(ext):
                filename = img_path.name
                
                # Scene ID extraction
                match = re.match(r'^([a-zA-Z\-_]+)', filename)
                scene_id = match.group(1).strip('-_') if match else 'unknown'
                
                img = cv2.imread(str(img_path))
                material = 'asphalt'
                lighting = 'daylight'
                shadow_level = 'none'
                hard_negative_tags = ''
                
                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    mean_brightness = np.mean(gray)
                    if mean_brightness < 80:
                        lighting = 'night'
                    elif mean_brightness > 180:
                        lighting = 'bright'
                    else:
                        lighting = 'daylight'
                        
                    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                    if lap_var > 1000:
                        hard_negative_tags = 'high_texture'
                        
                    if mean_brightness >= 80:
                        dark_ratio = np.sum(gray < 50) / gray.size
                        if dark_ratio > 0.15:
                            shadow_level = 'heavy'
                        elif dark_ratio > 0.05:
                            shadow_level = 'mottled'
                    
                    if lap_var > 1500:
                        material = 'gravel'
                    elif mean_brightness > 150 and lap_var < 500:
                        material = 'cement'
                        
                records.append({
                    'filename': filename,
                    'split': split,
                    'scene_id': scene_id,
                    'source_id': 'auto_extracted',
                    'material': material,
                    'lighting': lighting,
                    'shadow_level': shadow_level,
                    'hard_negative_tags': hard_negative_tags,
                    'notes': ''
                })

    # Deduplicate by filename
    unique_records = {}
    for r in records:
        unique_records[r['filename']] = r
    records = list(unique_records.values())

    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(row)
            
    print(f'Successfully auto-filled {len(records)} entries into metadata.csv')

if __name__ == "__main__":
    main()
