import os
import cv2
import glob
import shutil
from pathlib import Path
import numpy as np

def main():
    raw_dir = Path('input/raw')
    output_dir = Path('input/raw_candidates')
    
    # Clean up previous run
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {raw_dir} for images...")
    img_paths = list(raw_dir.glob('*.jpg')) + list(raw_dir.glob('*.png'))
    print(f"Found {len(img_paths)} images.")

    analyzed = []
    for p in img_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Extract scene_id for diversity check later
        name = p.name
        if '-' in name:
            scene_id = name.split('-')[0]
        else:
            scene_id = 'unknown'
            
        analyzed.append({
            'path': p,
            'brightness': brightness,
            'texture': laplacian,
            'scene_id': scene_id
        })

    print(f"Analysis complete on {len(analyzed)} images.")

    # We want to select around 300 images total.
    # Let's pick:
    # 1. Top 100 highest texture images (Gravel, high texture asphalt)
    # 2. Top 100 darkest images (Night, heavy shadow)
    # 3. Top 100 images with high edge contrast or just diverse scenes
    
    selected_paths = set()
    
    # 1. High Texture (Sort by texture descending)
    analyzed.sort(key=lambda x: x['texture'], reverse=True)
    count = 0
    for item in analyzed:
        if count >= 100:
            break
        if item['path'] not in selected_paths:
            selected_paths.add(item['path'])
            count += 1
            
    print(f"Selected {count} high-texture candidates.")

    # 2. Low Light (Sort by brightness ascending)
    analyzed.sort(key=lambda x: x['brightness'])
    count = 0
    for item in analyzed:
        if count >= 100:
            break
        if item['path'] not in selected_paths:
            selected_paths.add(item['path'])
            count += 1
            
    print(f"Selected {count} additional low-light candidates.")

    # 3. Fill up to ~300 with diverse scenes or middle texture
    # Let's take from the middle of the texture list
    analyzed.sort(key=lambda x: x['texture'], reverse=True)
    middle_index = len(analyzed) // 2
    count = 0
    # Search outwards from middle
    i = 0
    while len(selected_paths) < 300 and i < len(analyzed):
        idx = middle_index + i if i % 2 == 0 else middle_index - i
        if 0 <= idx < len(analyzed):
            item = analyzed[idx]
            if item['path'] not in selected_paths:
                selected_paths.add(item['path'])
                count += 1
        i += 1
        
    print(f"Selected {count} additional middle-texture candidates to fill up.")

    # Copy selected files
    for p in selected_paths:
        shutil.copy(p, output_dir / p.name)

    print(f"\nAuto-Mining Complete!")
    print(f"Total candidates selected and copied to {output_dir}: {len(selected_paths)}")
    print("Please review these candidates and proceed to labeling.")

if __name__ == "__main__":
    main()
