import csv
import os
from pathlib import Path
from collections import defaultdict

def main():
    base_dir = Path('train')
    metadata_path = base_dir / 'metadata.csv'
    train_img_dir = base_dir / 'prepared/images/train'
    train_lbl_dir = base_dir / 'prepared/labels/train'
    report_path = base_dir / 'reports/scene_downsample_report.csv'
    
    if not metadata_path.exists():
        print("Metadata not found!")
        return
        
    # Read metadata
    records = []
    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
            
    # Filter for train split only
    train_records = [r for r in records if r['split'] == 'train']
    
    # Group by scene_id
    scene_groups = defaultdict(list)
    for r in train_records:
        scene_groups[r['scene_id']].append(r)
        
    report_data = []
    
    for scene_id, items in scene_groups.items():
        original_count = len(items)
        if original_count <= 8:
            report_data.append({
                'scene_id': scene_id,
                'original_count': original_count,
                'kept_count': original_count,
                'removed_count': 0,
                'reason': 'Count <= 8'
            })
            continue
            
        # Priority sorting
        # Rule: failure-derived > hard-negative > rare material > high-shadow > general
        def get_priority(item):
            score = 0
            # 1. hard_negative_tags not empty
            if item.get('hard_negative_tags'):
                score += 10
            # 2. shadow_level = heavy or mottled
            if item.get('shadow_level') in ['heavy', 'mottled']:
                score += 5
            # 3. material = Belgian, gravel, cement (lowercase check just in case)
            mat = item.get('material', '').lower()
            if mat in ['belgian', 'gravel', 'cement']:
                score += 8
            return score
            
        # Sort descending by priority score
        sorted_items = sorted(items, key=get_priority, reverse=True)
        
        kept_items = sorted_items[:8]
        removed_items = sorted_items[8:]
        
        # Delete removed items from disk
        for item in removed_items:
            fname = item['filename']
            img_p = train_img_dir / fname
            lbl_p = train_lbl_dir / (os.path.splitext(fname)[0] + '.txt')
            
            if img_p.exists():
                img_p.unlink()
                print(f"Deleted image: {img_p.name}")
            if lbl_p.exists():
                lbl_p.unlink()
                print(f"Deleted label: {lbl_p.name}")
                
        report_data.append({
            'scene_id': scene_id,
            'original_count': original_count,
            'kept_count': len(kept_items),
            'removed_count': len(removed_items),
            'reason': f'Downsampled from {original_count} to 8'
        })
        
    # Write report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['scene_id', 'original_count', 'kept_count', 'removed_count', 'reason'])
        writer.writeheader()
        for row in report_data:
            writer.writerow(row)
            
    print(f"Scene downsampling complete. Report saved to {report_path}")

if __name__ == "__main__":
    main()
