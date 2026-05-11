import os
import glob
from pathlib import Path

def polygon_area(coords):
    """Calculate area of a polygon using Shoelace formula."""
    n = len(coords)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += coords[i][0] * coords[j][1]
        area -= coords[j][0] * coords[i][1]
    return abs(area) / 2.0

def main():
    labels_dir = Path('input/road/labels')
    output_file = Path('train/relabel_audit/priority_audit_list.csv')
    
    print(f"Scanning labels in {labels_dir}...")
    
    audit_items = []
    
    # Walk through all txt files
    for root, dirs, files in os.walk(labels_dir):
        for f in files:
            if f.endswith('.txt'):
                label_path = Path(root) / f
                
                # Get material from parent folder name
                material = label_path.parent.name
                if material == 'labels':
                    material = 'root'
                    
                total_area = 0.0
                
                try:
                    with open(label_path, 'r') as file:
                        for line in file.readlines():
                            parts = line.strip().split()
                            if len(parts) < 5:
                                continue
                            
                            # YOLO format: class x1 y1 x2 y2 ...
                            # Extract coordinates
                            coords = []
                            for i in range(1, len(parts), 2):
                                x = float(parts[i])
                                y = float(parts[i+1])
                                coords.append((x, y))
                                
                            if len(coords) >= 3:
                                total_area += polygon_area(coords)
                                
                    # If total area > 60% of the image, mark as priority
                    if total_area > 0.6:
                        audit_items.append({
                            'filename': f,
                            'material': material,
                            'area_pct': total_area * 100,
                            'path': label_path
                        })
                except Exception as e:
                    print(f"Error processing {label_path}: {e}")

    # Sort by area percentage descending
    audit_items.sort(key=lambda x: x['area_pct'], reverse=True)
    
    # Write to CSV
    with open(output_file, 'w') as out:
        out.write("Filename,Material,Area_Percentage,Label_Path\n")
        for item in audit_items:
            out.write(f"{item['filename']},{item['material']},{item['area_pct']:.2f}%,{item['path']}\n")
            
    print(f"\nAudit complete!")
    print(f"Found {len(audit_items)} files with mask area > 60%.")
    print(f"List saved to {output_file}")

if __name__ == "__main__":
    main()
