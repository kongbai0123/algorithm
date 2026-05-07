import csv
import json
import math
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict

def evaluate_dataset_health(metadata_csv: Path, output_dir: Path) -> dict:
    """
    Evaluates dataset health based on metadata.csv
    Returns a dictionary of scores.
    """
    if not metadata_csv.exists():
        return _empty_health_report("metadata.csv not found.", output_dir)
        
    records = []
    with open(metadata_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
            
    if not records:
        return _empty_health_report("metadata.csv is empty.", output_dir)
        
    total_images = len(records)
    
    # 1. Metadata Completeness (10 points)
    required_fields = ['filename', 'split', 'scene_id', 'source_id', 'material', 'lighting', 'shadow_level', 'hard_negative_tags']
    total_expected = total_images * len(required_fields)
    filled_fields = 0
    for r in records:
        for f in required_fields:
            if r.get(f) and r[f].strip():
                filled_fields += 1
    completeness_ratio = filled_fields / total_expected if total_expected > 0 else 0
    score_completeness = completeness_ratio * 10.0

    # 2. Material Balance (20 points)
    material_counts = defaultdict(int)
    for r in records:
        mat = r.get('material', '').strip()
        if mat:
            material_counts[mat] += 1
    num_materials = len(material_counts)
    if num_materials <= 1:
        score_material = 0.0
    else:
        entropy = 0.0
        for count in material_counts.values():
            p = count / sum(material_counts.values())
            entropy -= p * math.log(p)
        max_entropy = math.log(num_materials)
        balance_ratio = entropy / max_entropy if max_entropy > 0 else 0
        score_material = balance_ratio * 20.0

    # 3. Scene Diversity Score (20 points)
    unique_scenes = len(set(r.get('scene_id', '').strip() for r in records if r.get('scene_id', '').strip()))
    scene_ratio = unique_scenes / total_images if total_images > 0 else 0
    if scene_ratio >= 0.5:
        score_scene = 20.0
    elif scene_ratio >= 0.25:
        score_scene = 10.0 + 10.0 * ((scene_ratio - 0.25) / 0.25)
    else:
        score_scene = (scene_ratio / 0.25) * 10.0

    # 4. Lighting / Shadow Diversity (15 points)
    hard_lighting_count = sum(1 for r in records if r.get('lighting', '').strip().lower() in ['night', 'dusk', 'overcast'] or r.get('shadow_level', '').strip().lower() in ['heavy', 'mottled'])
    lighting_ratio = hard_lighting_count / total_images if total_images > 0 else 0
    if lighting_ratio >= 0.15:
        score_lighting = 15.0
    else:
        score_lighting = (lighting_ratio / 0.15) * 15.0

    # 5. Hard-Negative Density (20 points)
    hard_negative_count = sum(1 for r in records if r.get('hard_negative_tags', '').strip())
    hn_ratio = hard_negative_count / total_images if total_images > 0 else 0
    if hn_ratio >= 0.20:
        score_hn = 20.0
    elif hn_ratio >= 0.10:
        score_hn = 10.0 + 10.0 * ((hn_ratio - 0.10) / 0.10)
    else:
        score_hn = (hn_ratio / 0.10) * 10.0

    # 6. Duplicate Ratio (15 points)
    scene_counts = defaultdict(int)
    for r in records:
        sid = r.get('scene_id', '').strip()
        if sid:
            scene_counts[sid] += 1
    max_images_per_scene = max(scene_counts.values()) if scene_counts else 0
    duplicate_risk = max_images_per_scene / total_images if total_images > 0 else 0
    
    if duplicate_risk <= 0.15:
        score_duplicate = 15.0
    elif duplicate_risk <= 0.30:
        score_duplicate = 15.0 - 7.5 * ((duplicate_risk - 0.15) / 0.15)
    else:
        score_duplicate = max(0.0, 7.5 - 7.5 * ((duplicate_risk - 0.30) / 0.70))

    total_score = score_completeness + score_material + score_scene + score_lighting + score_hn + score_duplicate
    
    if total_score >= 80:
        decision = "PASS"
        decision_desc = "Dataset is healthy and diverse."
    elif total_score >= 60:
        decision = "WARN"
        decision_desc = "Dataset has weaknesses. E.g., hard-negative samples or diversity may be insufficient."
    else:
        decision = "FAIL"
        decision_desc = "Dataset lacks diversity or metadata. Health check failed."

    report_dict = {
        "Total Score": round(total_score, 1),
        "Decision": decision,
        "Decision Description": decision_desc,
        "Metrics": {
            "Material Balance (20)": round(score_material, 1),
            "Scene Diversity (20)": round(score_scene, 1),
            "Lighting/Shadow (15)": round(score_lighting, 1),
            "Hard-Negative Density (20)": round(score_hn, 1),
            "Duplicate Ratio (15)": round(score_duplicate, 1),
            "Metadata Completeness (10)": round(score_completeness, 1)
        },
        "Raw Ratios": {
            "Completeness Ratio": round(completeness_ratio, 3),
            "Scene Ratio": round(scene_ratio, 3),
            "Lighting Ratio": round(lighting_ratio, 3),
            "Hard Negative Ratio": round(hn_ratio, 3),
            "Max Scene Duplicate Risk": round(duplicate_risk, 3)
        }
    }
    
    # Generate Markdown Output
    md_content = f"""# Dataset Health Report
**Total Score:** {round(total_score, 1)}/100 [{decision}]

| Metric | Score | Max |
| :--- | ---: | ---: |
| Material Balance | {round(score_material, 1)} | 20 |
| Scene Diversity | {round(score_scene, 1)} | 20 |
| Lighting/Shadow | {round(score_lighting, 1)} | 15 |
| Hard-Negative Density | {round(score_hn, 1)} | 20 |
| Duplicate Ratio | {round(score_duplicate, 1)} | 15 |
| Metadata Completeness | {round(score_completeness, 1)} | 10 |

**Decision:**
{decision} - {decision_desc}
"""

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "dataset_health_report.json", "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)
        
    with open(output_dir / "dataset_health_report.md", "w", encoding="utf-8") as f:
        f.write(md_content)
        
    print(md_content)
    return report_dict

def _empty_health_report(reason: str, output_dir: Path) -> dict:
    report_dict = {
        "Total Score": 0,
        "Decision": "WARN",
        "Decision Description": f"Bootstrap mode - {reason}",
    }
    md_content = f"""# Dataset Health Report
**Total Score:** 0/100 [WARN]

**Decision:**
WARN - Bootstrap mode - {reason}
"""
    print(md_content)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "dataset_health_report.md", "w", encoding="utf-8") as f:
        f.write(md_content)
    return report_dict

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    evaluate_dataset_health(base_dir / "metadata.csv", base_dir / "reports")
