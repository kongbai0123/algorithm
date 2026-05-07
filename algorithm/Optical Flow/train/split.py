import os
import random
import shutil
from pathlib import Path

def main():
    base_dir = Path(r"c:\workspace\algorithm\Optical Flow\train\prepared")
    input_dir = Path(r"c:\workspace\algorithm\Optical Flow\input")
    # base_dir = Path(r"C:\antigravity_data\origin_img\temp1")
    # input_dir = Path(r"C:\antigravity_data\origin_img\temp1")


    images_dir = base_dir / "images"
    labels_dir = base_dir / "labels"

    # 清理舊的分類資料夾，確保每次切分都是乾淨的
    for split in ['train', 'val', 'test']:
        for d in [images_dir / split, labels_dir / split]:
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

    # 收集圖片：從 input/ 遞迴找，以及從 prepared/images/ 的「第一層」找 (支援多種格式)
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp', '*.JPG', '*.JPEG', '*.PNG']
    all_images = []
    for ext in image_extensions:
        all_images.extend(input_dir.rglob(ext))
        all_images.extend(images_dir.glob(ext))
    
    # 收集標籤：從 input/ 遞迴找，以及從 prepared/labels/ 的「第一層」找
    all_labels = list(input_dir.rglob("*.txt")) + list(labels_dir.glob("*.txt"))

    # 建立字典以檔名 (不含副檔名) 為 key 加速配對
    img_dict = {f.stem: f for f in all_images}
    label_dict = {f.stem: f for f in all_labels if f.stem != "classes"}

    # 配對影像與標籤 (以圖片為主，支援負樣本)
    valid_items = []
    for name, img_path in img_dict.items():
        txt_path = label_dict.get(name)
        valid_items.append((img_path, txt_path))
        if not txt_path:
            print(f"提示：圖片 {name}.jpg 沒有對應的標籤，將作為負樣本 (Background) 處理。")

    if not valid_items:
        print("錯誤：找不到任何圖片！請確認圖片在 input 目錄。")
        return

    print(f"共找到 {len(valid_items)} 張圖片參與訓練 (包含負樣本)。")

    # 隨機打亂並切分 (70% train, 20% val, 10% test)
    random.seed(42)  # 固定亂數種子
    random.shuffle(valid_items)

    total = len(valid_items)
    train_end = int(total * 0.7)
    val_end = int(total * 0.9)

    train_items = valid_items[:train_end]
    val_items = valid_items[train_end:val_end]
    test_items = valid_items[val_end:]

    def copy_pairs(items, split_name):
        for img_path, txt_path in items:
            # 複製圖片
            shutil.copy(img_path, images_dir / split_name / img_path.name)
            
            # 處理標籤：若有標籤則複製，若無標籤(負樣本)則建立空白檔
            target_txt = labels_dir / split_name / (img_path.stem + ".txt")
            if txt_path:
                shutil.copy(txt_path, target_txt)
            else:
                target_txt.touch() # 建立空白標籤檔
                
        print(f"成功分配 {len(items)} 張圖片到 {split_name}。")

    copy_pairs(train_items, 'train')
    copy_pairs(val_items, 'val')
    copy_pairs(test_items, 'test')
    
    print("\n[成功] 資料集切分完畢！原始資料已安全保留，分類副本已進入 train/val/test。")

if __name__ == "__main__":
    main()
