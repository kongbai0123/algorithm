import os
import glob
import zipfile
import argparse

def main():
    parser = argparse.ArgumentParser(description="將現有的 YOLO .txt 標籤打包為 CVAT YOLO 1.1 匯入格式")
    parser.add_argument("--txt-dir", required=True, help="包含 .txt 標籤檔案的資料夾路徑")
    parser.add_argument("--out-zip", default="cvat_import_yolo_1.1.zip", help="輸出的 ZIP 檔案名稱")
    args = parser.parse_args()

    txt_files = glob.glob(os.path.join(args.txt_dir, "*.txt"))
    if not txt_files:
        print(f"錯誤：在 {args.txt_dir} 找不到任何 .txt 檔案")
        return

    # 找出所有出現過的 class_id
    class_ids = set()
    for txt_file in txt_files:
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    try:
                        class_ids.add(int(parts[0]))
                    except ValueError:
                        pass
    
    if not class_ids:
        print("警告：在 .txt 檔案中沒有找到任何有效的標籤資料。")
        return

    max_class_id = max(class_ids)
    # 建立臨時的 class 名稱清單 (例如 class_0, class_1)
    names = [f"class_{i}" for i in range(max_class_id + 1)]

    # 打包 ZIP
    with zipfile.ZipFile(args.out_zip, 'w') as zf:
        # 建立並寫入 obj.names
        zf.writestr("obj.names", "\n".join(names))
        
        # 建立並寫入 obj.data
        obj_data = f"classes = {len(names)}\nnames = obj.names\ntrain = train.txt\n"
        zf.writestr("obj.data", obj_data)
        
        # 將所有的 .txt 放入 obj_train_data/ 目錄下
        for txt_file in txt_files:
            basename = os.path.basename(txt_file)
            zf.write(txt_file, os.path.join("obj_train_data", basename))

    print(f"成功將 {len(txt_files)} 個標籤檔案打包至 {args.out_zip}")
    print(f"發現的類別 ID: {sorted(list(class_ids))}")
    print("您可以直接將此 ZIP 上傳至 CVAT 的 Upload Annotations (YOLO 1.1) 功能。")

if __name__ == "__main__":
    main()
