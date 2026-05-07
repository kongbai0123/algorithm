# train 訓練決策與計畫

建立時間：2026-05-05 16:46:32

## 目標

建立一個獨立於主流程的訓練工作區，專門用於 `road_surface` segmentation 的資料準備、訓練、實驗比較與報告整理。

## 決策

目前不直接改動既有 `road_dataset`，而是另外建立：

```text
algorithm/Optical Flow/train/
```

原因如下：

1. 避免把訓練中間資料與目前主流程用的資料夾混在一起。
2. 方便你後續分版本管理不同訓練輪次。
3. 方便把 `incoming -> prepared -> weights -> experiments` 做成清楚的資料生命週期。

## 已預處理路徑

以下路徑已先建立，但目前尚未搬入實際訓練資料：

```text
train/incoming/
train/prepared/images/train/
train/prepared/images/val/
train/prepared/images/test/
train/prepared/labels/train/
train/prepared/labels/val/
train/prepared/labels/test/
train/configs/
train/weights/
train/experiments/
train/reports/
```

另已建立資料集設定檔骨架：

```text
train/prepared/road.yaml
```

## 路徑用途定義

- `incoming/`
  - 放原始影像、原始影片幀、未整理的資料來源
- `prepared/images/*`
  - 放完成切分後的訓練影像
- `prepared/labels/*`
  - 放與影像對應的 YOLO segmentation polygon 標註
- `configs/`
  - 放訓練參數設定，例如 v1、v2、sweep 配置
- `weights/`
  - 放 `best.pt`、`last.pt` 等訓練結果
- `experiments/`
  - 放每一輪訓練的 config、metrics、summary、notes
- `reports/`
  - 放人工檢查、錯誤分析、benchmark 結果

## 訓練計畫

### Phase 1：資料定義

先把 `road_surface` 範圍固定：

```text
包含：近端道路、遠端可見道路、陰影道路、車道線所在道路、斑馬線所在道路、可辨識的濕路面
排除：草地、人行道、路肩（除非任務另定）、車體、行人、建築、天空、不可見但被推測的道路
```

### Phase 2：資料收集

優先補以下場景：

```text
草地旁道路
人行道旁道路
停車干擾道路
遠端收斂道路
陰影覆蓋道路
強反光道路
低對比道路
左右不對稱道路
```

### Phase 2.5：標註流程

目前最推薦的標註方法是：

```text
CVAT + SAM 半自動標註
```

建議流程：

```text
train/incoming/
-> 匯入 CVAT task
-> 使用 SAM 互動式分割建立初始道路面 mask
-> 人工修正邊界
-> 匯出 road_surface polygon / mask
-> 整理到 train/prepared/images/* 與 train/prepared/labels/*
```

採用原因：

1. 道路面通常是大面積區域，SAM 很適合先切出候選區。
2. 標註工作可從「完整描邊」改成「檢查與修正」。
3. 對目前目標最有幫助的是先快速取得高品質 `road_surface` 標註資料。

已知限制：

1. SAM 不理解道路語意，只是依視覺邊界切區域。
2. 在以下情況仍需要人工修正：

```text
陰影
斑馬線
草地邊界
人行道 / 路肩
停車遮擋
低對比遠端道路
```

### Phase 2.8：資料切分政策 (Split Policy)

**決策 (Decision):**
正式廢棄隨機切分 (Random Split)，因其會造成影片相鄰畫格的資料洩漏 (Data Leakage) 風險。

**新政策 (New policy):**
採用「場景隔離切分 (Scene-Aware Split)」，依據檔名前綴、來源影片或場景群組來進行切分。

**原因 (Reason):**
驗證集的分數必須真實反映模型對「未見過場景」的泛化能力，而不是對相鄰畫格的死背能力。

### Phase 2.9：v24 困難負樣本修復 (Hard-Negative Repair)

**背景 (Background):**
我們已使用 Scene-Aware Split 取代隨機切分，成功防堵了資料洩漏問題。`v23_scene_split` 的結果為我們提供了一個更具可信度的驗證基準點。

**發現 (Finding):**
模型整體的分割能力尚可，但水泥 (Cement) 與碎石 (Gravel) 的召回率 (Recall) 偏低，顯示在未見過的場景中容易漏抓。此外，森林 (Forest) 與古道 (Via Appia Antica) 的分數異常偏高，需要加入困難負樣本來驗證其真實性。

**決策 (Decision):**
下一階段不優先追求更大的模型容量 (如升級 YOLOv8s)。相反地，我們將專注於「精準補料」，透過補充特定的困難負樣本與少數類別來修復資料集。

**行動項目 (Action Items):**
1. 補充多樣化的**水泥 (Cement)** 樣本，涵蓋不同光線、距離與路面狀況。
2. 補充多樣化的**碎石 (Gravel)** 樣本，涵蓋不同紋理大小與陰影變化。
3. 加入類似森林環境但**非泥土路**的困難負樣本。
4. 加入類似古道石板路但**非目標道路**的困難負樣本。
5. 使用修復後的資料集重新訓練，命名為 `road_surface_custom_v24_hard_negative_repair`。
6. 對比 v23 與 v24，重點觀察各類別的 Mask Recall、Mask mAP50、Mask mAP50-95，並進行失敗案例 (Failure-frame) 檢討。

### Phase 3：模型訓練
建議第一輪以：

```powershell
python train_road_segmentation.py --model yolov8n-seg.pt --data train/prepared/road.yaml --imgsz 640 --epochs 100 --batch 8 --seed 42 --patience 20 --name road_surface_custom_v1
```

### Phase 4：參數客製化

推論時先優先調：

```text
conf
bottom_roi_ratio
min_area_ratio
```

之後再視需要擴充 morphology kernel 參數。

### Phase 5：評估

除既有指標外，後續應補：

```text
road_recall
road_precision
road_iou
top_region_coverage
middle_region_coverage
bottom_region_coverage
```

## 目前不做的事

現階段不直接在這個 `train` 工作區內導入：

```text
PWC-Net 主流程化
硬體部署
導航判斷
複雜控制邏輯
```

先把 `road_surface mask` 本身訓練準，再處理 temporal fusion。
