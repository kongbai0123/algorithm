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

### Phase 3.1：失敗分類學 (Failure Taxonomy System)

建立統一錯誤分類機制：
- `false_negative` (道路漏抓)
- `under_segmentation` (道路有抓到，但覆蓋不足)
- `boundary_collapse` (邊界破碎)
- `over_segmentation` (吃到草地、人行道)
- `shadow_confusion` (陰影誤判)
- `perspective_failure` (遠端收斂崩潰)
- `temporal_flicker` (前後幀跳動)
- `label_noise` (標註本身錯誤或邊界不一致)

每次 Validation 後，必須將 failure frame 分類並放入專屬目錄進行統計。

### Phase 3.2：空間基準 (Spatial Benchmark)

將畫面分為：
- `top_region`
- `middle_region`
- `bottom_region`

分別統計 Recall, IoU, Coverage。
原因：遠端道路通常最容易崩潰，不能只看 Overall mAP。

### Phase 3.3：元資料治理 (Metadata Governance)

每筆資料應記錄於 `metadata.csv`：
- `material`
- `weather`
- `lighting`
- `shadow_level`
- `camera_angle`
- `scene_id`
- 等擴充欄位...

未來 Split Policy 將逐步由 filename-based 過渡至 metadata-based。

### Phase 3.4：失敗驅動資料修復 (Failure-driven Dataset Repair)

v25 開始，模型優化不再以單一 mAP 作為唯一依據，而是以 Failure Taxonomy 統計結果作為補料依據。每次 validation 後，需將失敗樣本依 `false_negative`、`under_segmentation`、`over_segmentation`、`shadow_confusion`、`perspective_failure`、`temporal_flicker`、`label_noise` 等類別歸檔，並根據高頻錯誤類型制定下一輪資料補強策略。

### Phase 3.5：覆蓋率優先評估 (Coverage-first Evaluation)

由於道路面分割任務重視完整覆蓋，v25 起需新增 `road_recall`、`top_region_coverage`、`middle_region_coverage`、`bottom_region_coverage` 與 `under_segmentation_rate`。若 mAP 提升但道路覆蓋率下降，則不視為有效優化。

### Phase 3.6：元資料填充與語意切分準備 (Metadata Population)

`metadata.csv` 不再僅作為欄位骨架，需逐步填入每張圖片的 `scene_id`、`source_id`、`material`、`lighting`、`shadow_level`、`camera_angle`、`road_distance` 與 `hard_negative_tags`。後續 split.py 將從 filename-based scene split 升級為 metadata-based semantic split。

### Phase 3.7：資料集健康評分 (Dataset Health Scoring)

每輪訓練前，計算：
- class balance score (類別均衡)
- scene diversity score (場景多樣性)
- lighting diversity (光線分布)
- shadow distribution (陰影比例)
- duplicate ratio (重複場景率)
- hard-negative density (干擾樣本比例)

若 dataset health score 低於門檻，禁止進行正式訓練。

### Phase 3.8a：失敗量化驗證穩定化 (Failure Verifier Stabilization)

所有 failure taxonomy 必須具備量化定義，避免依賴人工感覺：
- `gt=0 且 pred=0` → `true_negative / pass`
- `gt=0 且 pred>0` → `false_positive`
- `gt>0 且 pred=0` → `false_negative`
- `gt>0 且 pred<gt` → `under_segmentation`
- `gt>0 且 pred>gt` → `over_segmentation`

此工具需支援 `--conf`、`--imgsz` 與影像副檔名篩選，以確保推論穩定。

### Phase 3.8b：v25 目標補料 (Targeted Repair)

v25 補料優先順序（根據 v24 failure_verifier 診斷結果）：
1. Belgian block / cobblestone road
2. 低對比石板路
3. 濕路面 / 反光路面
4. 無道路但相似紋理的背景圖
5. 邊界模糊的人行道 / 草地交界

### Phase 3.8c：重複幀降採樣 (Duplicate Reduction)

因 Duplicate Ratio 只有 5.8 / 15，需加入降採樣策略：
- `scene_id` 內最多保留 N 張
- 相似連續幀下採樣
- 同場景只保留代表性樣本

### Phase 3.9：自動元資料萃取 (Auto Metadata Extraction)

等 metadata 啟動後，再將部分特徵改由自動分析產生，減輕人工維護成本：
- `brightness` (平均亮度)
- `shadow_ratio` (dark pixel ratio)
- `texture_complexity` (Laplacian variance)
- `motion_level` (optical flow magnitude)
- `edge_density` (Canny edge ratio)

### Phase 4.1：模型訓練

```powershell
python train_road_segmentation.py --model yolov8n-seg.pt --data train/prepared/road.yaml --imgsz 640 --epochs 100 --batch 8 --seed 42 --patience 20 --name road_surface_custom_v1
```

### Phase 4.2：推論參數客製化

推論時先優先調：

```text
conf
bottom_roi_ratio
min_area_ratio
```

之後再視需要擴充 morphology kernel 參數。

### Phase 5：進階評估指標

除既有指標外，後續應補：

```text
road_recall
road_precision
road_iou
top_region_coverage
middle_region_coverage
bottom_region_coverage
```

## 最終長期決策藍圖與優先順序 (Roadmap & Priorities)

目前核心精神：**不再依賴 mAP 總分與人工感覺，建立以數學與自動化為基礎的 Perception Governance System。**

**優先順序：**
1. **第一優先：Dataset Health Score** (Phase 3.7)。沒有量化的資料品質指標，Data-Centric 會淪為感覺。
2. **第二優先：Failure Verification Protocol** (Phase 3.8)。讓錯誤分類具備數學定義 (IoU, Coverage)。
3. **第三優先：Auto Metadata Extraction** (Phase 3.9)。自動計算亮度、複雜度等特徵，避免人工填表爆炸。
4. **第四優先：Temporal Benchmark**。引入 temporal IoU 等動態指標，為未來的影片推論打地基。
5. **第五優先：模型升級與硬體部署**。待上述閉環成熟後，才推進 YOLOv8s、PWC-Net 或 Jetson 部署。

## 目前不做的事

現階段不直接在這個 `train` 工作區內導入：

```text
PWC-Net 主流程化
硬體部署
導航判斷
複雜控制邏輯
```

先把資料引擎的數學與評量基礎打穩，不要急著換模型！
