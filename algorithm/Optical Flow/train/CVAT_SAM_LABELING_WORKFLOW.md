# CVAT + SAM 半自動標註流程

## 目的

用較低人工成本建立高品質 `road_surface` 標註資料，作為後續 YOLO segmentation 訓練基礎。

## 建議工具

```text
CVAT + SAM 互動式分割
```

## 適用原因

道路面是大面積連續區域。  
對這類任務，純手動畫 polygon 太慢；SAM 先切候選區，再由人工修邊，效率通常更高。

## 操作流程

### 1. 建立資料來源

把原始影像或影片擷取影格放進：

```text
train/incoming/
```

### 2. 匯入 CVAT

在 CVAT 中建立新 task，匯入 `train/incoming/` 影像。

建議類別先只設：

```text
road_surface
```

### 3. 使用 SAM 建立初始道路區域

操作方式：

```text
在道路區域點正樣本
-> 必要時點幾個負樣本
-> 讓 SAM 生成初始 mask
```

### 4. 人工修正

不要直接相信 SAM 結果，必須人工檢查與修正以下邊界：

- 陰影下仍屬道路的區域
- 斑馬線與車道線所在道路
- 草地邊界
- 人行道邊界
- 路肩是否應納入
- 車輛遮擋附近
- 遠端低對比道路

### 5. 匯出資料

從 CVAT 匯出可轉成 YOLO segmentation 的 polygon / mask 標註。

整理後放入：

```text
train/prepared/images/train/
train/prepared/images/val/
train/prepared/images/test/
train/prepared/labels/train/
train/prepared/labels/val/
train/prepared/labels/test/
```

### 6. 檢查資料一致性

標註完成後至少做三種檢查：

1. 類別是否只有 `road_surface`
2. 標註是否只包含可見道路面
3. train / val / test 是否依場景切分，而不是隨機拆連續影格

## 標註原則

### 要包含

- 近端道路
- 遠端可見道路
- 陰影覆蓋但仍可辨識的道路
- 車道線所在道路
- 斑馬線所在道路
- 可見濕路面

### 不要包含

- 草地
- 人行道
- 路肩（除非任務另定）
- 車體
- 行人
- 建築物
- 天空
- 被遮擋且不可見、只能推測存在的道路

## 產出目標

第一輪至少完成：

```text
300-500 張可訓練 road_surface 標註
```

並優先涵蓋：

```text
草地旁道路
人行道旁道路
停車遮擋道路
陰影道路
遠端收斂道路
強反光道路
低對比道路
```

## 完成後銜接

資料整理完成後，直接使用：

```powershell
python train_road_segmentation.py --model yolov8n-seg.pt --data train/prepared/road.yaml --imgsz 640 --epochs 100 --batch 8 --seed 42 --patience 20 --name road_surface_custom_v1
```
