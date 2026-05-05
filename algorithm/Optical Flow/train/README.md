# train 工作區說明

此資料夾是道路面分割訓練專用工作區，與目前執行中的 `road_dataset` 分開管理。

## 目的

把後續的訓練資料、路徑規劃、實驗輸出與訓練決策集中管理，避免直接混改主流程資料夾。

## 目前已預先建立的路徑

```text
train/
  incoming/                    原始待整理影像或影片幀
  prepared/
    images/
      train/
      val/
      test/
    labels/
      train/
      val/
      test/
    road.yaml                  訓練資料集設定檔骨架
  configs/                     訓練參數設定檔
  weights/                     訓練輸出權重
  experiments/                 各輪實驗紀錄
  reports/                     評估報告、人工檢查結果
  TRAIN_DECISION_PLAN.md       本地訓練計畫與決策
```

## 使用原則

1. `incoming/` 只放尚未整理的原始來源。
2. `prepared/` 只放已完成切分與標註格式整理的資料。
3. `prepared/road.yaml` 只描述 `train/prepared` 這套訓練資料，不直接覆蓋既有 `road_dataset/road.yaml`。
4. `weights/` 與 `experiments/` 分開，避免權重與報告混雜。

## 下一步

1. 把要訓練的影像或影片幀放進 `incoming/`
2. 以 `CVAT + SAM` 做半自動標註
3. 完成標註後，整理到 `prepared/images/*` 與 `prepared/labels/*`
4. 依 `prepared/road.yaml` 啟動訓練

## 建議標註方式

目前最推薦：

```text
CVAT + SAM 半自動標註
```

流程概念：

```text
使用者點道路區域
-> SAM 產生初始 mask
-> 人工修正邊界
-> 匯出為 road_surface polygon / mask
```

對道路面這種大區域任務，比純手動畫 polygon 更有效率。
