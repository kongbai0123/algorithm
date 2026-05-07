# Metadata Governance Schema

為了推動真正的 Semantic Scene-Aware Split，每張參與訓練與驗證的圖片都應該在 `metadata.csv` 中記錄其語意屬性。

## 欄位定義

| 欄位名稱 (Column) | 填寫說明 (Description) | 範例值 (Examples) |
| :--- | :--- | :--- |
| **`filename`** | 檔案名稱 (包含副檔名)，作為主鍵 | `3_Video Project_000004.jpg`, `pexels_11794520.png` |
| **`split`** | 目前所屬的資料集切分狀態 | `train`, `val`, `test` |
| **`scene_id`** | 場景唯一識別碼，用於強制隔離切分 | `video3_scene1`, `pexels_forest_dirt` |
| **`source_id`** | 原始影片或素材庫來源 | `youtube_dashcam_01`, `pexels` |
| **`material`** | 道路主要材質 | `asphalt` (柏油), `cement` (水泥), `gravel` (碎石), `dirt` (泥土) |
| **`lighting`** | 光線與時間點 | `daylight` (白天), `night` (夜間), `dusk` (黃昏), `overcast` (陰天) |
| **`weather`** | 天氣狀況 | `clear` (晴天), `rain` (雨天), `snow` (雪地) |
| **`shadow_level`** | 畫面中陰影的干擾程度 | `none` (無), `light` (輕微), `heavy` (嚴重), `mottled` (斑駁樹影) |
| **`camera_angle`** | 攝影機視角 | `dashcam` (行車紀錄器平視), `high_angle` (高視角), `low_angle` (低視角) |
| **`road_distance`** | 道路可見的深度 | `near_only` (僅近端), `far_vanishing` (可見遠端收斂點) |
| **`hard_negative_tags`** | 畫面中含有的強烈干擾因素 (可複選，用 `\|` 分隔) | `grass_edge`, `sidewalk`, `zebra_crossing`, `parked_cars` |
| **`notes`** | 人工檢討或特殊標註備註 | `人工確認此圖 ground truth 邊界較模糊` |
