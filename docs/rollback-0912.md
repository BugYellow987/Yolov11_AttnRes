# 回到 0912 基準版本

2026-09-18：依使用者回報，後續 shadow 與 boundary/P2（含 DCNv2 選項）的結果不理想，回到 0912，不再疊加新的架構改動。

## 目前使用的模型

- 權重：`C:/Users/USER/Downloads/last0912.pt`。
- 對應架構：`ultralytics/cfg/models/11_myself/yolo11-10csar-dr-msat-0911.yaml`。檔名是 0911，但這就是 0912 checkpoint 實際使用的 YAML。
- 架構：`IIMStem` → backbone → 5 CSAR → 5 FSNetShuffle → 5 CSAR → 5 MSATMultiLabel → `Segment26MultiLabel`。
- 0912 本身已有 P2 偵測與 D/R 多標籤監督，這些保留；不使用後續新增的亮度細節分支、淺層 bypass、邊界 loss、原生 P2 prototype 解碼或 DCNv2。

## 已調整的地方

`tools/predict-2.py` 的預設權重由 `last0915-2.pt` 改成 `last0912.pt`。原本的七張圖片、`imgsz=640`、`conf=0.25`、`iou=0.3`、device、輸出根目錄與 retina masks 設定不變。

原 0911 YAML 與原 multi-label loss 沒有被後續版本覆蓋，因此直接選回它們即可；沒有重置 Git、刪除後續模型／Python 模組或刪除任何 checkpoint、推論結果。後續模組保留供載入歷史權重，但不會加入 0912 模型的計算圖。

這次切換的是本機推論預設。專案沒有指定全域預設訓練模型，遠端訓練主機或其他推論指令仍須自行換成以下模型路徑。沒有啟動正式訓練或批次推論。

已檢查推論入口的預設參數；原架構仍為 47 層、11,927,937 個參數（nano、nc=8），與 `last0912.pt` 的 2102 個 state tensors 全數相容且可嚴格載入。確認使用原 `IIMStem`、`Segment26MultiLabel`、`Proto26MultiLabel` 與 multi-label criterion，計算圖不含後兩版分支。低亮度合成輸入推論為有限值，6 項 IIM／MSATMultiLabel 迴歸測試通過；這不是對真實資料效果的新評估。

## 推論

沿用原本按類別輸出的工具：

```powershell
python tools/predict-2.py
```

也可以直接用 YOLO 指令載入原權重，不需要重新訓練：

```powershell
yolo segment predict model="C:/Users/USER/Downloads/last0912.pt" source="你的圖片或資料夾" imgsz=640 conf=0.25 iou=0.3 retina_masks=True
```

後者是標準 YOLO 推論輸出，不等同於 `predict-2.py` 的逐類別存圖格式。要重現 0912 模型，必須選回原本的 `last0912.pt`；只改 YAML 而仍載入 0915 權重，不是回到原 0912 的預測。

## 若要繼續訓練 0912 架構

在訓練主機專案根目錄執行，將資料與 checkpoint 換成實際路徑：

```bash
yolo segment train \
  model=ultralytics/cfg/models/11_myself/yolo11-10csar-dr-msat-0911.yaml \
  pretrained=/path/to/last0912.pt \
  data=/path/to/data.yaml \
  epochs=250 batch=8 imgsz=640 \
  optimizer=auto amp=False overlap_mask=False mask_ratio=4 \
  resume=False name=yolo11-dr-msat-0912-return
```

`overlap_mask=False` 仍然必要，因為原模型也使用 multi-label 共存監督。`last0912.pt` 已剝除 optimizer 等續訓狀態，所以這是從原權重初始化的新一輪訓練，不能設 `resume=True`。這個指令不是恢復舊實驗剩餘 epoch，也不保證重新訓練後與原權重結果相同。

後續再比較模型時，先固定此基準與推論條件；本次沒有根據單張疊圖推斷後兩版退步的具體原因。
