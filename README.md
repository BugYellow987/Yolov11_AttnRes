# YOLO11 Backbone Attention Residuals 修改說明

本專案將 YOLO11 detection model 的 backbone 中原本的 `C3k2` 模組替換為 `AttentionResiduals` 模組。

## 修改內容

- 模型設定檔：`ultralytics/cfg/models/11/yolo11.yaml`
- 新增模組：`AttentionResiduals`
- 模組位置：`ultralytics/nn/modules/block.py`
- Parser 註冊位置：`ultralytics/nn/tasks.py`

目前只替換 backbone 中的 `C3k2`，head 仍維持原本 YOLO11 的結構。

## 訓練

```powershell
yolo detect train model=ultralytics/cfg/models/11/yolo11.yaml data=你的資料集.yaml imgsz=640 epochs=100 batch=16 device=0
```

使用 CPU 訓練：

```powershell
yolo detect train model=ultralytics/cfg/models/11/yolo11.yaml data=你的資料集.yaml imgsz=640 epochs=100 batch=16 device=cpu
```

續訓：

```powershell
yolo detect train resume=True
```

## 驗證

```powershell
yolo detect val model=runs/detect/train/weights/best.pt data=你的資料集.yaml imgsz=640 device=0
```

使用 CPU 驗證：

```powershell
yolo detect val model=runs/detect/train/weights/best.pt data=你的資料集.yaml imgsz=640 device=cpu
```

## 推論

對圖片或資料夾進行推論：

```powershell
yolo detect predict model=runs/detect/train/weights/best.pt source=圖片或資料夾路徑 imgsz=640 device=0
```

使用 CPU 推論：

```powershell
yolo detect predict model=runs/detect/train/weights/best.pt source=圖片或資料夾路徑 imgsz=640 device=cpu
```

## Python 推論

```python
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
results = model.predict(source="圖片或資料夾路徑", imgsz=640, device=0, save=True)
```

如果沒有 GPU，可以將 `device=0` 改成 `device="cpu"`。
