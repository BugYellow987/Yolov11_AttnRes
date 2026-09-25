# YOLO11-6CSAR 損傷語意 token 架構

模型設定：`ultralytics/cfg/models/11_myself/yolo11-6csar-semantic-token.yaml`。

本版以目前的 `yolo11-6csar.yaml` 為基礎，實作 [HackMD 第二部分](https://hackmd.io/8gre3-RoQsG02dfl_OER_Q) 的損傷 token 查詢與特徵回饋。原本第 0–20 層、六個 CSAR，以及 head 的 P3/P2/P4/P5 特徵順序均保留；P2 沿用原 YAML 的第 2 層淺層特徵。

```text
Image → AttentionResiduals → FSNetShuffle → CSAR → Visual Feature
                                                    │
                  Damage Tokens (Q)                 ├── 視覺殘差路徑 ─────────┐
                          │                         │                       │
                          └── Cross-Attention ←─────┘ (visual tokens K/V)   │
                                      │                                     │
                           更新後的 damage tokens                            │
                                      │                                     │
                     各類別獨立的逐像素 response                              │
                             │                 │                            │
                      多標籤 BCE + Dice      語意特徵 ────────────────────────┤
                                                                           ↓
                                                            Semantic-aware Feature
                                                                           ↓
                                                               YOLO detection / masks
```

`DamageSemanticAttention` 在每個特徵尺度建立 `nc` 個可學習 token。token 的第 i 列對應資料集的第 i 個類別，例如 Rust、Dent、Hole；類別數在訓練時由資料集 YAML 覆寫。此版本的語意來自類別標註監督，沒有使用文字編碼器或預訓練文字 embedding，也不支援未訓練類別的文字查詢。

Cross-attention 使用 damage token 作為 Q，視覺特徵作為 K/V。softmax 僅作用於空間位置，每個類別各自讀取影像。更新後的 token 與各像素特徵計算 logits，再用獨立 sigmoid 產生 response，因此同一像素可同時對多種損傷呈現高分。response 加權的 token 特徵經投影、可學習殘差係數後加回原視覺特徵，送入 YOLO head；標準分割 loss 也能更新此分支。

新 YAML 的第 21、24、27、30 層是四個語意分支，各回傳 `(enhanced_feature, multilabel_logits)`，由 `Index` 分別取出，送入現有 `Segment26MultiLabel`。多標籤目標由各 instance mask 依類別合併，同一位置保留多個正類別。沿用的 BCE + Dice loss 對重疊像素增加權重，不需要新增文字標註。

## 訓練

在專案根目錄、已安裝本專案依賴的 Python 環境執行：

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/11_myself/yolo11-6csar-semantic-token.yaml")
model.train(
    data="path/to/data.yaml",  # 替換為實際資料集 YAML
    epochs=100,
    imgsz=640,
    batch=8,
    device=0,
    overlap_mask=False,
    mask_ratio=4,
)
```

`overlap_mask=False` 是必要設定，讓 Rust/Dent 等重疊標註仍是獨立 instance mask；若設為 True，現有多標籤 loss 會明確拒絕訓練。`mask_ratio=4` 對應此 head 的 prototype 解析度。資料集應保留重疊物件的各自多邊形與類別；單一互斥的語意標籤圖無法提供多標籤目標。

模型預設使用 n 尺度，保留與基礎 YAML 相同的 n/s/m/l/x scaling 設定。語意分支參數 `[nc, 4, 128, 0.001]` 依序為類別數、attention heads、embedding 維度、初始殘差係數。最後 head 的 `[nc, 32, 256, 0.5, 2.0]` 中，`0.5` 為多標籤 loss gain，`2.0` 為重疊像素額外權重。

可用 `model.load("path/to/base_checkpoint.pt")` 載入形狀相容的基礎權重。新增層使最終 head 索引從 21 變成 33，因此一般權重載入主要沿用 backbone 與 CSAR；不要假設原 head 權重會自動搬移。

## 輸出與驗證範圍

一般 `model.predict()` 保留 YOLO 的框、類別與 instance mask 介面；語意強化分支在推論時仍參與特徵計算。直接呼叫底層 PyTorch 模型時，訓練字典的 `multilabel_logits`（eval 時在回傳值的第二項字典中）包含 P3/P2/P4/P5 四組 `[B, nc, H, W]` logits；對它們取 sigmoid 可分析各損傷的獨立空間 response。標準 `Results` 不會額外包裝這些 response maps。

本版提供可訓練架構，不代表已證明重疊損傷辨識率改善。需以相同資料切分、訓練設定比較基礎模型，並檢查重疊區域的各類別召回率。
