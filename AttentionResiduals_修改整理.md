# YOLO11 Backbone 修改為 AttentionResiduals 整理

## 修改目的

原本 YOLO11 backbone 使用 `C3k2`：

```yaml
- [-1, 2, C3k2, [256, False, 0.25]]
```

現在改成直接使用 `AttentionResiduals`：

```yaml
- [-1, 2, AttentionResiduals, [256, 0.25]]
```

也就是 backbone 的 `C3k2` 被放棄，不再使用原本的 CSP split、concat、`C3k` / `C3k2` 結構。

## 主要修改檔案

### 模型結構設定

檔案：

```text
ultralytics/cfg/models/11/yolo11.yaml
```

backbone 中 4 個 `C3k2` 已替換成 `AttentionResiduals`，head 的 `C3k2` 保留原樣。

### 新增模組

檔案：

```text
ultralytics/nn/modules/block.py
```

這裡新增了兩個核心類別：

```python
AttentionResiduals2d
AttentionResiduals
```

### 模組註冊

檔案：

```text
ultralytics/nn/modules/__init__.py
```

讓 Ultralytics 可以 import 到 `AttentionResiduals`。

### YAML parser 註冊

檔案：

```text
ultralytics/nn/tasks.py
```

把 `AttentionResiduals` 加進 `base_modules` 和 `repeat_modules`，這樣 YAML 裡寫：

```yaml
[-1, 2, AttentionResiduals, [256, 0.25]]
```

Ultralytics 才知道它是需要自動處理 `c1`、`c2`、`repeats` 的模組。

## 重要邏輯

### AttentionResiduals2d

`AttentionResiduals2d` 負責做 Attention Residuals 的核心運算。它會把多個 previous states 疊起來，對 depth 維度做 softmax 權重，再加權聚合：

```text
states = [x0, x1, x2, ...]
weights = softmax(query · normalized_states)
output = weighted_sum(states)
```

因為 YOLO backbone 是 CNN feature map，所以這裡處理的是：

```text
[B, C, H, W]
```

不是 Transformer 常見的：

```text
[B, T, D]
```

### AttentionResiduals

`AttentionResiduals` 是完整的 YOLO backbone block。流程大致如下：

```text
input
  -> 1x1 Conv 調整 hidden channel
  -> 多層 Conv feature transform
  -> 每一層前用 AttentionResiduals2d 聚合前面 states
  -> 最後再聚合所有 states
  -> 1x1 Conv 輸出成 YOLO 需要的 channel
```

簡化後像這樣：

```text
s0 = cv1(x)
s1 = conv1(AttnRes([s0]))
s2 = conv2(AttnRes([s0, s1]))
out = cv2(AttnRes([s0, s1, s2]))
```

## 目前架構重點

這次不是：

```text
C3k2 + AttentionResiduals
```

而是：

```text
AttentionResiduals 直接取代 backbone C3k2
```

所以 pretrained `yolo11n.pt` 的 backbone 權重大多不能直接沿用，訓練上比較接近重新訓練一個修改後的模型。
