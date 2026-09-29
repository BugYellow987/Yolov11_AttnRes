# 0929：保留主幹的損壞架構修改規格

目標：在 yolo11-6csar-semantic-token 上增加真正參與推論的特徵與解碼旁路，優先處理低對比陰影凹損、細節流失及只覆蓋局部的問題。訓練參數維持共同設定，不把改 loss 或解析度當成架構改動。

## 全域限制

- 原 AttentionResiduals、FSNetShuffle、六個 CSAR、四個 DamageSemanticAttention 的定義、連接與索引保持原樣。
- 保留原入口 Conv 的計算與參數名稱；新增亮度細節殘差旁路，不替換為 IIMStem。
- 保留 P3/P2/P4/P5 次序、八類覆寫、stride 4 遮罩、獨立 D/R 多標籤與原偵測分支。
- 新增模組保留所有基底 state_dict 的 key 與形狀，新增權重由訓練學習。
- 四組訓練 recipe 的 optimizer、loss、640 輸入與資料前提相同，不啟用 training_aux；overlap_mask=False、mask_ratio=4。
- 不覆寫 0928 YAML、原圖、標記或權重；保留照片預設不顯示分數。
- 不自動啟動完整訓練；不能宣稱未經資料集驗證的準確率改善。

## 四組定義

00-baseline：完整複製 semantic-token 基底，作為共同訓練對照。

01-luma：layer 0 換為繼承 Conv 的 LuminanceResidualStem，主路為原 Conv；旁路重用 LuminanceDetail 的亮度、雙尺度相對對比、x/y 梯度，經 Conv/DWConv 與 sigmoid gate，加上 tanh(scale) 殘差。forward_fuse 必須保留旁路，不能被 BN 融合略過。

02-shallow：在 01 上重用 Segment26MultiLabelShadow，head 33 追加 layer 6、3 作為 P3/P2 淺層輸入。原六個 CSAR、語意模組與框／類別／係數路徑保留，只新增 DentDetailFusion 殘差。

03-dualproto：在 02 上採 Segment26MultiLabelExtent，其 proto 為 DualPathP2Proto。原 Proto26MultiLabel 完整保留；新增 P2 投影、上採樣的 P4 context、門控、DWConv 與輸出到 32 個原型的殘差。此旁路直接在 P2 尺度處理，不先將 P2 降到 P3。原 proto 的上採樣權重也保留。

## 驗收

- 入口旁路及解碼旁路關閉增益時與基底等價；開啟後有非零梯度。
- 00–03 皆可 YAML 建模、dataset nc 覆寫、真實 loss/backward、空標記、非正方形 eval、EMA、checkpoint round trip 與 fuse。
- 原基底與真實 last0926.pt 的所有權重均能同名同形載入；只允許新增模組缺少初始化權重，不掩蓋舊權重不相容。
- 各版模組數／參數量與新梯度路徑有量測記錄；保留原多標籤語意行為。
- 文件明確區分主幹保留、局部網路新增、舊 loss 版與新架構版、完整分割標記需求及推論成本。
