# 0928 完整損壞範圍訓練規格

以使用者重新標記的 `C:/Users/USER/Desktop/0928-1/CBHU0708054-A.xml` 為本例依據（D 25、R 69）。目前模型存在完全漏檢及只覆蓋損壞局部兩種問題；本輪提供可執行、可比較的訓練修改與依序編號的 YAML，不宣稱未經訓練即可改善。

## 全域限制

- 保留 yolo11-6csar-semantic-token.yaml 的 backbone、head、P3/P2/P4/P5 順序與推論參數形狀。
- 保留獨立 D/R 實例及原多標籤監督；overlap_mask=False、mask_ratio=4、compile=False、resume=False。
- 不覆寫原始影像、XML、現有分割標記、權重與 tools/predict-2.py。
- XML 框不自動轉成精細遮罩；範圍 loss 需要已核對的實例 mask。
- 新 loss 僅作用於訓練；舊 YAML 預設關閉；驗證保持原 loss。
- 不新增外部依賴，不自動啟動完整訓練，不修改既有推論門檻。

## 實作

使用者追加：推論照片預設不顯示數值。修改 `tools/predict_seg_by_class.py` 的顯示選項，讓 `predict-2.py` 呼叫時預設不畫信心分數；保留框／遮罩與模型計算出的分數。

1. 新增最終實例 mask 的範圍 loss：FN 權重 0.7 的 Tversky，加上前景／背景分別平均的 BCE，二者各占一半。只計算 matched instance 的 GT 框外擴 1 個 mask pixel 區域；沒有前景的實例不加入此項。保留原 BCE，避免只鼓勵擴張。
2. training_aux.extent 預設 gain=0.0、fn_weight=0.7、margin=1.0；啟用 gain=0.2。作用於實際 mask coefficients/prototypes，加入 seg_loss，沿用正 anchor 數量及 hyp.box 正規化。
3. 建立六個完整 model YAML 及六個 train recipe：01 baseline；02 shadow；03 consistency；04 extent；05 boundary（基於 04，boundary gain=0.05）；06 detail960（基於 04，模型圖相同，只在配套 recipe 改 imgsz=960）。05/06 是 04 的對照分支，不是必須連續載入前一步權重。
4. 01–05 recipe imgsz=640；06 imgsz=960；所有 recipe epochs=200、batch=4、seed=928、optimizer=AdamW、lr0=0.0003、mosaic=0.0、mixup=0.0、scale=0.2、translate=0.05、hsv_v=0.2、amp=False。這些為共同起點，非已驗證最佳參數。data、pretrained 由執行時提供。
5. 加入讀取 VOC XML 與 YOLO polygon 標記的核對工具：同類框以最大總 IoU 做一對一指派，低於 0.1 不視為對應；輸出未對應物件、各方向範圍比及 bbox IoU。範圍比低於 0.75 只標示人工複核，不宣稱 mask 錯誤。工具不改標記。

## 驗收

- 完整遮罩的新增 loss 小於局部遮罩；全塗滿背景也有代價；飽和漏掉的前景仍有恢復梯度；空目標、FP16、框外、非正方形輸入皆有限且不修改 GT。
- 六個 YAML 的 graph/state shapes 與基底相同，stage 01 loss 完全一致；stage 04/05 額外梯度到達真正 mask 參數；完整 train/backward、empty batch、eval、EMA 正常。
- 六個 recipe 可由本機 get_cfg 解析；實際 last0926 權重與資料集類別覆寫後的模型嚴格相容。
- 核對工具測試漏標、兩個相鄰實例的一對一配對及非法 polygon，並在使用者新 XML 與已找到的本機舊 polygon 上輸出報告。
- 說明如何選資料、執行順序、公平比較、公式、可能副作用與實測限制。960 訓練是尺度對照；本輪不新增會裁斷實例的線上裁切管線。
