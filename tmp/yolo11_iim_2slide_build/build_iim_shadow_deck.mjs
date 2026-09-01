import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const BUILD_DIR = "C:\\Users\\USER\\Documents\\old_version_iim\\Yolov11_AttnRes\\tmp\\yolo11_iim_2slide_build";
const STARTER = path.join(BUILD_DIR, "template-starter.pptx");
const FINAL = "C:\\Users\\USER\\Documents\\old_version_iim\\Yolov11_AttnRes\\yolo11-6csar-iim_陰影辨識_2頁.pptx";
const RENDER_DIR = path.join(BUILD_DIR, "final-render");
const LAYOUT_DIR = path.join(BUILD_DIR, "final-layout", "final");

async function writeBlob(filePath, blob) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

function parseInspect(ndjson) {
  return ndjson
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

function findTextRecord(records, slide, exactText) {
  const match = records.find(
    (record) => record.kind === "textbox" && record.slide === slide && record.text === exactText,
  );
  if (!match) throw new Error(`Missing inherited textbox on slide ${slide}: ${exactText}`);
  return match;
}

function setInheritedText(presentation, records, slide, oldText, newText, style) {
  const record = findTextRecord(records, slide, oldText);
  const target = presentation.resolve(record.id);
  target.text = newText;
  target.text.style = style;
  return target;
}

async function main() {
  await fs.mkdir(RENDER_DIR, { recursive: true });
  await fs.mkdir(LAYOUT_DIR, { recursive: true });

  const presentation = await PresentationFile.importPptx(await FileBlob.load(STARTER));
  const before = await presentation.inspect({
    kind: "slide,textbox,shape,image,notes,layout",
    maxChars: 60000,
  });
  const records = parseInspect(before.ndjson || "");

  setInheritedText(
    presentation,
    records,
    1,
    "第五版：重疊損傷 Multi-label 監督",
    "第六版：IIM 處理陰影，6-CSAR 主體不變",
    {
      fontSize: 46,
      typeface: "Microsoft JhengHei",
      color: "#000000",
      bold: true,
      alignment: "left",
      verticalAlignment: "top",
      wrap: "square",
      autoFit: "shrinkText",
      insets: { top: 0, right: 0, bottom: 0, left: 0 },
    },
  );

  setInheritedText(
    presentation,
    records,
    1,
    "yolo11-0621-2 架構",
    "原 6-CSAR：下游架構與層號不變",
    {
      fontSize: 30,
      typeface: "Microsoft JhengHei",
      color: "#000000",
      bold: true,
      alignment: "left",
      verticalAlignment: "top",
      wrap: "square",
      autoFit: "shrinkText",
      insets: { top: 0, right: 0, bottom: 0, left: 0 },
    },
  );

  setInheritedText(
    presentation,
    records,
    1,
    "同一像素允許多種損傷",
    "新 IIMStem：雙分支抑制陰影偏移",
    {
      fontSize: 28,
      typeface: "Microsoft JhengHei",
      color: "#3D8DFF",
      bold: true,
      alignment: "left",
      verticalAlignment: "top",
      wrap: "square",
      autoFit: "shrinkText",
      insets: { top: 0, right: 0, bottom: 0, left: 0 },
    },
  );

  setInheritedText(
    presentation,
    records,
    1,
    "T_c(h,w) = max_{j:y_j=c} M_j(h,w)\np_c(h,w) = σ(z_c),  not softmax_c\nL_ov = 0.5 BCE_w + 0.5 Dice",
    "x_c=ln(max(I_c,ε));  K′_m=K_m−μ(K_m)\nD_m^ab=K′_m*(x_a−x_b),  ab=RG/GB/RB\nF₀=Conv₁×₁([F_RGB ‖ F_IIM])",
    {
      fontSize: 20,
      typeface: "Cascadia Mono",
      color: "#000000",
      alignment: "left",
      verticalAlignment: "middle",
      wrap: "square",
      autoFit: "shrinkText",
      insets: { top: 0, right: 0, bottom: 0, left: 0 },
    },
  );

  setInheritedText(
    presentation,
    records,
    1,
    "01  AR＋Shuffle：保留紋理與深層語意\n02  2×CSAR：重疊區重新選擇尺度\n03  Segment：prototype 分離每個 instance",
    "01  8 個共享 3×3 核 × 3 色差 → 24 張 IIM maps\n02  RGB、IIM 各輸出 c₂/2；Concat→1×1 回到 c₂\n03  stride=2 維持 P1/2；層號 0–21 與 head 索引不變",
    {
      fontSize: 21,
      typeface: "Microsoft JhengHei",
      color: "#000000",
      alignment: "left",
      verticalAlignment: "top",
      wrap: "square",
      autoFit: "shrinkText",
      insets: { top: 0, right: 0, bottom: 0, left: 0 },
    },
  );

  setInheritedText(
    presentation,
    records,
    2,
    "問題 03｜光線",
    "第六版｜陰影辨識機制",
    {
      fontSize: 14,
      typeface: "Arial",
      color: "#3D8DFF",
      bold: true,
      alignment: "left",
      verticalAlignment: "middle",
      wrap: "square",
      autoFit: "shrinkText",
      insets: { top: 4.8, right: 9.6, bottom: 4.8, left: 9.6 },
    },
  );

  setInheritedText(
    presentation,
    records,
    2,
    "光線改變的是像素分布，不只是亮或暗",
    "IIM 以對數色差抵消共同照度，並保留原始 RGB",
    {
      fontSize: 38,
      typeface: "Arial",
      color: "#111827",
      bold: true,
      alignment: "left",
      verticalAlignment: "middle",
      wrap: "square",
      autoFit: "shrinkText",
      insets: { top: 4.8, right: 9.6, bottom: 4.8, left: 9.6 },
    },
  );

  setInheritedText(
    presentation,
    records,
    2,
    "為什麼會發生",
    "陰影如何改變特徵",
    {
      fontSize: 24,
      typeface: "Arial",
      color: "#E45858",
      bold: true,
      alignment: "left",
      verticalAlignment: "middle",
      wrap: "square",
      autoFit: "shrinkText",
      insets: { top: 4.8, right: 9.6, bottom: 4.8, left: 9.6 },
    },
  );

  setInheritedText(
    presentation,
    records,
    2,
    "1. 過曝會剪裁亮部、欠曝會埋掉暗部：資訊不可逆消失\n\n2. 色溫、反光、陰影會改變紋理與類別外觀\n\n3. 訓練與現場相機的曝光／白平衡不同，形成 domain shift\n\n4. 自動曝光逐幀波動，導致信心值不穩定",
    "1. 成像：I_c=ρ_cL+n；陰影令 L→αL（0<α<1）\n\n2. 對數：x_c=lnρ_c+lnL；乘法照度轉為加法偏移\n\n3. 原始 K*I 同時學反射率與照度；暗部邊界、紋理與 BN 統計一起漂移\n\n4. 欠曝／截黑：x_c=ln ε；噪聲與量化誤差被放大",
    {
      fontSize: 21,
      typeface: "Arial",
      color: "#111827",
      alignment: "left",
      verticalAlignment: "middle",
      wrap: "square",
      autoFit: "shrinkText",
      insets: { top: 4.8, right: 9.6, bottom: 4.8, left: 9.6 },
    },
  );

  setInheritedText(
    presentation,
    records,
    2,
    "怎麼解",
    "IIM 如何抵消共同陰影",
    {
      fontSize: 24,
      typeface: "Arial",
      color: "#16856B",
      bold: true,
      alignment: "left",
      verticalAlignment: "middle",
      wrap: "square",
      autoFit: "shrinkText",
      insets: { top: 4.8, right: 9.6, bottom: 4.8, left: 9.6 },
    },
  );

  setInheritedText(
    presentation,
    records,
    2,
    "1. 先穩定相機：固定曝光／白平衡、補光、偏振片或遮光罩\n\n2. 收集真實極端光線資料；用 HSV、gamma、contrast、shadow 增強\n\n3. 推論前僅做一致的輕量校正，避免過度增強放大雜訊\n\n4. 依亮度分桶驗證，必要時做相機別校正或 domain adaptation",
    "1. 色差：D_m^ab=K′_m*(x_a−x_b)；共同 lnL 抵消\n\n2. 零均值：Σ_qK′_m(q)=0；移除 DC、強化局部色彩邊緣\n\n3. RGB‖IIM→1×1；穩定特徵再進 AR→Shuffle→2×CSAR→Segment26\n\n4. 有色光／反光／極暗會失效；以 shadow AP/Recall、Mask IoU、延遲/VRAM 分桶消融",
    {
      fontSize: 21,
      typeface: "Arial",
      color: "#111827",
      alignment: "left",
      verticalAlignment: "middle",
      wrap: "square",
      autoFit: "shrinkText",
      insets: { top: 4.8, right: 9.6, bottom: 4.8, left: 9.6 },
    },
  );

  setInheritedText(
    presentation,
    records,
    2,
    "原則：先修資料與成像，再調模型；否則只能把錯誤訊號學得更快。",
    "結論：暗部邊界與小物件可更穩定；代價是全解析度 log＋6 次共享卷積，且無法恢復已剪裁資訊。",
    {
      fontSize: 22,
      typeface: "Arial",
      color: "#3D8DFF",
      bold: true,
      alignment: "center",
      verticalAlignment: "middle",
      wrap: "square",
      autoFit: "shrinkText",
      insets: { top: 4.8, right: 9.6, bottom: 4.8, left: 9.6 },
    },
  );

  setInheritedText(
    presentation,
    records,
    2,
    "05",
    "02",
    {
      fontSize: 13,
      typeface: "Arial",
      color: "#586174",
      alignment: "right",
      verticalAlignment: "middle",
      wrap: "square",
      autoFit: "shrinkText",
      insets: { top: 4.8, right: 9.6, bottom: 4.8, left: 9.6 },
    },
  );

  const slide1 = presentation.slides.getItem(0);
  const slide2 = presentation.slides.getItem(1);

  slide1.speakerNotes.textFrame.setText(`報告重點：第六版只把 backbone 第 0 層由 Conv 改為 IIMStem；YAML 的總層數沒有增加，所以 P2/P3/P4/P5、兩階段 CSAR（15–20）以及 Segment26（21）的索引全部維持。

一、輸入與對數。程式先做 x_c(p)=ln(max(I_c(p), ε))，ε=10^-7。影像形成可近似 I_c=ρ_c L+n：ρ_c 是表面反射／材質，L 是照度，n 是感測雜訊。若陰影是 RGB 共同的乘法 α(p)，則 ln(αI_c)=lnα+lnI_c，把乘法陰影變成可相消的加法項。clamp 避免 ln0，但極暗區會集中到 lnε，因此不能把已剪裁資訊救回來。

二、零均值核心。每個可學習 3×3 核 K_m 投影為 K̃_m=K_m−mean_q K_m(q)，所以 Σ_qK̃_m(q)=0。這代表對任何空間常數 u，K̃_m*u=0；在影像辨識上相當於抑制 DC／平坦區，改看局部色彩邊緣、紋理轉折與材質變化。

三、共享核色差。程式以相同 K̃_m 分別卷積兩個 log 色頻，再相減：D_m^{ab}=K̃_m*x_a−K̃_m*x_b=K̃_m*(x_a−x_b)，ab∈{RG,GB,RB}。若 x_a=lnρ_a+lnL、x_b=lnρ_b+lnL，共同 lnL 會精確抵消，只留下 lnρ_a−lnρ_b。共享權重是此等式成立的關鍵；8 個核乘 3 個色對，產生 24 張 IIM 特徵圖。

四、BN 與穩定啟動。RG、GB、RB 各自做 BN(z)=γ(z−μ_B)/sqrt(σ_B²+δ)+β。γ 初始化為 0.01、β=0，使 IIM 分支在訓練初期振幅很小，模型先接近原 RGB stem，再逐步學會何時依賴陰影不變特徵，降低突然改 stem 的訓練震盪。

五、雙分支融合。F_RGB=SiLU(BN(Conv_{3×3,s=2}(I)))；F_IIM=SiLU(BN(Conv_{3×3,s=2}(D)))。兩支各輸出 c₂/2 通道，串接後以 F₀=SiLU(BN(Conv_{1×1}([F_RGB‖F_IIM]))) 回到 c₂。RGB 分支保留絕對亮度、色彩與灰階物體線索；IIM 分支補上對共同陰影較穩定的色彩邊界，避免純不變特徵犧牲有用的亮度訊號。

六、YAML 參數。IIMStem 的 args [64,3,2,8,3] 依序表示名義輸出通道 c₂=64、stem kernel=3、stride=2、IIM kernel 數=8、IIM kernel size=3；compound scale 仍會縮放 c₂。輸出仍是 P1/2，因此後續 6-CSAR 圖與 head 接點不改。

[Sources]
- Local: C:/Users/USER/Documents/old_version_iim/Yolov11_AttnRes/ultralytics/cfg/models/11_myself/yolo11-6csar-iim.yaml
- Local: C:/Users/USER/Documents/old_version_iim/Yolov11_AttnRes/ultralytics/cfg/models/11_myself/yolo11-6csar.yaml
- Local: C:/Users/USER/Documents/old_version_iim/Yolov11_AttnRes/ultralytics/nn/modules/conv.py (IlluminationInvariantConv, IIMStem)
- Visual template/architecture image: C:/Users/USER/Desktop/YOLO11_.pptx, source slide 16`);
  slide1.speakerNotes.setVisible(true);

  slide2.speakerNotes.textFrame.setText(`報告重點：這頁由陰影成像一路推到 IIM、下游辨識效益、代價與失效條件；可依 1→4 的順序逐式解釋。

一、陰影成像模型。I_c(p)=ρ_c(p)L(p)+n_c(p)。陰影令 L(p)→α(p)L(p)，0<α<1；因此同一材質在亮區與暗區會落在不同 RGB 分布。一般卷積 K*I 無法區分變化來自 ρ 還是 L，會把陰影邊界誤當物體邊界，或把真實邊界因低對比而壓弱；BN 的 batch mean/variance 也會隨照度分布改變。

二、對數域的作用。忽略很小的 n 時，x_c=lnI_c≈lnρ_c+lnL。這不是把影像變亮，而是把乘法照度拆成加法項。影像辨識上的影響是：後續做色頻差時可以代數消除共同照度，而不是要求網路用大量資料自己近似這種不變性。

三、色頻差相消。對任一色對 a,b：x_a−x_b=(lnρ_a+lnL)−(lnρ_b+lnL)=lnρ_a−lnρ_b；再以共享零均值核得到 D_m^{ab}=K̃_m*(x_a−x_b)。專案測試直接以空間變化的單通道 illumination 乘到 RGB，並檢查 IIM(image)≈IIM(image*illumination)，同時驗證 ΣK̃=0。

四、零均值不是同一件事。共同 lnL 的抵消主要來自「log＋通道差＋共享核」；ΣK̃=0 另外讓常數 log-chromaticity／DC 分量歸零，使輸出更像色彩高通。這會突出陰影下仍存在的材質邊界，但對純灰階、顏色相近或只有亮度差的目標可能較弱，所以保留 RGB 分支是必要折衷。

五、精確實作流程。程式先記錄 zero_mask，對 RGB clamp 後取 log；同一組 8 個 3×3 核用於 R、G、B；計算 RG、GB、RB，分別 BN；若配對的來源像素原本恰為 0，就把對應 response 設 0，避免 lnε 產生假強邊。24 張 IIM map 再經 stride-2 Conv 壓到 c₂/2，和 RGB stem 串接，最後 1×1 融合。

六、對本專案的正面影響。較穩定的 P1/2 輸入會沿 AttentionResiduals→FSNetShuffle 傳遞：AR 在同一像素選擇深度 state 時較不會被陰影幅度主導；Shuffle 跨尺度交換時能帶入較可靠的淺層邊界；兩階段 CSAR 再選尺度；Segment26 的 prototype/mask 可獲得較連續的暗部輪廓。最可能受益的是陰影內小物件召回、遮陰邊界、mask IoU/Boundary IoU 與不同曝光域的信心穩定性；這些是預期機制，不是尚未測得的提升數字。

七、成本。IIM 在下採樣前的全解析度 log RGB 上，對三個色對各做兩次 conv2d，實作上共 6 次共享核卷積；之後還有 IIM stride-2 Conv 和 1×1 fusion，因此 stem latency/VRAM 一定高於原單一 Conv。好處是額外成本集中在第 0 層，且後續層數與 head 索引不變。

八、失效與風險。有色光使各 channel 的 L_c 不同，lnL_a−lnL_b 不再為 0；鏡面反射、非 Lambertian 材質與飽和像素破壞乘法模型；極暗區 clamp 到 ε 會放大量化／read noise；IIM 也不能恢復過曝或截黑後已消失的訊息。評估時固定資料切分與訓練設定，只替換 stem，至少分 bright/shadow/deep-shadow、有色光、反光區，回報 box/mask AP、Recall、Mask IoU、Boundary IoU、latency、VRAM，並與 RGB-only 做 controlled ablation。

[Sources]
- Local: C:/Users/USER/Documents/old_version_iim/Yolov11_AttnRes/ultralytics/nn/modules/conv.py (forward path, zero mask, BN initialization)
- Local: C:/Users/USER/Documents/old_version_iim/Yolov11_AttnRes/tests/test_yolo11_6csar_iim.py (zero-mean and achromatic relighting invariance tests)
- Local: C:/Users/USER/Documents/old_version_iim/Yolov11_AttnRes/ultralytics/cfg/models/11_myself/yolo11-6csar-iim.yaml
- Visual template: C:/Users/USER/Desktop/YOLO11_.pptx, source slide 23`);
  slide2.speakerNotes.setVisible(true);

  for (const [index, slide] of presentation.slides.items.entries()) {
    const n = String(index + 1).padStart(2, "0");
    await writeBlob(
      path.join(RENDER_DIR, `slide-${n}.png`),
      await presentation.export({ slide, format: "png", scale: 2 }),
    );
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(path.join(LAYOUT_DIR, `slide-${n}.layout.json`), await layout.text(), "utf8");
  }

  await writeBlob(
    path.join(BUILD_DIR, "final-montage.webp"),
    await presentation.export({ format: "webp", montage: true, scale: 1 }),
  );

  const after = await presentation.inspect({
    kind: "slide,textbox,shape,image,notes,layout",
    maxChars: 100000,
  });
  await fs.writeFile(path.join(BUILD_DIR, "final-inspect.ndjson"), after.ndjson || "", "utf8");

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(FINAL);
  process.stdout.write(`${FINAL}\n`);
}

main().catch((error) => {
  console.error(error.stack || error.message || String(error));
  process.exitCode = 1;
});
