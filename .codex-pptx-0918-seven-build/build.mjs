import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = "C:\\Users\\USER\\Documents\\iim_0905\\Yolov11_AttnRes";
const SKILL_DIR = "C:\\Users\\USER\\.codex\\plugins\\cache\\openai-primary-runtime\\presentations\\26.905.11957\\skills\\presentations";
const TMP_DIR = path.join(workspaceDir, ".codex-pptx-0918-seven-build");
const OUTPUT_DIR = path.join(workspaceDir, "docs", "presentations", "0918-seven");
const RUNTIME_PYTHON = "C:\\Users\\USER\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe";
const { resolvePresentationFont, finalizePresentation } = await import(
  pathToFileURL(path.join(SKILL_DIR, "container_tools", "artifact_tool_utils.mjs")).href
);
await fs.mkdir(TMP_DIR, { recursive: true });
await fs.mkdir(OUTPUT_DIR, { recursive: true });
const FONT = resolvePresentationFont({ fontFamily: "Noto Sans TC" });
const NAVY = "#14243A", TEAL = "#007C81", GREY = "#475569", LIGHT = "#F3F6F8", WHITE = "#FFFFFF";
const photoPath = "C:\\Users\\USER\\Documents\\推論結果\\0912推論結果\\predict-2\\000000_CBHU0708054-A\\CBHU0708054-A__all.jpg";
const photoBytes = new Uint8Array(await fs.readFile(photoPath));
const yamlPrefix = "ultralytics/cfg/models/11_myself/yolo11-10csar-dr-trainonly-0918-";

const cases = [
  {
    slug: "baseline",
    title: "Baseline：0912 原始多標籤目標",
    subtitle: "新增陰影、一致性、邊界、真交集增益均為 0",
    steps: [
      ["獨立實例保留 D／R 重疊", "Yc(u) = maxᵢ:cᵢ=c Mi(u)", "同一像素可同時標為 D 與 R。訓練必須使用 overlap_mask=False。"],
      ["既有共存權重", "w(u) = 1 + 2·1[Σc Yc(u) > 1]", "兩類以上重疊像素的 BCE 權重為 3，其餘為 1。"],
      ["既有五尺度多標籤目標", "LMSAT = ½ L加權BCE + ½ LDice", "五尺度平均後以原 gain 0.5 納入 semantic loss。"],
    ],
    callout: "本版沒有額外陰影訓練，作為其他六版的共同對照。",
    rows: [
      ["影像辨識作用", "獨立 sigmoid 與 mask 保留 D、R 同時為前景的能力。共存加權減少稀少交疊像素被背景淹沒。"],
      ["陰影下的 D", "陰影中的凹洞只靠原資料標註學習。沒有本輪的合成陰影或光照一致性約束。"],
      ["可能限制", "暗處細微凹洞仍可能漏檢，正常暗紋也可能誤報。這些是待量測問題，不能從單張疊圖量化。"],
      ["驗證對照", "固定資料切分與推論閾值，分別記錄陰影 D、亮部 D、D+R 和 D-only 的 recall、mask AP。"],
    ],
    footer: "與 shadow-aug 比較可測合成陰影曝光的額外效果。主架構與推論輸出保持相同。",
  },
  {
    slug: "shadow-aug",
    title: "Shadow Aug：合成陰影雙視角",
    subtitle: "僅增加陰影影像與 GT 監督；一致性 gain = 0",
    steps: [
      ["隨機柔邊半平面", "d(u) = r sinφ + c cosφ − o", "方向 φ 與偏移 o 各圖獨立取樣，r、c 為正規化座標。"],
      ["光度變換，幾何不變", "x̃ = clip[x·(1 − aσ(d / 0.15)), 0, 1]", "a ∈ [0.15, 0.45]；RGB 同比例變暗，mask 與 bbox 保持原標註。"],
      ["原圖與陰影圖都做 GT 監督", "Lpair = ½[L₀(x,Y) + L₀(x̃,Y)]", "同一模型一次 forward 處理 2B 個視角，兩視角 loss 取平均。"],
    ],
    callout: "此版回答：陰影曝光本身能否改善暗處 D，尚未加入一致性。",
    rows: [
      ["影像辨識作用", "相同破損幾何搭配不同亮度，使模型少依賴暗度單一線索。陰影區 D 標籤不變。"],
      ["陰影下的 D", "預期可能減少低亮度造成的漏檢；也須檢查純陰影是否被誤認為凹洞。尚無實測提升證據。"],
      ["代價與風險", "一次處理 2B 視角會增加運算、顯存並改變 BatchNorm 統計。半平面陰影未必符合真實反光與遮蔽。"],
      ["驗證對照", "與 baseline 比陰影／亮部 D recall、純陰影誤報和訓練成本；與 consistency 比可隔離新增一致性。"],
    ],
    footer: "本版不使用陰影 mask 作目標，也不新增陰影類別輸出。",
  },
  {
    slug: "consistency",
    title: "Consistency：陰影前後輸出穩定",
    subtitle: "與 shadow-aug 相同的雙視角；另加 gain 0.2 的五尺度一致性",
    steps: [
      ["成對監督仍保留", "Lpair = ½[L₀(x,Y) + L₀(x̃,Y)]", "陰影圖不靠 teacher 取代 GT，兩個視角都受真實 mask 監督。"],
      ["只有 teacher 符合 GT 才比較", "P = {Y=1, t≥0.7}；N = {Y=0, t≤0.3}", "t 是原圖 sigmoid 輸出並 stop-gradient；P、N 分別平均。"],
      ["拉近陰影與原圖機率", "LC = ⅕Σs meanP,N[(ps − stopgrad(ts))²]", "陰影 student 在五個 MSAT 尺度計算；總 loss 增加 0.2LC。"],
    ],
    callout: "原圖若漏掉真 D，該像素不當 teacher；仍由兩視角 GT loss 學習。",
    rows: [
      ["影像辨識作用", "對原圖已判對的 D，要求陰影圖保留相近機率。對 GT 背景，可信原圖低分可壓制陰影假陽性。"],
      ["陰影下的 D", "預期可減少同一凹洞在亮暗兩視角的預測落差。前景與背景分開平均，避免大量背景主導。"],
      ["代價與風險", "teacher 若對困難 D 不夠可靠，該處一致性為零。極端飽和 sigmoid 的一致性梯度也可能很弱。"],
      ["驗證對照", "優先與 shadow-aug 比較，兩者同為 2B 視角。量測陰影 D recall、純陰影誤報與預測落差。"],
    ],
    footer: "teacher 是同一模型原圖分支，沒有 EMA、第二套權重或陰影類別。",
  },
  {
    slug: "boundary",
    title: "Boundary：實例遮罩的輪廓監督",
    subtitle: "無合成陰影；新增 matched-instance 邊界項，gain 0.1",
    steps: [
      ["沿用原 segmentation 輸出", "zi(u) = Σk αik Pk(u)；pi = σ(zi)", "mask coefficients 與 prototypes 形成每個匹配實例的預測。"],
      ["由 mask 產生柔性輪廓", "B(F) = max₃×₃(F) − min₃×₃(F)", "對預測機率與 GT mask 各算邊界，不對 RGB 的暗線取邊。"],
      ["邊界貼合與邊帶分類", "LBi = ½[edge Dice + GT邊帶 BCE]", "各 matched mask 加 0.1LBi，再沿用正 anchor 正規化與原 seg gain。"],
    ],
    callout: "形狀訊號來自損傷 mask；照片中的陰影線不被當作邊界 GT。",
    rows: [
      ["影像辨識作用", "區域重疊之外，額外強調實例輪廓位置。Dice 管整體對齊，GT 邊帶 BCE 提供逐像素方向。"],
      ["陰影下的 D", "若陰影使凹洞局部對比變弱，輪廓目標可能幫助 mask 形狀；但本版沒有光照不變訓練。"],
      ["代價與風險", "標註邊緣過粗或漏標會放大噪音。此項只作用於已匹配實例，無法保證產生原本完全缺失的 D proposal。"],
      ["驗證對照", "與 baseline 比暗處與亮部 D 的 mask AP、邊界品質及假陽性；另看所有損傷類別是否受影響。"],
    ],
    footer: "boundary loss 作用於所有實例類別，不只 D；推論時不需要額外邊界 head。",
  },
  {
    slug: "overlap",
    title: "Overlap：凹洞與生鏽真交集",
    subtitle: "無合成陰影；在原有共存加權之外，新增 D／R joint loss",
    steps: [
      ["原生 GT 解析度先求交集", "T = YD ∩ YR；U = YD ∪ YR", "先交集再縮圖，避免相鄰 D、R 被粗尺度捏成假交疊。"],
      ["兩類同時成立的分數", "qDR = σ(zD)·σ(zR)", "它是訓練分數，不宣稱已校準的統計聯合機率。"],
      ["穩定 joint-logit 監督", "logit(qDR) = a+b−logsumexp(0,a,b)", "僅真交集圖片啟用，五尺度平均後以 gain 0.2 加入 sem_loss。"],
    ],
    callout: "D-only 樣本的新增交集 loss 為 0，不要求凹洞一定伴隨生鏽。",
    rows: [
      ["影像辨識作用", "原本的共存 BCE 權重已是 3；新增項再要求 D 與 R 的乘積對準真正同像素交疊。"],
      ["陰影下的 D", "若鏽蝕覆蓋凹洞且同處陰影，交疊目標可能保留兩類訊號；本版本身沒有陰影增強。"],
      ["代價與風險", "真交集圖太少時梯度稀疏。交疊標註若不準，可能拉高 R 假陽性或傷害 D-only 預測。"],
      ["驗證對照", "與 baseline 比 D+R 真交集 recall、D-only recall、R precision；相鄰但未重疊樣本應獨立檢查。"],
    ],
    footer: "原本的 D／R 共存能力保持不變；本版測試的是「額外真交集約束」。",
  },
  {
    slug: "shadow-boundary",
    title: "Shadow + Boundary：暗處形狀辨識",
    subtitle: "雙視角監督 + 一致性 0.2 + 實例邊界 0.1；新交集 gain = 0",
    steps: [
      ["合成陰影保持真值", "x̃ = clip[x·(1 − aσ(d / 0.15)), 0, 1]", "同一凹洞在原圖與陰影圖都使用原 GT。"],
      ["兩視角都學實例邊界", "LA(x) = L₀(x) + 0.1 LB(x)", "LB 包含邊界 Dice 與 GT 邊帶 BCE，實際還經原 seg 正規化。"],
      ["以可信原圖穩定陰影預測", "L = ½[LA(x)+LA(x̃)] + 0.2 LC", "LC 僅在原圖預測符合 GT 且信心達門檻時啟用。"],
    ],
    callout: "同時訓練暗處的分類穩定性與破損輪廓，沒有新增推論分支。",
    rows: [
      ["影像辨識作用", "陰影擾動讓模型接觸低亮度；邊界項要求保留 mask 幾何；一致性減少亮暗預測落差。"],
      ["陰影下的 D", "最直接對應「陰影中的凹洞漏檢且輪廓弱」。改善只是研究假說，須與單一機制比較。"],
      ["代價與風險", "2B 視角增加計算與顯存。錯誤邊界標註可放大噪音；可信 teacher 太少會削弱一致性。"],
      ["驗證對照", "與 consistency 比可隔離邊界貢獻；與 boundary 比會同時加入陰影曝光與一致性，需謹慎解讀。"],
    ],
    footer: "原 0912 的多標籤共存加權仍在；本版只是未啟用新增的真交集 loss。",
  },
  {
    slug: "full",
    title: "Full：陰影、邊界與 D／R 真交集",
    subtitle: "雙視角監督 + 一致性 0.2 + 邊界 0.1 + 真交集 0.2",
    steps: [
      ["原圖與合成陰影共用 GT", "x̃ = clip[x·(1 − aσ(d / 0.15)), 0, 1]", "陰影改光度，不改 D、R 的實例 mask 與位置。"],
      ["每個視角加入形狀與交集", "LA = L₀ + 0.1 LB + 0.2 LO", "LB 約束實例輪廓；LO 只監督真 D/R 交集圖片。"],
      ["可信原圖約束陰影視角", "Lfull = ½[LA(x)+LA(x̃)] + 0.2 LC", "LC 在五尺度上前背景分組平均，原圖 teacher stop-gradient。"],
    ],
    callout: "三種新增目標都只在訓練期作用，推論仍使用同一 0912 計算圖。",
    rows: [
      ["影像辨識作用", "同時約束光照穩定性、實例幾何和 D／R 同像素共存，檢驗三者是否互補。"],
      ["陰影下的 D", "研究假說是找回暗處 D，同時維持凹洞輪廓與覆鏽交集；不能預設 full 一定優於單一機制。"],
      ["代價與風險", "2B 視角增加成本。三項 loss 可能競爭梯度，交集標註錯誤也可能傷 R precision。"],
      ["驗證對照", "與 shadow-boundary 比可隔離新真交集項。同步報陰影 D、亮部 D、D-only、D+R 與純陰影誤報。"],
    ],
    footer: "所有新增 gain 都是起始超參數。以同一 0912 權重各自初始化七版，不能接續訓練。",
  },
];

function addText(slide, name, value, x, y, w, h, size, color = NAVY, bold = false) {
  const shape = slide.shapes.add({
    geometry: "textbox", name,
    position: { left: x, top: y, width: w, height: h },
    fill: "none", line: { fill: "none", width: 0 },
  });
  shape.text = value;
  shape.text.style = { typeface: FONT, fontSize: size, color, bold, autoFit: "none", wrap: true };
}

function addFirstSlide(presentation, item) {
  const slide = presentation.slides.add();
  slide.background.fill = WHITE;
  addText(slide, "title", item.title, 68, 43, 1136, 63, 43, NAVY, true);
  addText(slide, "subtitle", item.subtitle, 70, 109, 1130, 35, 20, GREY);
  const ys = [178, 329, 480];
  for (let i = 0; i < item.steps.length; i++) {
    const [label, formula, note] = item.steps[i];
    addText(slide, "step-" + i, String(i + 1) + "  " + label, 70, ys[i], 685, 34, 24, TEAL, true);
    addText(slide, "formula-" + i, formula, 73, ys[i] + 41, 685, 48, formula.length > 47 ? 22 : 25, NAVY, true);
    addText(slide, "explain-" + i, note, 73, ys[i] + 93, 684, 48, 18, GREY);
  }
  slide.images.add({
    blob: photoBytes, contentType: "image/jpeg",
    alt: "使用者提供的 0912 貨櫃損傷推論示例，含陰影區域及 D、R 預測標記",
    fit: "contain", position: { left: 802, top: 192, width: 410, height: 308 },
  });
  addText(slide, "photo-caption", "0912 推論示例（非 GT）", 803, 510, 405, 27, 17, GREY);
  addText(slide, "callout", item.callout, 802, 562, 408, 96, 21, TEAL, true);
  slide.speakerNotes.textFrame.setText(
    "對應 YAML：" + yamlPrefix + item.slug + ".yaml。數學與訓練流程來源：ultralytics/utils/training_aux_ops.py、ultralytics/utils/training_aux_0918.py、ultralytics/utils/loss.py。圖片為使用者提供的 0912 推論疊圖，不是 GT，也不是本次七版的測試結果。"
  );
}

function addSecondSlide(presentation, item) {
  const slide = presentation.slides.add();
  slide.background.fill = WHITE;
  addText(slide, "title", item.title + "：影響與驗證", 68, 43, 1136, 62, 42, NAVY, true);
  addText(slide, "subtitle", "研究目標是辨識陰影中的損傷。此模型沒有輸出陰影類別或陰影 mask。", 70, 109, 1130, 37, 21, GREY);
  const table = slide.tables.add({
    rows: item.rows.length, columns: 2,
    left: 69, top: 171, width: 1142, height: 432,
    columnWidths: [215, 927], values: item.rows,
  });
  table.borders.assign({ style: "solid", fill: "#D8E2E7", width: 1 });
  for (let row = 0; row < item.rows.length; row++) {
    for (let col = 0; col < 2; col++) {
      const cell = table.getCell(row, col);
      cell.fill = row % 2 === 0 ? WHITE : LIGHT;
      cell.text.style = { typeface: FONT, fontSize: col === 0 ? 20 : 19, bold: col === 0, color: col === 0 ? TEAL : NAVY };
    }
  }
  addText(slide, "footer", item.footer, 70, 628, 1135, 61, 20, GREY);
  slide.speakerNotes.textFrame.setText(
    "對應 YAML：" + yamlPrefix + item.slug + ".yaml。機制來源：ultralytics/utils/training_aux_ops.py、ultralytics/utils/training_aux_0918.py、ultralytics/utils/loss.py。所有改善敘述均為待驗證假說，目前沒有完整真實貨櫃資料集的七版結果。"
  );
}

for (const item of cases) {
  const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });
  addFirstSlide(presentation, item);
  addSecondSlide(presentation, item);
  const candidatePath = path.join(TMP_DIR, item.slug + "-candidate.pptx");
  const finalPath = path.join(OUTPUT_DIR, "0918-" + item.slug + "-2slides.pptx");
  await (await PresentationFile.exportPptx(presentation)).save(candidatePath);
  await finalizePresentation({
    explicitTotalSlideCount: 2,
    requiredNativeTableOwnerSlides: [2],
    requiredNativeChartOwnerSlides: [],
    workspaceDir, candidatePath, finalPath,
    pythonExecutable: RUNTIME_PYTHON,
    integrityValidatorPath: path.join(SKILL_DIR, "container_tools", "inspect_presentation_package_integrity.py"),
    layoutValidatorPath: path.join(SKILL_DIR, "container_tools", "inspect_presentation_layout_geometry.py"),
    layoutArgs: ["--expected-slide-size-emu", "12192000,6858000", "--validate-bullet-geometry", "--validate-heading-fit", "--require-native-table-slide", "2"],
    fontPolicy: { basis: "design", families: [FONT] },
    verifyArtifactToolImport: true,
    receiptPath: path.join(TMP_DIR, item.slug + "-validation.json"),
  });
  console.log(item.slug + ": " + finalPath);
}
