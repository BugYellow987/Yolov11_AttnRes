import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = "C:\\Users\\USER\\Documents\\iim_0905\\Yolov11_AttnRes";
const SKILL_DIR = "C:\\Users\\USER\\.codex\\plugins\\cache\\openai-primary-runtime\\presentations\\26.905.11957\\skills\\presentations";
const TMP_DIR = path.join(workspaceDir, ".codex-pptx-0918-build");
const FINAL_PPTX = path.join(workspaceDir, "docs", "presentations", "training-only-0918-shadow-math-v1.pptx");
const RUNTIME_PYTHON = "C:\\Users\\USER\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe";
const { resolvePresentationFont, finalizePresentation } = await import(
  pathToFileURL(path.join(SKILL_DIR, "container_tools", "artifact_tool_utils.mjs")).href
);
await fs.mkdir(TMP_DIR, { recursive: true });
await fs.mkdir(path.dirname(FINAL_PPTX), { recursive: true });
const FONT = resolvePresentationFont({ fontFamily: "Noto Sans TC" });
const NAVY = "#14243A", TEAL = "#007C81", GREY = "#475569", LIGHT = "#F3F6F8", WHITE = "#FFFFFF";
const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });

function textBox(slide, name, value, x, y, w, h, size, color = NAVY, bold = false) {
  const shape = slide.shapes.add({
    geometry: "textbox", name,
    position: { left: x, top: y, width: w, height: h },
    fill: "none", line: { fill: "none", width: 0 },
  });
  shape.text = value;
  shape.text.style = { typeface: FONT, fontSize: size, color, bold, autoFit: "none", wrap: true };
  return shape;
}

{
  const slide = presentation.slides.add();
  slide.background.fill = WHITE;
  textBox(slide, "title", "陰影下的凹洞辨識：主架構不變", 66, 43, 1150, 66, 45, NAVY, true);
  textBox(slide, "subtitle", "0912 YOLO11／IIMStem／AttentionResiduals／FSNetShuffle／10 CSAR／MSAT 皆保留；只改訓練", 69, 111, 1110, 34, 21, GREY);
  textBox(slide, "shadow-step", "1  合成陰影，保留原 GT", 69, 177, 685, 35, 25, TEAL, true);
  textBox(slide, "shadow-formula", "x̃ = clip[x × (1 − a·σ(d / 0.15)), 0, 1]", 72, 220, 670, 47, 27, NAVY, true);
  textBox(slide, "shadow-explain", "a ∈ [0.15, 0.45]；隨機方向的柔邊陰影。像素幾何、bbox 與 mask 不變。", 72, 268, 673, 58, 19, GREY);
  textBox(slide, "pair-step", "2  原圖與陰影圖都做完整監督", 69, 340, 685, 35, 25, TEAL, true);
  textBox(slide, "pair-formula", "Lpair = ½ [L₀(x, Y) + L₀(x̃, Y)]", 72, 381, 670, 45, 26, NAVY, true);
  textBox(slide, "pair-explain", "同一模型一次處理 2B 個視角；陰影中的 D 仍是 D，不能改成背景。", 72, 430, 675, 58, 19, GREY);
  textBox(slide, "cons-step", "3  只模仿正確且高信心的原圖預測", 69, 511, 685, 35, 25, TEAL, true);
  textBox(slide, "cons-formula", "Lcons = mean[(p陰影 − stopgrad(p原圖))²]", 72, 553, 690, 45, 24, NAVY, true);
  textBox(slide, "cons-explain", "GT 前景且 p原圖 ≥ 0.7，或 GT 背景且 p原圖 ≤ 0.3；五尺度平均，gain = 0.2。", 72, 601, 690, 54, 18, GREY);
  const photoPath = "C:\\Users\\USER\\Documents\\推論結果\\0912推論結果\\predict-2\\000000_CBHU0708054-A\\CBHU0708054-A__all.jpg";
  const imageBytes = await fs.readFile(photoPath);
  slide.images.add({
    blob: new Uint8Array(imageBytes), contentType: "image/jpeg",
    alt: "使用者提供的 0912 貨櫃損傷推論示例，可見不同明暗區及 D、R 預測標記",
    fit: "contain", position: { left: 800, top: 184, width: 412, height: 309 },
  });
  textBox(slide, "photo-caption", "0912 推論示例（非 GT）", 801, 502, 412, 26, 17, GREY);
  textBox(slide, "slide-note", "核心問題：讓模型辨識陰影中的破損，而非輸出「陰影」類別。", 800, 553, 407, 83, 21, TEAL, true);
  slide.speakerNotes.textFrame.setText("公式來源：ultralytics/utils/training_aux_ops.py 的 shadow_view、consistency_loss，以及 ultralytics/utils/training_aux_0918.py 的 training_auxiliary_loss。圖片由使用者提供；這是 0912 推論疊圖，不是 GT，不能據此宣稱七版效果。");
}

{
  const slide = presentation.slides.add();
  slide.background.fill = WHITE;
  textBox(slide, "title", "七版消融：陰影、輪廓與真 D／R 交集", 66, 43, 1150, 66, 44, NAVY, true);
  textBox(slide, "subtitle", "全部維持相同推論圖；原 0912 多標籤共存加權仍存在", 69, 111, 1110, 32, 21, GREY);
  const values = [
    ["YAML 版本", "雙視角", "一致性", "邊界", "真 D∩R", "主要研究問題"],
    ["baseline", "—", "—", "—", "—", "0912 基準"],
    ["shadow-aug", "✓", "—", "—", "—", "陰影曝光本身"],
    ["consistency", "✓", "0.2", "—", "—", "光照穩定性"],
    ["boundary", "—", "—", "0.1", "—", "凹洞輪廓"],
    ["overlap", "—", "—", "—", "0.2", "鏽蝕／凹洞真交疊"],
    ["shadow-boundary", "✓", "0.2", "0.1", "—", "暗處形狀辨識"],
    ["full", "✓", "0.2", "0.1", "0.2", "三者互補性"],
  ];
  const table = slide.tables.add({
    rows: values.length, columns: values[0].length,
    left: 68, top: 168, width: 1144, height: 373,
    columnWidths: [210, 113, 130, 110, 120, 461], values,
  });
  table.borders.assign({ style: "solid", fill: "#D8E2E7", width: 1 });
  for (let row = 0; row < values.length; row++) {
    for (let col = 0; col < values[0].length; col++) {
      const cell = table.getCell(row, col);
      cell.fill = row === 0 ? NAVY : (row % 2 === 0 ? LIGHT : WHITE);
      cell.text.style = {
        typeface: FONT, fontSize: 17, bold: row === 0 || col === 0,
        color: row === 0 ? WHITE : (col === 0 ? TEAL : NAVY),
      };
    }
  }
  textBox(slide, "boundary-detail", "邊界  B(M) = max₃×₃(M) − min₃×₃(M)；用實例 mask 的輪廓監督，不把 RGB 陰影線當 GT。", 70, 558, 1141, 47, 19, NAVY);
  textBox(slide, "overlap-detail", "交集  q(D∩R) = σ(zD)σ(zR)；只在真交疊影像啟用，D 不必伴隨 R。", 70, 606, 1141, 42, 19, NAVY);
  textBox(slide, "risk-detail", "驗證重點：陰影／亮部 D recall、D+R 重疊、R precision、純陰影誤報；雙視角增加計算與顯存。", 70, 654, 1141, 36, 17, GREY);
  slide.speakerNotes.textFrame.setText("七版來源：ultralytics/cfg/models/11_myself/yolo11-10csar-dr-trainonly-0918-*.yaml。邊界與交集來源：ultralytics/utils/training_aux_ops.py；訓練 loss 插入位置：ultralytics/utils/training_aux_0918.py。原有共存權重來源：ultralytics/utils/loss.py。這些是機制與待驗證假說，尚無真實資料集效益數據。");
}

const candidatePath = path.join(TMP_DIR, "candidate.pptx");
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);
for (let i = 0; i < presentation.slides.length; i++) {
  const preview = await presentation.export({ slide: presentation.slides.get(i), format: "png", scale: 1 });
  await fs.writeFile(path.join(TMP_DIR, "slide-" + (i + 1) + ".png"), new Uint8Array(await preview.arrayBuffer()));
}
const result = await finalizePresentation({
  explicitTotalSlideCount: 2,
  requiredNativeTableOwnerSlides: [2],
  requiredNativeChartOwnerSlides: [],
  workspaceDir, candidatePath, finalPath: FINAL_PPTX,
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR, "container_tools", "inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(SKILL_DIR, "container_tools", "inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", "12192000,6858000",
    "--validate-bullet-geometry", "--validate-heading-fit",
    "--require-native-table-slide", "2",
  ],
  fontPolicy: { basis: "design", families: [FONT] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(TMP_DIR, "validation.json"),
});
console.log(JSON.stringify({ final: FINAL_PPTX, result }, null, 2));
