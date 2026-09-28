import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { Presentation, PresentationFile, FileBlob } from '@oai/artifact-tool';

const workspaceDir = 'C:/Users/USER/Documents/iim_0905/Yolov11_AttnRes';
const SKILL_DIR = 'C:/Users/USER/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations';
const TMP_DIR = path.join(workspaceDir, '.codex-pptx-semantic-0927-build');
const FINAL_PPTX = path.join(workspaceDir, 'docs/presentations/yolo11-6csar-semantic-token-2pages.pptx');
const RUNTIME_PYTHON = 'C:/Users/USER/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const { resolvePresentationFont, finalizePresentation } = await import(pathToFileURL(path.join(SKILL_DIR, 'container_tools/artifact_tool_utils.mjs')).href);
const FONT = resolvePresentationFont({fontFamily:'Noto Sans TC'});
const NAVY='#182E44', TEAL='#006E77', GRAY='#4C5D6D', WHITE='#FFFFFF', LIGHT='#EFF4F6';
const presentation=Presentation.create({slideSize:{width:1280,height:720}});

function text(slide,name,value,x,y,w,h,size=24,color=NAVY,bold=false){
  const box=slide.shapes.add({geometry:'textbox',name,position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  box.text=value;
  box.text.style={typeface:FONT,fontSize:size,color,bold,autoFit:'none'};
  return box;
}
function table(slide,values,x,y,w,h,widths,size=24){
  const tab=slide.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,columnWidths:widths,values});
  tab.borders.assign({style:'solid',fill:'#D4DFE5',width:0.8});
  for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){
    const cell=tab.getCell(r,c);
    cell.fill=r===0?NAVY:(r%2===0?LIGHT:WHITE);
    cell.text.style={typeface:FONT,fontSize:size,color:r===0?WHITE:(c===0?TEAL:NAVY),bold:r===0||c===0};
  }
  return tab;
}
const report=await fs.readFile(path.join(workspaceDir,'docs/yolo11-6csar-semantic-token-shadow-overlap-math.md'),'utf8');
const split=report.indexOf('## 5. 重疊監督');
const sources='\n\n來源：ultralytics/cfg/models/11_myself/yolo11-6csar-semantic-token.yaml；ultralytics/nn/modules/damage_semantic.py；ultralytics/nn/modules/block.py；ultralytics/nn/modules/head.py；ultralytics/utils/loss.py。一般公式背景：https://arxiv.org/html/1706.03762v7、https://arxiv.org/abs/1607.06450、https://arxiv.org/abs/1512.03385。概念參考：https://hackmd.io/8gre3-RoQsG02dfl_OER_Q。以 2026-09-27 工作區程式為準。';

{
  const s=presentation.slides.add(); s.background.fill=WHITE;
  text(s,'title','語意 token 架構與陰影中的損傷線索',60,35,1160,70,44,NAVY,true);
  text(s,'subtitle','保留原 6CSAR，在四個尺度加入各損傷類別的獨立查詢',62,113,1150,42,25,GRAY);
  text(s,'layer-heading','新增層與原有特徵的對應',62,174,552,42,27,TEAL,true);
  table(s,[
    ['尺度','來源層','Token 層','特徵 / logits'],
    ['P3','18','21','22 / 23'],
    ['P2','2','24','25 / 26'],
    ['P4','19','27','28 / 29'],
    ['P5','20','30','31 / 32'],
  ],62,225,548,250,[95,105,132,216],24);
  text(s,'layer-context','第 0–20 層沿用原模型\nP2 是原有淺層捷徑，保留局部細節\n第 33 層整合強化特徵與多標籤 logits',63,490,550,111,23,GRAY);

  text(s,'math-heading','類別查詢與語意特徵回饋',670,174,547,42,27,TEAL,true);
  text(s,'attention-equation','A = softmax空間(QKᵀ / √32)',670,224,550,52,30,NAVY,true);
  text(s,'attention-meaning','Q 來自損傷 token，K / V 來自影像\n4 heads，各 32 維，T′ 為更新後的 token',670,284,548,72,23,GRAY);
  text(s,'response-equation','R = σ(Q′K′ᵀ / √128 + b)',670,363,548,50,30,NAVY,true);
  text(s,'response-meaning','Q′、K′ 為 token 與視覺的再投影\n每類各自取 sigmoid，同位置可同時高分',670,418,548,72,23,GRAY);
  text(s,'residual-equation','X̂ = X + α P[(RᵀVₜ) / K]',670,492,548,45,30,NAVY,true);
  text(s,'residual-meaning','Vₜ：token 投影　P：輸出投影\nK：類別數　α 初值：0.001',670,540,548,66,22.7,GRAY);
  text(s,'shadow-takeaway','陰影主軸：P2 細節與類別上下文，可能幫助暗處損傷辨識',62,614,1155,42,27,TEAL,true);
  text(s,'shadow-limitation','此版使用可學習類別 token，沒有文字編碼器或陰影專用監督；成效待實驗驗證。',62,661,1128,36,22.7,GRAY);
  s.speakerNotes.textFrame.setText(report.slice(0,split)+sources);
}
{
  const s=presentation.slides.add(); s.background.fill=WHITE;
  text(s,'title','重疊損傷的監督與辨識影響',60,35,1160,70,44,NAVY,true);
  text(s,'subtitle','同一位置可同時保留 Rust 與 Dent 的正標籤',62,113,1150,42,25,GRAY);

  text(s,'response-heading','獨立 response 與 multi-hot 真值',62,174,545,42,27,TEAL,true);
  text(s,'response-example','σ(2) = 0.881　σ(1.5) = 0.818',62,228,555,48,29,NAVY,true);
  text(s,'response-example-meaning','重疊位置的 (Rust, Dent) 真值為 (1, 1)\n兩類都可高分，示例數值並非訓練結果',62,286,555,72,23,GRAY);

  text(s,'loss-heading','共存區域加權，四尺度等權平均',660,174,560,42,27,TEAL,true);
  text(s,'overlap-equation','w = 1 + 2 × 1[ΣₖYₖ > 1]',660,226,558,46,28,NAVY,true);
  text(s,'loss-equation','Laux = 0.5 × mean尺度(0.5 LwBCE + 0.5 LDice)',660,278,558,42,23,NAVY,true);
  text(s,'loss-explanation','BCE 的相對權重為 3:1，Dice 不另加權',660,324,558,38,23,GRAY);

  table(s,[
    ['情境','本版作用','需驗證的影響'],
    ['陰影內損傷','P2 細節與 token 上下文','暗處 recall、純陰影誤報'],
    ['Rust + Dent 重疊','multi-hot 與加權多標籤 loss','配對 recall、非重疊區誤報'],
    ['最終框與 masks','語意特徵送回 YOLO head','assignment、NMS 仍可能漏類'],
  ],62,384,1156,220,[218,449,489],23);
  text(s,'training','訓練設定：overlap_mask=False，mask_ratio=4',62,626,1148,39,25,TEAL,true);
  text(s,'cost','n、3 類參數：5.53M 增至 6.30M（+13.9%）。尚無此版精度增益數據。',62,669,1150,33,22.7,GRAY);
  s.speakerNotes.textFrame.setText(report.slice(split)+sources+'\n投影片公式省略 batch 與尺度下標。w 的定義來自每個尺度經 max pooling 的 multi-hot 目標。例子是數學示意。參數量由 n 尺度、nc=3 的本機模型實例統計，非 FPS 或顯存測量。');
}

const candidatePath=path.join(TMP_DIR,'candidate.pptx');
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);
const result=await finalizePresentation({
  explicitTotalSlideCount:2, requiredNativeTableOwnerSlides:[1,2], requiredNativeChartOwnerSlides:[],
  workspaceDir,candidatePath,finalPath:FINAL_PPTX,pythonExecutable:RUNTIME_PYTHON,
  integrityValidatorPath:path.join(SKILL_DIR,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(SKILL_DIR,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit','--require-native-table-slide','1','--require-native-table-slide','2'],
  fontPolicy:{basis:'design',families:[FONT]}, verifyArtifactToolImport:true,
  receiptPath:path.join(TMP_DIR,'validation-final.json'),
});
console.log(JSON.stringify({final:FINAL_PPTX,result}));
const finalDeck=await PresentationFile.importPptx(await FileBlob.load(FINAL_PPTX));
for(let i=0;i<2;i++){
  const slide=finalDeck.slides.getItem(i);
  try{
    const preview=await finalDeck.export({slide,format:'png',scale:1.5});
    await fs.writeFile(path.join(TMP_DIR,`slide-${i+1}.png`),new Uint8Array(await preview.arrayBuffer()));
  }catch(e){console.log('Preview error: '+e.message);}
  const layout=await slide.export({format:'layout'});
  await fs.writeFile(path.join(TMP_DIR,`slide-${i+1}.layout.json`),await layout.text());
}
