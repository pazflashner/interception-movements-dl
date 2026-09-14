// Run using the bundled artifact-tool runtime; see production/README.md.
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';

const root=process.env.COURSE_REPO;
const skill=process.env.COURSE_SLIDE_SKILL;
if(!root||!skill)throw new Error('Set COURSE_REPO and COURSE_SLIDE_SKILL');
const {importRuntimeModule}=await import(pathToFileURL(path.join(skill,'container_tools/runtime_helpers.mjs')).href);
const {Presentation,PresentationFile}=await importRuntimeModule('@oai/artifact-tool');
const stage=path.join(root,'production/assets/course_slide_build');
const out=path.join(stage,'final');
await fs.mkdir(out,{recursive:true});
const content=JSON.parse(await fs.readFile(path.join(root,'production/course_slides.json'),'utf8'));
const p=Presentation.create({slideSize:{width:1280,height:720}});
const font='Arial';
const {applyPresentationChartFont}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
function text(s,value,x,y,w,h,size=26,bold=false,color='#161E2E'){
  const sh=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  sh.text=value;sh.text.style={typeface:font,fontSize:size,bold,color,autoFit:'none'};return sh;
}
async function pic(s,file,x,y,w,h,alt){
  file=path.resolve(root,file);
  s.images.add({blob:new Uint8Array(await fs.readFile(file)),contentType:'image/png',alt,fit:'contain',position:{left:x,top:y,width:w,height:h}});
}
const owners=[],chartOwners=[];
function chart(s,d,x,y,w,h){
  text(s,d.title,x,y,w,32,23,true);
  const c=s.charts.add('bar',{position:{left:x,top:y+40,width:w,height:h-40},categories:d.categories,series:d.series,
    barOptions:{direction:'column',grouping:'clustered',gapWidth:85},hasLegend:true,
    legend:{position:'bottom',textStyle:{fontSize:18,fill:'#161E2E'}},
    xAxis:{textStyle:{fontSize:18,fill:'#161E2E'}},
    yAxis:{min:0,numberFormatCode:'0.00',textStyle:{fontSize:16,fill:'#161E2E'}},dataLabels:{showValue:false}});
  applyPresentationChartFont(c,{fontFamily:font});
}
for(let i=0;i<content.length;i++){
  const d=content[i],s=p.slides.add();s.background.fill='#FFFFFF';
  // Paz's reference design: navy cover, blue-teal-violet top rule, white content slides.
  if(i===0)s.background.fill='#0E1729';
  const band=['#1B6FEA','#17A2A2','#7A5CF0'];
  for(let b=0;b<3;b++)s.shapes.add({geometry:'rect',position:{left:b*1280/3,top:0,width:1280/3,height:7},fill:band[b],line:{fill:'none',width:0}});
  if(i!==0)s.shapes.add({geometry:'rect',position:{left:56,top:150,width:1168,height:1},fill:'#E1E6EC',line:{fill:'none',width:0}});
  text(s,d.title,56,i===0?130:54,1168,i===0?130:94,i===0?48:36,true,i===0?'#FFFFFF':'#161E2E');
  let y=164;
  if(i===0){text(s,d.body,58,310,1100,165,27,false,'#A3B2C6');text(s,d.caption,58,530,1110,90,29,true,'#3EB1EC');}
  else {
    if(d.layout==='wide_image'){
      text(s,d.body,58,161,1160,65,25);await pic(s,d.image,58,238,1160,354,'Recorded movement examples and model outputs');
    }else if(d.layout==='image_chart'){
      text(s,d.body,58,161,1160,60,25);await pic(s,d.image,58,230,720,352,'Recorded and modeled trajectories or feature distributions');
      chart(s,d.charts[0],808,231,408,352);chartOwners.push(i+1);
    }else if(d.layout==='two_charts'){
      text(s,d.body,58,161,1160,60,25);
      if(d.charts.length===1)chart(s,d.charts[0],160,231,960,352);
      else d.charts.forEach((c,j)=>chart(s,c,58+j*600,231,560,352));
      chartOwners.push(i+1);
    }else if(d.image&&i===1){
      text(s,d.body,58,180,630,285,30);await pic(s,d.image,760,170,435,340,'Task display illustration');
    }else if(d.image){
      text(s,d.body,58,160,1160,115,25);await pic(s,d.image,60,288,1158,288,'Recorded and generated paths at equal x/y scale');
    }else{
      if(d.body){
        const lines=d.body.split('\n').length;
        const bodyH=d.table?(lines>2?112:60):(lines>2?155:112);
        text(s,d.body,58,y,1155,bodyH,d.table?23:29);y+=bodyH+14;
      }
      if(d.equation){await pic(s,d.equation,58,y,1120,68,'Mathematical definition');y+=86;}
      if(d.table){
        const rows=d.table.length,cols=d.table[0].length;
        const rh=rows>8?35:rows>6?44:52;
        const h=rows*rh;
        if(y+h>585) y=Math.min(y,585-h);
        const weights=[2,13].includes(i)?[.18,.41,.41]:i===18?[.22,.78]:cols===2?[.69,.31]:cols===3?[.46,.27,.27]:cols===4?[.30,.17,.265,.265]:[.30,...Array(cols-1).fill(.70/(cols-1))];
        const t=s.tables.add({rows,columns:cols,left:58,top:y,width:1160,height:h,columnWidths:weights.map(w=>w*1160),values:d.table});
        t.cells.block({row:0,column:0,rowCount:rows,columnCount:cols}).assign({textStyle:{typeface:font,fontSize:rows>8?23:22,color:'#161E2E'},margins:{left:12,right:10,top:rows>8?4:7,bottom:rows>8?4:6},anchor:'center'});
        for(let r=0;r<rows;r++)for(let c=0;c<cols;c++){
          const cell=t.getCell(r,c);cell.fill=r===0?'#161E2E':(r%2?'#F8FAFC':'#FFFFFF');
          cell.text.style={typeface:font,fontSize:rows>8?23:22,bold:r===0,color:r===0?'#FFFFFF':'#161E2E'};
        }
        owners.push(i+1);
      }
    }
    s.shapes.add({geometry:'rect',position:{left:56,top:599,width:1168,height:78},fill:'#EFF5FC',line:{fill:'none',width:0}});
    s.shapes.add({geometry:'rect',position:{left:56,top:599,width:4,height:78},fill:'#1B6FEA',line:{fill:'none',width:0}});
    text(s,d.caption,72,609,1136,62,19,false,'#3D4756');
  }
  text(s,`${i+1} / ${content.length}`,1160,690,70,20,14,false,i===0?'#A3B2C6':'#737E8B');
  s.speakerNotes.textFrame.setText(d.notes+`\nSuggested duration: ${d.seconds} seconds. Design reference: Paz Flashner's production/presentation_styled.pdf. Content: course report and evidence tables.`);
}
const candidate=path.join(stage,'candidate.pptx');
await(await PresentationFile.exportPptx(p)).save(candidate);
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const final=path.join(out,`presentation-${Date.now()}.pptx`);
await finalizePresentation({workspaceDir:stage,candidatePath:candidate,finalPath:final,pythonExecutable:process.env.COURSE_RUNTIME_PYTHON,materializeLiteralChartWorkbooks:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit','--validate-bullet-geometry',...owners.flatMap(n=>['--require-native-table-slide',String(n)])],
 requiredNativeTableOwnerSlides:owners,requiredNativeChartOwnerSlides:chartOwners,fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,receiptPath:path.join(stage,path.basename(final)+'.validation.json')});

for(let i=0;i<p.slides.items.length;i++){
 const slide=p.slides.items[i];
 const img=await p.export({slide,format:'png',scale:1});
 await fs.writeFile(path.join(stage,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await img.arrayBuffer()));
 console.log(`Rendered ${i+1}/${content.length}`);
}
await fs.copyFile(final,path.join(root,'production/Interception_Movements_Course_Presentation.pptx'));
console.log('Editable presentation exported and finalized.');
