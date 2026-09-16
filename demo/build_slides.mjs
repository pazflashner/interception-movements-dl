// New slides only. assemble_presentation.ps1 combines them with Paz's source deck.
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import {createHash} from 'node:crypto';
import {fileURLToPath, pathToFileURL} from 'node:url';

const dir = path.dirname(fileURLToPath(import.meta.url));
const root = path.dirname(dir);
const runtime = process.env.COURSE_RUNTIME_ROOT || path.join(os.homedir(), '.cache/codex-runtimes/codex-primary-runtime/dependencies');
const skill = process.env.COURSE_SLIDE_SKILL || path.join(os.homedir(), '.codex/plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations');
process.env.RUNTIME_NODE_MODULES = path.join(runtime, 'node/node_modules');
const {importRuntimeModule} = await import(pathToFileURL(path.join(skill, 'container_tools/runtime_helpers.mjs')).href);
const {Presentation, PresentationFile} = await importRuntimeModule('@oai/artifact-tool');
const {applyPresentationChartFont, finalizePresentation} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const stage = path.join(dir, '.build');
await fs.mkdir(path.join(stage,'final'), {recursive:true});
const source = JSON.parse(await fs.readFile(path.join(root,'production/course_slides.json'),'utf8'));
const paths = source.find(s=>s.title === 'Full-trajectory generation supports VAE and CVAE');
const personal = source.find(s=>s.title === 'The participant’s own center improves generation');
if(!paths || !personal) throw new Error('Audited source slides not found');
const width = 12191695 / 9525, height = 720;
const p = Presentation.create({slideSize:{width,height}});
const font = 'Calibri';
const navy = '#161E2E', blue = '#1B6FEA';
function text(s, value, x,y,w,h,size=23,bold=false,color=navy) {
  const t=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  t.text=value;t.text.style={typeface:font,fontSize:size,bold,color,autoFit:'none'};return t;
}
function rect(s,x,y,w,h,fill) {
  s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:'none',width:0}});
}
function slide(title,number) {
  const s=p.slides.add();s.background.fill='#FFFFFF';
  ['#1B6FEA','#17A2A2','#7A5CF0'].forEach((c,i)=>rect(s,i*width/3,0,width/3,7,c));
  text(s,title,82,32,1030,57,32,true);
  text(s,`${number}`,1120,45,78,22,16,false,blue);
  rect(s,82,100,1116,1,'#DFE5EE');
  return s;
}
function callout(s,head,body) {
  rect(s,82,573,1116,116,'#EFF5FC');rect(s,82,573,5,116,blue);
  text(s,head,105,586,1060,29,24,true,blue);
  text(s,body,105,625,1060,52,21,false,'#3D4756');
}
function chart(s,d,x,y,w,h,max) {
  text(s,d.title,x,y,w,34,25,true);
  const c=s.charts.add('bar',{
    position:{left:x,top:y+43,width:w,height:h-43},categories:d.categories,
    series:d.series.map(({marks,tests,...series})=>({...series,
      dataLabelOverrides:series.values.map((value,idx)=>({idx,text:value.toFixed(3),position:'outEnd',
        showValue:false,showCategoryName:false,showSeriesName:false,
        textStyle:{typeface:font,fontSize:w>600?19:16,fill:navy}}))})),
    barOptions:{direction:'column',grouping:'clustered',gapWidth:80},hasLegend:true,
    legend:{position:'bottom',textStyle:{typeface:font,fontSize:19,fill:navy}},
    xAxis:{textStyle:{typeface:font,fontSize:19,fill:navy},line:{fill:'#D6DEE8',width:1}},
    yAxis:{min:0,max,numberFormatCode:'0.00',textStyle:{typeface:font,fontSize:18,fill:navy},
      majorGridlines:{fill:'#DFE5EE',width:1}},
    dataLabels:{showValue:false},chartFill:'#FFFFFF',plotAreaFill:'#FFFFFF'});
  applyPresentationChartFont(c,{fontFamily:font});
}

const s9=slide('Full-trajectory generation',9);
text(s9,'Do generated paths match the held-out set of complete trajectories?',82,121,1116,35,26);
text(s9,'Energy and MMD² compare between-set differences with within-set variability. Lower is better.',82,164,1116,30,21,false,'#4C5969');
chart(s9,paths.charts[0],82,211,530,310,.16);
chart(s9,paths.charts[1],668,211,530,310,.10);
callout(s9,'At n=8, VAE and CVAE outperform spline + PCA on both distances',
  'Paired tests: BH- and Holm-adjusted p < .05. VAE vs CVAE is not significant; n=3 evidence is mixed.');
s9.speakerNotes.textFrame.setText(paths.notes.split(' Bars show participant-balanced means')[0] + '\nEach generated path and each recorded query path is a 200-coordinate vector. This is a set-to-set comparison, not pairing trial 1 with trial 1. Each participant contributes one score per model after averaging training seeds. Bars give equal weight to each of the 28 participants; labels are means rounded to three decimals. Three-dimensional CAE energy is significantly worse under BH only. Exact values and adjusted p-values are in source course_slides.json and the trajectory_distribution_2026_09_13 CSVs. The selected plot uses raw RMS coordinates, so energy has units cm and MMD² is dimensionless. The full-path adjustment family contains 80 comparisons. At n=8, the VAE/CVAE improvement over spline also survives balancing the axes. A smaller distance does not by itself prove absolute physiological realism.');

const s10=slide('Personal context improves generation',10);
text(s10,'Replace only the fingerprint center; keep the decoder, noise and conditions fixed.',82,121,1116,60,26);
text(s10,'27 / 28',82,255,360,92,82,true,'#1D7A3E');
text(s10,'participants had lower mean KS',82,361,350,66,27,true);
text(s10,'Own center vs population center\nVAE, n=8',82,438,350,60,23,false,'#4C5969');
const controlChart=structuredClone(personal.charts[0]);
controlChart.title='Mean feature KS (lower is better)';
controlChart.categories=controlChart.categories.slice(1);
controlChart.series=controlChart.series.map(s=>({...s,values:s.values.slice(1),marks:s.marks.slice(1),tests:s.tests.slice(1)}));
chart(s10,controlChart,466,211,732,310,.38);
callout(s10,'The correct personal center improves the generated feature distributions',
  'Both n=8 models beat population and other-person centers (BH-adjusted p < .05): useful personal information within the session.');
s10.speakerNotes.textFrame.setText(personal.notes.split(' Bars show participant-balanced means')[0] + '\nThe 27/28 count is specifically VAE n=8 mean feature KS, own center versus population center. It is not an identification accuracy and not a claim of individual significance for 27 people. Mean KS: CVAE8 own .219404, population .289636, wrong .317203; VAE8 own .215121, population .284126, wrong .314405. CVAE mean-KS control BH q: population 3.397464752e-7, wrong 2.086162567e-7. VAE both 2.980232239e-8. Each wrong-person result averages alternative test-person centers. CVAE adjustment family has 12 contrasts; VAE n=8 has 6. Sources: studies/review_corrected_evaluation/results/review_controls/fingerprint_paired.csv and production/audit_2026_09_08/uvae8_fingerprint_paired.csv.');

const s11=slide('An interactive dashboard',11);
text(s11,'Explore a latent representation and see its generated trajectory and predicted timing.',82,121,1116,60,26);
s11.images.add({blob:new Uint8Array(await fs.readFile(path.join(dir,'dashboard_preview.png'))),contentType:'image/png',
  alt:'Actual demo dashboard: latent sliders at left, generated trajectory and timing at right',fit:'contain',
  position:{left:187,top:188,width:906,height:510}});
s11.speakerNotes.textFrame.setText('Live demo, under construction. Double-click demo/START_DEMO.cmd to open Chrome. The original research dashboard is unchanged. Start with VAE n=8, move a latent offset, then optionally switch to n=3 or another model. Each move decodes one latent vector and separately predicts movement and initiation times. It does not sample a distribution or predict one particular future trial. The selected center is the training-population center or a saved context fingerprint. Offsets are measured in training-latent standard deviations; latent axes do not have established psychological meanings. VAE ignores task-condition controls; spline + PCA uses conditions for timing only. All live references are fold 0, seed 42 for neural models. Source: demo/app.py; src/confirmatory_dashboard.py decode; src/dashboard_models.py. Screenshot: the actual locally running demo.');

const s12=p.slides.add();s12.background.fill='#0E1729';
['#1B6FEA','#17A2A2','#7A5CF0'].forEach((c,i)=>rect(s12,i*width/3,0,width/3,7,c));
for(const [value,y,h,size,color] of [
  ['Thank you',225,95,68,'#FFFFFF'],
  ['Questions?',335,72,44,'#3EB1EC'],
  ['Seman Libbiss & Paz Flashner',490,52,29,'#A3B2C6']]) {
  const t=text(s12,value,82,y,1116,h,size,true,color);
  t.text.style={typeface:font,fontSize:size,bold:true,color,alignment:'center',autoFit:'none'};
}
s12.speakerNotes.textFrame.setText('Thank the audience and invite questions. Seman Libbiss and Paz Flashner.');

const candidate=path.join(stage,'new_slides_candidate.pptx');
await (await PresentationFile.exportPptx(p)).save(candidate);
const final=path.join(stage,'final',`new_slides_${Date.now()}.pptx`);
await finalizePresentation({workspaceDir:dir,candidatePath:candidate,finalPath:final,
  pythonExecutable:path.join(runtime,'python/python.exe'),
  integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12191695,6858000','--validate-heading-fit'],
  explicitTotalSlideCount:4,requiredNativeChartOwnerSlides:[1,2],materializeLiteralChartWorkbooks:true,
  fontPolicy:{basis:'reference',families:[font],referencePath:path.join(root,'.tmp/demo_build/Paz_latest.pptx'),
    referenceSha256:createHash('sha256').update(await fs.readFile(path.join(root,'.tmp/demo_build/Paz_latest.pptx'))).digest('hex')},
  verifyArtifactToolImport:true,receiptPath:path.join(stage,path.basename(final)+'.validation.json')});
await fs.copyFile(final,path.join(stage,'new_slides.pptx'));
console.log('New slides finalized: '+final);
