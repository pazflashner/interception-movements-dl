"""Visual main talk and separate methods explanation document."""
import json
from xml.sax.saxutils import escape
from reportlab.lib.units import mm
from reports.article_layout import Article
from production.course_evidence import *

def revise(slides):
    explanations=slides[10:]
    for s in explanations:s['title']=s['title'].removeprefix('Backup: ')
    explanations.append(dict(title='What changed after checking feature correlations?',body='',table=None,image=None,equation=None,caption='Training-only feature selection; complete 72-test sensitivity family.',notes='We recorded a priority and a 0.90 correlation threshold before computing the results. A feature needed strong pooled and median within-participant Spearman correlation with a retained feature. All four training rotations dropped straight-line distance and endpoint y in favor of path length. With nine features, both VAE and CVAE still beat spline on all three metrics at both capacities under BH and Holm. VAE versus CVAE at eight dimensions no longer passed BH on any reduced metric: q=.149 for KS, .289 for energy and .428 for MMD. We therefore do not call VAE uniquely best. Removing features changes what we measure; it does not automatically reduce our correction family, because mean KS is one score, not eleven feature p-values. Original and reduced views were corrected together across six pairs, two capacities and three metrics: 72 tests. See report Appendix A11 and production/feature_redundancy_2026_09_14/.'))
    (OUT/'course_explanations.json').write_text(json.dumps(explanations,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    slides=slides[:10]
    def fig(name):return f'production/course_assets/{name}.png'
    def chart(metric,title,geometry=None):
        series=[]
        for n,color in [(3,'#8395A3'),(8,'#087C83')]:
            marks=[''];tests=[None]
            for f in FAMILIES[1:]:
                r=comparison('spline_pca',f,n,metric,geometry)
                q=float(r['q_bh_80' if geometry else 'q_bh_140'])
                holm=float(r['p_holm_80' if geometry else 'p_holm_140'])
                marks.append('*' if holm<.05 and q<.05 else '†' if q<.05 else 'ns')
                tests.append(dict(reference='spline_pca',model=f,n=n,metric=metric,geometry=geometry,q_bh=q,p_holm=holm))
            series.append(dict(name=f'n = {n}',values=[round(value(f,n,metric,geometry),6) for f in FAMILIES],marks=marks,tests=tests,fill=color))
        return dict(title=title,annotation_key='vs spline: * BH + Holm; † BH only; ns neither',categories=['Spline','CAE','CVAE','VAE'],series=series)

    slides[1].update(body='4,732 trials · 28 participants\nFour rotations: 17 train / 4 validation / 7 test',image=fig('task_and_recordings'),layout='wide_image',caption='Free eye movement. All 180 trajectories from the first participant by identifier. Median retained trials: 179.5.')
    slides[1]['notes']+=' Data were collected by Beatris Giterman in Jason Friedman’s laboratory. Spatial coordinates are cm; nominal starting positions are mm from screen center.'
    slides[2]['notes']+= ' Five interior knots plus cubic degree three plus one give nine basis coefficients per axis. Timing uses log(t + 0.001) to accommodate zero and compress its scale, then training standardization.'
    slides[2]['notes']=slides[2]['notes'].replace('Backup slides explain','The separate methods explanation PDF explains')
    slides[2]['caption']='VAE: variational autoencoder. CVAE adds task conditions; CAE uses a deterministic code. Matched n = 3 and 8.'
    slides[3].update(table=None,image=fig('reconstruction_slide'),layout='image_chart',charts=[chart('trajectory_mse','Reconstruction MSE (cm²)')],caption='Black: recorded; dashed: reconstructed. One median-duration trial. Lower MSE is better. Spline beats each neural model under BH-140; spline vs CAE n=8 does not survive Holm.')
    slides[4].update(body='Context codes establish a personal center. We decode 120 draws around it using shared training covariance.',equation=None,image=fig('generation_slide_n8'),layout='wide_image',caption='Same participant, separate query recordings. n=8, all models. Both capacities appear in the report. Thin: first 30 paths. Thick: mean of all paths. Equal x/y scale in cm.')
    slides[5].update(body='Cumulative curves show individual features. Mean KS summarizes all eleven feature discrepancies.',table=None,image=fig('features_selected_n8'),layout='image_chart',charts=[chart('mean_ks','Mean feature KS')],caption='Illustration: subject01, n=8. VAE/CVAE beat spline on all three feature metrics at both capacities (BH and Holm-140). VAE and CVAE give comparable results.')
    slides[6].update(body='Energy and MMD compare complete sets of ordered paths, including their variability.',table=None,layout='two_charts',charts=[chart('energy','Full-path energy (cm)','raw_rms'),chart('mmd2','Full-path MMD squared','raw_rms')],caption='Lower is better. At n=8, both VAE/CVAE beat spline (BH and Holm-80), also with balanced axes. VAE vs CVAE: q=.672 and 1.000. At n=3, results depend on metric and scaling.')
    ctr=control_means()
    slides[7].update(table=None,layout='two_charts',charts=[dict(title='Mean feature KS after replacing the center',categories=['CVAE n=3','CVAE n=8','VAE n=8'],series=[dict(name=label,values=[round(r[j],6) for r in ctr],fill=color) for j,(label,color) in enumerate([('Own','#087C83'),('Population','#8395A3'),('Other people','#B38349')])])])
    ctrl_chart=slides[7]['charts'][0]
    ctrl_chart['annotation_key']='vs own center: * BH-adjusted p < .05'
    for j,arm in enumerate(['own','population','wrong']):
        marks=[];tests=[]
        for f,n in [('cvae',3),('cvae',8),('unconditional_vae',8)]:
            if arm=='own':marks.append('');tests.append(None);continue
            source=pd.read_csv(ROOT/'studies/review_corrected_evaluation/results/review_controls/fingerprint_paired.csv') if f=='cvae' else pd.read_csv(OUT/'audit_2026_09_08/uvae8_fingerprint_paired.csv')
            if f=='cvae':source=source[source.latent_dim==n]
            r=source[(source.control==arm)&(source.metric=='mean_ks')].iloc[0]
            q=float(r['p_fdr_bh' if f=='cvae' else 'q_bh_6'])
            marks.append('*' if q<.05 else 'ns');tests.append(dict(reference='own',control=arm,model=f,n=n,metric='mean_ks',q_bh=q))
        ctrl_chart['series'][j].update(marks=marks,tests=tests)
    slides[8].update(body='Stored context resampling has lower mean path discrepancy.\nIt retains complete paths, while the compact models retain a personal center.',caption='Descriptive reference means. Resampling retains all context paths; repeating their mean loses movement variability.')
    slides[8]['table'][0][1]='Energy (cm)'
    slides[8]['notes']+=' The new nine-feature sensitivity is covered in Appendix A11. Both variational models retain their corrected advantage over spline, but VAE versus CVAE does not pass BH on any reduced-feature metric.'
    slides[9].update(image=fig('generation_slide_n8'),layout='wide_image',body='VAE n=8 is the default. Explore n=3, CVAE, CAE and spline + PCA alongside the cohort results.',caption='Explore recorded and generated paths, feature distributions and model benchmarks for the same session.')
    slides[5]['notes']+=' The restored figure uses empirical cumulative distributions, without histogram bins. KS is the largest vertical gap between two such curves.'
    # Borrow Paz's conceptual opening and feature overview, while preserving the audited course results.
    intro=dict(title='One project, two modeling tasks',body='Can a compact personal representation capture movement variability?',layout='question',
        columns=[dict(title='Reconstruction',text='Encode an observed trial and decode it back.\n\nDoes compression preserve its trajectory?'),dict(title='Generation',text='Use separate context trials to sample new movements.\n\nDo their distributions match held-out recordings?')],
        caption='Main evaluation: generation. Reconstruction tests compression; personal-center controls test the value of personal context.',
        notes='Our project models movement variability using compact personal representations and exposes the models in a dashboard. Reconstruction sees the particular trial; generation never sees the query path. Personal-center replacement asks whether the correct participant context improves the generator. This is evidence of within-session personalization, not a complete stable fingerprint. Sources: course report sections 1, 3 and 6.',seconds=40,backup=False,table=None,image=None,equation=None)
    slides[1].update(layout='data_features',features=[('Timing','Response time; movement time; relative time to peak speed'),('Speed','Peak speed; speed-peak count'),('Geometry','Path length; straight-line distance; curvature; maximum lateral deviation'),('Endpoint','Final x and y')],caption='Eleven features summarize timing, speed and geometry. Coordinates are in cm. Every participant enters a test fold once.')
    slides[6]['body']='Compare generated and held-out path sets: between-set distances plus within-set variability.'
    slides[7]['caption']='Own context improves all tested feature endpoints (BH: VAE < 1.6e-6; CVAE < .000382). This supports useful personal context within the session.'
    slides=slides[:8]+[slides[9]]  # Remove the standalone empirical-reference slide; retain it in the paper.
    slides.insert(1,intro)
    times=[20,40,50,65,45,40,50,80,70,80]
    for i,s in enumerate(slides):
        s['seconds']=times[i]
        s['presenter']='Paz' if i<=6 else 'Seman'
        if s.get('charts'):
            s['notes']+=' Bars show participant-balanced means after seed averaging. Markers compare each bar with the reference named above the chart, using the saved adjusted paired tests. In model-comparison charts, * means BH and Holm both pass .05; dagger means BH only; ns means neither. The personal-center chart uses its original BH-corrected control family. There are no error bars, and no p-value has been converted into an uncertainty interval.'
    for s in slides:
        s['backup']=False
        s['body']=s['body'].replace(' · ',' from ')
    return slides

def build_explanations():
    def clean(text):return escape(text.replace('\u2212','-').replace('on-slide equation','equation above'))
    items=json.loads((OUT/'course_explanations.json').read_text(encoding='utf-8'))
    a=Article(OUT/'Course_Methods_Explanations.pdf','Methods and Statistics: Presentation Companion','Seman Libbiss and Paz Flashner<br/>Questions, pipeline details and speaking explanations')
    a.p('This companion contains the detailed material removed from the presentation. The presentation itself contains only the ten main slides. Use these sections to prepare answers; the main report and its appendix provide the formal account and numerical tables.')
    for i,s in enumerate(items):
        if i:a.page()
        a.h(f'{i+1}. '+s['title'][0].upper()+s['title'][1:])
        if s.get('body'):a.p(clean(s['body']).replace('\n','<br/>'))
        if s.get('equation'):
            from production.report_equations import equation
            equation(a,ASSETS,Path(s['equation']).stem.removeprefix('equation_'))
        if s.get('table'):a.table(s['table'],s['caption'])
        elif s.get('caption'):a.p(clean(s['caption']),'caption')
        a.h('How to explain it',True)
        a.p(clean(s['notes']))
    a.finish()
