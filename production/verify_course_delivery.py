"""Check headline claims, final tables, and full-path follow-up independently.

Run after both artifact builders. No training and no saved-score replacement.
"""
import sys,json,hashlib,zipfile
from pathlib import Path
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from pypdf import PdfReader
from production.course_evidence import *

def main():
    check_frozen();checks=[]
    def ok(name,condition):
        assert condition,name
        checks.append(name)
    # Actual claims in the main captions, using the complete declared families.
    for n in (3,8):
        for f in FAMILIES[1:]:
            r=comparison('spline_pca',f,n,'trajectory_mse')
            ok(f'spline reconstruction BH {f} n{n}',r.q_bh_140<.0031)
            ok(f'spline reconstruction direction {f} n{n}',value('spline_pca',n,'trajectory_mse')<value(f,n,'trajectory_mse'))
            ok(f'Holm reconstruction distinction {f} n{n}',(r.p_holm_140<.05)==(not(n==8 and f=='conditional_ae')))
        for f in ('cvae','unconditional_vae'):
            for m in ('mean_ks','energy_distance','mmd_rbf'):
                r=comparison('spline_pca',f,n,m)
                ok(f'feature improvement {f} n{n} {m}',r.q_bh_140<.05 and r.p_holm_140<.05 and value(f,n,m)<value('spline_pca',n,m))
    for g in ('raw_rms','axis_balanced'):
        for f in ('cvae','unconditional_vae'):
            for m in ('energy','mmd2'):
                r=comparison('spline_pca',f,8,m,g)
                ok(f'path improvement {f} n8 {g} {m}',r.q_bh_80<.05 and r.p_holm_80<.05)
        for m in ('energy','mmd2'):
            r=comparison('cvae','unconditional_vae',8,m,g)
            ok(f'no unique variational winner {g} {m}',r.q_bh_80>.05)
    for f in ('cvae','unconditional_vae'):
        ok(f'n3 raw energy nonsignificant {f}',comparison('spline_pca',f,3,'energy','raw_rms').q_bh_80>.05)
        ok(f'n3 raw MMD significant {f}',comparison('spline_pca',f,3,'mmd2','raw_rms').p_holm_80<.05)
        ok(f'n3 balanced MMD nonsignificant {f}',comparison('spline_pca',f,3,'mmd2','axis_balanced').q_bh_80>.05)
    # Independently recompute all 80 signed-rank tests and both adjustments.
    people=pd.read_csv(OUT/'trajectory_distribution_2026_09_13/participant_scores.csv')
    for r in PATH_TESTS.itertuples():
        part=people[(people.geometry==r.geometry)&((people.latent_dim==r.latent_dim)|(people.model_family=='condition_ridge'))]
        w=part.pivot(index='subject',columns='model_family',values=r.metric)
        d=np.round((w[r.model_b]-w[r.model_a]).to_numpy(),12)
        actual=wilcoxon(d,zero_method='pratt',alternative='two-sided',method='auto').pvalue if np.any(d) else 1
        ok(f'paired p {r.Index}',abs(actual-r.p)<1e-12)
    p=PATH_TESTS.p.to_numpy();order=np.argsort(p);n=len(p)
    bh=np.minimum(1,np.minimum.accumulate((p[order]*n/np.arange(1,n+1))[::-1])[::-1])
    holm=np.minimum(1,np.maximum.accumulate(p[order]*np.arange(n,0,-1)))
    ok('BH80 independently recomputed',np.allclose(bh,PATH_TESTS.q_bh_80.to_numpy()[order],atol=1e-12))
    ok('Holm80 independently recomputed',np.allclose(holm,PATH_TESTS.p_holm_80.to_numpy()[order],atol=1e-12))
    # Recheck one fixed participant in each model/capacity/geometry directly from decoded arrays.
    refs={(r['fold'],r['geometry']):r for r in json.loads((OUT/'trajectory_distribution_2026_09_13/distance_references.json').read_text())}
    rows=pd.read_csv(OUT/'trajectory_distribution_2026_09_13/per_seed_scores.csv')
    maximum=0.;array_cases=0
    for dim in (3,8):
        for f in FAMILIES:
            seed=-1 if f=='spline_pca' else 42
            data=np.load(OUT/'assets/trajectory_distribution_2026_09_13'/f'{f}_z{dim}_{"" if seed==-1 else "seed42_"}fold2_subject01.npz')
            for g in ('raw_rms','axis_balanced'):
                ref=refs[2,g]
                x=(data['recorded']/ref['axis_scale']).reshape(-1,200)/np.sqrt(200)
                y=(data['generated']/ref['axis_scale']).reshape(-1,200)/np.sqrt(200)
                xx=np.linalg.norm(x[:,None]-x[None,:],axis=-1);yy=np.linalg.norm(y[:,None]-y[None,:],axis=-1);xy=np.linalg.norm(x[:,None]-y[None,:],axis=-1)
                energy=2*xy.mean()-xx.mean()-yy.mean()
                kxx,kyy,kxy=[np.exp(-ref['rbf_gamma']*d*d) for d in (xx,yy,xy)]
                np.fill_diagonal(kxx,0);np.fill_diagonal(kyy,0)
                mmd=kxx.sum()/(len(x)*(len(x)-1))+kyy.sum()/(len(y)*(len(y)-1))-2*kxy.mean()
                r=rows[(rows.subject=='subject01')&(rows.model_family==f)&(rows.latent_dim==dim)&(rows.seed==seed)&(rows.geometry==g)].iloc[0]
                diff=max(abs(energy-r.energy),abs(mmd-r.mmd2));maximum=max(maximum,diff);array_cases+=1
                ok(f'independent decoded metric {f} {dim} {g}',diff<1e-12)
    protected=json.loads((OUT/'trajectory_distribution_2026_09_13/verification.json').read_text())['protected_file_hashes']
    weights=0
    for name,h in protected.items():
        if name.endswith('checkpoint.pt'):
            ok(f'unchanged {name}',hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h);weights+=1
    # Exact output table cells must survive into native PPTX tables.
    slides=json.loads((OUT/'course_slides.json').read_text(encoding='utf-8'))
    ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
    with zipfile.ZipFile(OUT/'Interception_Movements_Course_Presentation.pptx') as z:
        for i,s in enumerate(slides,1):
            xml=ET.fromstring(z.read(f'ppt/slides/slide{i}.xml'))
            if s['table']:
                cells=[''.join(t.itertext()) for t in xml.findall('.//a:tc/a:txBody/a:p/a:r/a:t',ns)]
                for row in s['table']:
                    for cell in row:ok(f'slide {i} cell {cell}',str(cell) in cells)
            notes=ET.fromstring(z.read(f'ppt/notesSlides/notesSlide{i}.xml'))
            ok(f'speaker notes slide {i}',len(''.join(notes.itertext()))>100)
    outputs={}
    for file,count in [('8_pages_draft.pdf',8),('Course_Report_Appendix.pdf',10),('10_min_presentation.pdf',25)]:
        doc=PdfReader(OUT/file);ok(f'{file} page count',len(doc.pages)==count)
        outputs[file]={'pages':count,'sha256':hashlib.sha256((OUT/file).read_bytes()).hexdigest()}
    for f in ('8_pages_draft.pdf','Course_Report_Appendix.pdf'):
        txt='\n'.join(p.extract_text() for p in PdfReader(OUT/f).pages)
        ok(f'{f} no edit placeholders','[EDIT]' not in txt and '\u25a0' not in txt)
    outputs['Interception_Movements_Course_Presentation.pptx']={'slides':len(slides),'sha256':hashlib.sha256((OUT/'Interception_Movements_Course_Presentation.pptx').read_bytes()).hexdigest()}
    result={'checks_passed':len(checks),'outputs':outputs,'unchanged_checkpoints':weights,'frozen_lab_report_sha256':FROZEN_SHA,'new_trajectory_tests_recomputed':80,'decoded_array_cases_recomputed':array_cases,'maximum_distance_difference':maximum,'main_talk_seconds':sum(s['seconds'] for s in slides),'main_slides':10,'backup_slides':len(slides)-10,'checks':checks}
    (OUT/'COURSE_DELIVERY_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))

if __name__=='__main__':main()
