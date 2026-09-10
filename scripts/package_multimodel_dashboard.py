"""Package the multi-model dashboard separately from the frozen September 5 ZIP."""
from pathlib import Path
import sys,json,zipfile,uuid
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.package_final_release import build_bundle,copy_to_bundle


def main():
    package='Interception_Movements_Multimodel_Dashboard'
    staging=ROOT/'production/assets'/f'dashboard_bundle_{uuid.uuid4().hex}'/package
    build_bundle(staging)
    multi=ROOT/'studies/review_corrected_evaluation/results/dashboard/multimodel'
    manifest=json.loads((multi/'manifest.json').read_text())
    for source in multi.rglob('*'):
        if source.is_file():copy_to_bundle(source,source.relative_to(ROOT),staging)
    for reference in manifest['inventory']:
        if reference['family']=='spline_pca':continue
        source=ROOT/'studies/final_strategy_evaluation/runs'/reference['family']/'fold0'/reference['reference']/'checkpoint.pt'
        copy_to_bundle(source,source.relative_to(ROOT),staging)
    report=ROOT/'production/Interception_Movements_Methods_Reconstruction_Review.pdf'
    copy_to_bundle(report,report.relative_to(ROOT),staging)
    for name in ['README.md','DETAILED_REVIEW_VERIFICATION.json','remaining_results_provenance.json','all_models_direct_context_summary.csv']:
        source=ROOT/'production'/name
        copy_to_bundle(source,source.relative_to(ROOT),staging)
    (staging/'README.txt').write_text('''Interception movement explorer - multi-model working release

From this extracted folder:
  python -m pip install -r requirements_dashboard.txt
  python -m streamlit run src/confirmatory_dashboard.py

No raw recordings or retraining are required to use the dashboard.
CVAE and spline + PCA: n=2,3,4,8. VAE and CAE: n=3,8.
Default: VAE n=8, lowest cohort-average generation distances in the tested matrix.
This is a numerical, endpoint-specific ranking, not an overall significance claim.
VAE ignores task conditions. Spline conditions affect timing only.
Live models use fold 0, neural seed 42; personal fingerprints use context trials.
Validation plots use all four folds, seed 42, and 120 generated samples per person.
Score cards summarize all available study seeds.

The current manuscript under production is a working draft with red questions for
Jason. The PDFs under output/pdf are the frozen September 5 reports, not current.
''',encoding='utf-8')
    output=ROOT/'output/release'/f'{package}.zip';output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(staging.rglob('*')):
            if p.is_file():z.write(p,Path(package)/p.relative_to(staging))
    with zipfile.ZipFile(output) as z:
        assert z.testzip() is None
        assert sum(n.endswith('checkpoint.pt') for n in z.namelist())==8
        assert sum('/multimodel/spline/' in n and n.endswith('.json') for n in z.namelist())==4
    print(json.dumps(dict(archive=str(output),staging=str(staging),bytes=output.stat().st_size),indent=2))


if __name__=='__main__':main()
