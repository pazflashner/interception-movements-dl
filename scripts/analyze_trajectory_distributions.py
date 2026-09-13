"""Frozen-model follow-up: complete go-window trajectory distributions.

Run from the repository root. Requires the canonical cache and local checkpoints.
Writes separate evidence; never edits trained weights, existing scores or PDFs.
"""
from pathlib import Path
import hashlib
import itertools
import json
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import scipy
from scipy.stats import false_discovery_control
import torch
from threadpoolctl import threadpool_limits

import config
from scripts.run_review_controls import load_trials
from scripts.run_corrected_study import context_query_for_trials, load_per_trial_checkpoint, training_latent_noise_covariance
from src.evaluate import encode_trials
from src.baseline_spline import SplinePCARepresentation
from src.confirmatory_spline import TimingRidge, _training_noise_covariance
from src.confirmatory_controls import ConditionRidge
from src.features import compute_trial_features, features_from_generated_window, kinematic_features_for_dim
from src.context_query import DistanceReference, distribution_distances
from src.vae_model import encode_trial_condition
from src.statistical_tests import paired_wilcoxon, holm_adjust
from src.trajectory_distribution import TrajectoryReference, trajectory_distances

OUT = ROOT / 'production/trajectory_distribution_2026_09_13'
CACHE = ROOT / 'production/assets/trajectory_distribution_2026_09_13'
SOURCE = ROOT / 'studies/final_strategy_evaluation/runs'
CORRECTED = ROOT / 'studies/review_corrected_evaluation/runs'
FAMILIES = ['spline_pca', 'cvae', 'conditional_ae', 'unconditional_vae']
GEOMETRIES = ['raw_rms', 'axis_balanced']
METRICS = ['energy', 'mmd2', 'mmd2_biased_diagnostic', 'mean_path_rmse',
           'all_pairs_mse_diagnostic', 'recorded_dispersion', 'generated_dispersion', 'dispersion_ratio']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_protocol():
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    protocol = {
        'version': 'trajectory-distribution-followup-v1',
        'status': 'exploratory follow-up defined before calculating these endpoints; not preregistered research',
        'target': 'complete 100x2 phase-normalized target-motion-onset to last-tracker-sample paths; original translations retained; no recropping, filtering or dynamic time warping',
        'primary_geometry': 'Euclidean distance after flattening ordered paths and dividing by sqrt(200), equivalent to root mean coordinate squared difference; common tracker units retained',
        'sensitivity_geometry': 'divide each axis by its training RMS deviation from the training mean path, shared across 100 points; then divide flattened vector by sqrt(200)',
        'energy': '2 mean cross Euclidean distance minus within-recorded and within-generated means, including diagonal zeros; not square-rooted',
        'mmd2': 'unbiased RBF MMD squared, within-sample diagonals excluded; gamma=1/median positive training squared distance among up to 512 deterministic training trials',
        'diagnostics': 'biased MMD squared, mean-path RMSE, all-pairs MSE and generated/recorded total sample-variance ratio; none is a substitute for the primary distribution endpoints',
        'dimensions': [3, 8], 'families': FAMILIES + ['condition_ridge'],
        'folds': [0,1,2,3], 'neural_seeds': [42,43,44], 'condition_ridge_generation_seeds': [42,43,44],
        'spline_sampling': 'one historical deterministic fit and frozen sampling realization per fold/dimension',
        'samples_per_participant': 120, 'context_query_seed': config.CONTEXT_QUERY_SEED,
        'sampling': 'exact historical context centroid, covariance, query-condition mixture and RNG; each model generation must reproduce its existing feature scores',
        'references': 'resample 120 own context trajectories with replacement, using three stable SHA-derived seeds; repeat the context mean path 120 times as a collapse diagnostic. Neither is a compact model or population null reference.',
        'aggregation': 'average seeds within participant then give all 28 participants equal weight; record fold and per-seed results',
        'inference': '80 exploratory paired two-sided Pratt Wilcoxon comparisons, 10 pairs among 5 model families x 2 dimensions x 2 distribution endpoints x 2 geometries; BH and Holm over all 80 together. Condition Ridge is reused at both dimensions. No tests on diagnostic references.',
        'limits': 'no physical-time fidelity claim; same-session context/query; overlapping training folds; inherited char-sum RNG collisions; no independent validation of model selection; finite-sample distances are not accuracy percentages',
        'new_neural_training': False,
    }
    path = OUT / 'protocol.json'
    if path.exists():
        old = json.loads(path.read_text())
        assert old['protocol'] == protocol, 'Protocol changed; start a separate version'
    else:
        path.write_text(json.dumps({'written_utc': datetime.now(timezone.utc).isoformat(), 'protocol': protocol}, indent=2)+'\n')


def score_case(paths, query_paths, refs, metadata):
    rows = []
    for ref in refs:
        rows.append({**metadata, 'geometry': ref.geometry, **trajectory_distances(query_paths, paths, ref)})
    return rows


def main():
    torch.set_num_threads(1)
    threadpool_limits(1)
    freeze_protocol()
    files = list(SOURCE.glob('*/*/*/checkpoint.pt'))
    files += list((ROOT/'production').glob('*.pdf'))
    files += list((ROOT/'output/pdf').glob('*.pdf'))
    before = {str(p.relative_to(ROOT)): sha(p) for p in files}
    trials = load_trials()
    assert len(trials) == 4732
    features = kinematic_features_for_dim(2)
    provenance, reference_rows = [], []
    for fold in range(4):
        split_path = SOURCE / f'cvae/fold{fold}/cvae_z3_seed42/split.json'
        split = json.loads(split_path.read_text())
        sets = [set(split[k+'_subjects']) for k in ['train','validation','test']]
        assert [len(s) for s in sets] == [17,4,7]
        assert all(not a & b for a,b in itertools.combinations(sets, 2))
        train, validation, test = [[t for t in trials if t['metadata']['subject'] in ids] for ids in sets]
        train_paths = np.stack([t['pos_norm'] for t in train])
        refs = [TrajectoryReference.fit(train_paths, g) for g in GEOMETRIES]
        reference_rows.extend([{'fold':fold, 'n_train_trials':len(train), **r.to_dict()} for r in refs])
        feature_reference = DistanceReference.fit(pd.DataFrame([compute_trial_features(t) for t in train]), features)
        splits = context_query_for_trials(test, config.CONTEXT_QUERY_SEED)
        truth = {s.subject: pd.DataFrame([compute_trial_features(test[i]) for i in s.query_indices]) for s in splits}
        query_paths = {s.subject: np.stack([test[i]['pos_norm'] for i in s.query_indices]) for s in splits}
        for s in splits:
            assert not set(s.context_indices) & set(s.query_indices)
            provenance.append({'fold':fold, 'subject':s.subject,
                'context_ids':[test[i]['metadata']['trial_id'] for i in s.context_indices],
                'query_ids':[test[i]['metadata']['trial_id'] for i in s.query_indices]})
        baseline_path = CACHE / f'references_fold{fold}.csv'
        if not baseline_path.exists():
            rows=[]
            for s in splits:
                context = np.stack([test[i]['pos_norm'] for i in s.context_indices])
                for family in ['context_resample', 'context_mean_only']:
                    for seed in ([42,43,44] if family=='context_resample' else [-1]):
                        if family == 'context_resample':
                            tag=f'context-resample-{s.subject}-{seed}'.encode()
                            rng=np.random.default_rng(int.from_bytes(hashlib.sha256(tag).digest()[:8], 'little'))
                            generated=context[rng.integers(0,len(context),120)]
                        else:
                            generated=np.repeat(context.mean(0)[None],120,axis=0)
                        rows.extend(score_case(generated,query_paths[s.subject],refs,dict(model_family=family,latent_dim=0,fold=fold,seed=seed,subject=s.subject,n_context=len(context),n_query=len(s.query_indices),n_generated=120,feature_reproduction_error=0)))
            pd.DataFrame(rows).to_csv(baseline_path,index=False)
        specs=[(f,n,seed) for f in FAMILIES for n in [3,8] for seed in ([-1] if f=='spline_pca' else [42,43,44])]
        specs += [('condition_ridge',0,seed) for seed in [42,43,44]]
        ridge=None
        for family,dim,seed in specs:
            name = f'{family}_z{dim}' + ('' if seed==-1 else f'_seed{seed}')
            if family=='condition_ridge': name=f'condition_ridge_seed{seed}'
            cache_name = name + ('_blas2' if family == 'condition_ridge' else '')
            dest=CACHE/f'{cache_name}_fold{fold}.csv'
            if dest.exists():
                print('Resume',name,'fold',fold,flush=True)
                continue
            run=SOURCE/family/f'fold{fold}'/name
            if family=='spline_pca':
                result=json.loads((run/'result.json').read_text())
                rep=SplinePCARepresentation(dim,include_timing=False,standardize_coefficients=result['standardize_spline_coefficients']).fit(train)
                timing_model=TimingRidge.fit(rep,train,validation)
                train_mu=rep.encode(train);mu=rep.encode(test)
                covariance=_training_noise_covariance(train_mu,np.asarray([t['metadata']['subject'] for t in train]))
            elif family=='condition_ridge':
                if ridge is None:
                    # The earlier integrity audit identified BLAS sensitivity
                    # for the rank-deficient alpha=0 Ridge fit. Two threads
                    # reproduce its historical exported scores (audit row 60).
                    with threadpool_limits(limits=2):
                        ridge=ConditionRidge.fit(train,validation)
            else:
                model,norm=load_per_trial_checkpoint(run/'checkpoint.pt','cpu');model.eval()
                mu=encode_trials(model,test,norm,'cpu')[0]
                covariance=training_latent_noise_covariance(model,train,norm,'cpu')
            saved_path=CORRECTED/family/f'fold{fold}'/name/'context_query_fidelity.csv'
            saved=pd.read_csv(saved_path).set_index('subject')
            rows=[]
            for s in splits:
                query=[test[i] for i in s.query_indices]
                if family=='condition_ridge':
                    rng=np.random.default_rng(seed+sum(map(ord,s.subject)))
                    chosen=rng.integers(0,len(query),120)
                    with threadpool_limits(limits=2):
                        generated,timing=ridge.sample([query[i]['metadata'] for i in chosen],rng)
                else:
                    rng=np.random.default_rng(config.CONTEXT_QUERY_SEED+sum(map(ord,s.subject)))
                    z=rng.multivariate_normal(mu[s.context_indices].mean(0),covariance,size=120)
                    chosen=rng.integers(0,len(query),120)
                    conditions=np.stack([encode_trial_condition(query[i]['metadata'],5) for i in chosen])
                    if family=='spline_pca':
                        generated=rep.decode(z)[0];timing=timing_model.predict(z,conditions)
                    else:
                        with torch.no_grad():
                            xx,tt=model.decode(torch.as_tensor(z,dtype=torch.float32),torch.as_tensor(conditions,dtype=torch.float32))
                        generated=(xx.numpy()*norm.train_std+norm.train_mean).reshape(120,100,2)
                        timing=norm.denormalise_timing(tt.numpy())
                with threadpool_limits(limits=2 if family=='condition_ridge' else 1):
                    generated_features=pd.DataFrame([features_from_generated_window(p,max(float(t[0]),.001),max(float(t[1]),0),config.WINDOW_GO_TO_ARRIVAL) for p,t in zip(generated,timing)])
                    distances=distribution_distances(truth[s.subject],generated_features,features,feature_reference)
                keys=['energy_distance','mmd_rbf',*[f'ks_{f}' for f in features]]
                error=max(abs(distances[k]-saved.loc[s.subject,k]) for k in keys)
                assert error<1e-5,(name,fold,s.subject,error)
                metadata=dict(model_family=family,latent_dim=dim,fold=fold,seed=seed,subject=s.subject,n_context=len(s.context_indices),n_query=len(query),n_generated=120,feature_reproduction_error=error)
                rows.extend(score_case(generated,query_paths[s.subject],refs,metadata))
                # Auditable decoded paths stay in ignored cache, never raw public exports.
                np.savez_compressed(CACHE/f'{cache_name}_fold{fold}_{s.subject}.npz',generated=generated,recorded=query_paths[s.subject])
            pd.DataFrame(rows).to_csv(dest,index=False)
            print('Verified and scored',name,'fold',fold,'max feature difference',max(r['feature_reproduction_error'] for r in rows),flush=True)
    current_csv=[p for p in sorted(CACHE.glob('*.csv')) if not p.name.startswith('condition_ridge_') or '_blas2_' in p.name]
    all_rows=pd.concat([pd.read_csv(p) for p in current_csv],ignore_index=True)
    assert len(all_rows)==1512  # 644 model/seed/person evaluations + 112 references, twice for geometry
    all_rows.to_csv(OUT/'per_seed_scores.csv',index=False)
    people=all_rows.groupby(['model_family','latent_dim','fold','subject','geometry'],as_index=False)[METRICS].mean()
    assert (people.groupby(['model_family','latent_dim','geometry']).size()==28).all()
    people.to_csv(OUT/'participant_scores.csv',index=False)
    summary=people.groupby(['model_family','latent_dim','geometry'],as_index=False)[METRICS].mean()
    summary.to_csv(OUT/'summary.csv',index=False)
    paired=[]
    for geometry,dim,metric in itertools.product(GEOMETRIES,[3,8],['energy','mmd2']):
        part=people[(people.geometry==geometry)&((people.latent_dim==dim)|(people.model_family=='condition_ridge'))]
        wide=part.pivot(index='subject',columns='model_family',values=metric)
        for a,b in itertools.combinations(FAMILIES+['condition_ridge'],2):
            difference=wide[b]-wide[a]
            paired.append(dict(geometry=geometry,latent_dim=dim,metric=metric,model_a=a,model_b=b,n=28,mean_a=wide[a].mean(),mean_b=wide[b].mean(),mean_b_minus_a=difference.mean(),a_better_n=int((difference>0).sum()),p=paired_wilcoxon(difference).pvalue))
    paired=pd.DataFrame(paired);assert len(paired)==80
    paired['q_bh_80']=false_discovery_control(paired.p.to_numpy(),method='bh')
    paired['p_holm_80']=holm_adjust(paired.p.to_numpy())
    paired.to_csv(OUT/'paired_comparisons.csv',index=False)
    assert all(sha(ROOT/p)==h for p,h in before.items()), 'Frozen evidence changed'
    (OUT/'distance_references.json').write_text(json.dumps(reference_rows,indent=2)+'\n')
    (OUT/'context_query_manifest.json').write_text(json.dumps(provenance,indent=2)+'\n')
    sources=[ROOT/'src/trajectory_distribution.py',Path(__file__),ROOT/'config.py',ROOT/'src/confirmatory_spline.py',ROOT/'src/confirmatory_controls.py',ROOT/'src/baseline_spline.py',ROOT/'src/vae_model.py',ROOT/'scripts/run_corrected_study.py',ROOT/'studies/strategy_window_comparison/data/canonical_trials.pkl']
    verification=dict(completed_utc=datetime.now(timezone.utc).isoformat(),numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__,participant_model_seed_evaluations=int(len(all_rows[~all_rows.model_family.str.startswith('context_')])/2),references_participant_seed_evaluations=int(len(all_rows[all_rows.model_family.str.startswith('context_')])/2),paired_tests=len(paired),max_feature_reproduction_error=float(all_rows.feature_reproduction_error.max()),unchanged_neural_checkpoint_count=sum(p.endswith('checkpoint.pt') for p in before),protected_file_hashes=before,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in sources},new_neural_training=False,refitted_classical_models='8 deterministic spline/PCA/timing fits and 4 condition-Ridge fits reproducing original settings',numeric_threads='1 except Condition Ridge fit/sample/feature reproduction use 2, matching the previously documented Ridge BLAS sensitivity; superseded one-thread Ridge cache is excluded',prior_scores_and_pdfs_unchanged=True)
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(summary[['model_family','latent_dim','geometry','energy','mmd2','mean_path_rmse','dispersion_ratio']].to_string(index=False),flush=True)


if __name__=='__main__':
    main()
