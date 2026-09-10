"""Investigate the one classical generative numerical-reproduction mismatch."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_generation import *
from src.confirmatory_controls import ConditionRidge
from threadpoolctl import threadpool_info

def main():
    trials=load_trials();fold=1;seed=44;subject='subject08'
    split=json.loads((CORRECTED/f'runs/cvae/fold{fold}/cvae_z3_seed42/split.json').read_text())
    train,val,test=[[t for t in trials if t['metadata']['subject'] in split[k]] for k in ['train_subjects','validation_subjects','test_subjects']]
    s=next(s for s in context_query_for_trials(test,config.CONTEXT_QUERY_SEED) if s.subject==subject);query=[test[i] for i in s.query_indices]
    features=kinematic_features_for_dim(2);emp=pd.DataFrame([compute_trial_features(t) for t in query]);ref=DistanceReference.fit(pd.DataFrame([compute_trial_features(t) for t in train]),features)
    saved=pd.read_csv(CORRECTED/'runs/condition_ridge/fold1/condition_ridge_seed44/context_query_fidelity.csv').set_index('subject').loc[subject]
    rows=[]
    for threads in [1,2,4,8,16]:
        with threadpool_limits(limits=threads):
            model=ConditionRidge.fit(train,val);rng=np.random.default_rng(seed+sum(map(ord,subject)));idx=rng.integers(0,len(query),120)
            x,tt=model.sample([query[i]['metadata'] for i in idx],rng)
            gen=pd.DataFrame([features_from_generated_window(p,max(float(t[0]),.001),max(float(t[1]),0),config.WINDOW_GO_TO_ARRIVAL) for p,t in zip(x,tt)])
            d=independent_distances(emp,gen,ref,features)
            row=dict(threads=threads,trajectory_alpha=model.trajectory_alpha,timing_alpha=model.timing_alpha,max_trajectory_coefficient=float(np.max(np.abs(model.trajectory_model.coef_))),max_timing_coefficient=float(np.max(np.abs(model.timing_model.coef_))),energy=d['energy_distance'],saved_energy=saved.energy_distance,mmd=d['mmd_rbf'],saved_mmd=saved.mmd_rbf)
            rows.append(row);gen.to_csv(OUT/f'ridge_precision_generated_threads{threads}.csv',index=False);print(row,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'ridge_precision_diagnostic.csv',index=False)

if __name__=='__main__':main()
