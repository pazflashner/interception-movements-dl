"""Read-only frozen-artifact checks, independent arithmetic and metamorphic tests.
Run from any directory with Python; writes only beside this script.
"""
from pathlib import Path
import sys, json, pickle, re, hashlib, traceback, warnings
sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.io import loadmat
from scipy.stats import wilcoxon, false_discovery_control
from threadpoolctl import threadpool_limits
import config
from src.trajectory_view import select_trials_window, project_trials_to_table_plane
from src.vae_model import encode_trial_condition, TrajectoryDataset, vae_loss
from scripts.run_corrected_study import load_per_trial_checkpoint, context_query_for_trials
from src.baseline_spline import SplinePCARepresentation
from src.confirmatory_spline import TimingRidge
from src.confirmatory_controls import ConditionRidge
from src.features import kinematic_features_for_dim
torch.set_num_threads(1)
threadpool_limits(1)
SOURCE = ROOT/'studies/final_strategy_evaluation'
CORRECTED = ROOT/'studies/review_corrected_evaluation'
checks, neural_rows, participant_rows = [], [], []
METRICS = ['trajectory_mse','movement_time_mae_ms','initiation_time_mae_ms','mean_ks','ks_n_submovements','energy_distance','mmd_rbf']

def check(name, ok, detail=''):
    checks.append(dict(check=name, passed=bool(ok), detail=str(detail)))
    if not ok:
        print('FAIL', name, detail, flush=True)

def close(name, a, b, atol=1e-6, rtol=1e-5):
    a,b=np.asarray(a),np.asarray(b)
    ok = a.shape == b.shape and np.allclose(a,b,atol=atol,rtol=rtol,equal_nan=False)
    error=float(np.max(np.abs(a-b))) if a.shape==b.shape and a.size else None
    check(name,ok,f'max_abs_error={error}; atol={atol}; rtol={rtol}')
    return error

def raw_timing(ts):
    return np.array([[(t['move_end_idx']-t['move_start_idx'])/240,
                      (t['move_start_idx']-t['go_signal_idx'])/240] for t in ts], dtype=np.float32)

def conditions(ts):
    rows=[]
    for t in ts:
        m=t['metadata']; row=[0.,0.,0.,float(m['side']==2),float(m['target_speed_screen_s'])]
        row[int(m['sp'])-1]=1.;rows.append(row)
    return np.asarray(rows,dtype=np.float32)

def manual_encode(c,x,cond):
    s=c['model_state']; cfg=c['config']['model']; used=c.get('use_condition',cfg.get('use_condition',True))
    inp=torch.cat([x,cond if used else torch.zeros_like(cond)],1)
    h=F.relu(F.linear(inp,s['encoder.0.weight'],s['encoder.0.bias']))
    h=F.relu(F.linear(h,s['encoder.2.weight'],s['encoder.2.bias']))
    return F.linear(h,s['fc_mu.weight'],s['fc_mu.bias'])

def manual_decode(c,z,cond):
    s=c['model_state'];cfg=c['config']['model'];used=c.get('use_condition',cfg.get('use_condition',True))
    h=F.relu(F.linear(torch.cat([z,cond if used else torch.zeros_like(cond)],1),s['decoder.0.weight'],s['decoder.0.bias']))
    h=F.relu(F.linear(h,s['decoder.2.weight'],s['decoder.2.bias']))
    return (F.linear(h,s['traj_head.weight'],s['traj_head.bias']),
            F.linear(h,s['timing_head.weight'],s['timing_head.bias']))

def main():
    cached=pickle.load(open(ROOT/'studies/strategy_window_comparison/data/canonical_trials.pkl','rb'))
    trials=project_trials_to_table_plane(select_trials_window(cached,config.WINDOW_GO_TO_ARRIVAL))
    check('canonical cohort', len(trials)==4732 and len({t['metadata']['subject'] for t in trials})==28)
    check('unique trial IDs',len({t['metadata']['trial_id'] for t in trials})==len(trials))
    check('finite 100x2 origin-aligned model targets',all(t['pos_norm'].shape==(100,2) and np.isfinite(t['pos_norm']).all() and np.all(t['pos_norm'][0]==0) for t in trials))
    check('condition metadata domains',all(int(t['metadata']['condition'])==2 and int(t['metadata']['sp']) in (1,2,3) and int(t['metadata']['side']) in (1,2) and np.isfinite(t['metadata']['target_speed_screen_s']) for t in trials))
    close('independent condition matrix vs dataset',conditions(trials),TrajectoryDataset(trials).conditions,atol=0,rtol=0)
    close('timing order and values vs dataset',raw_timing(trials),TrajectoryDataset(trials).timings,atol=0,rtol=0)
    data_errors=[]; raw_speeds=[]
    for i,t in enumerate(trials):
        m=t['metadata']; parts=m['trial_id'].split('_'); subj=parts[0]; nums=list(map(int,parts[2:]))
        if nums != [int(m[k]) for k in ['condition','sp','side','rep']]:data_errors.append((m['trial_id'],'filename'))
        path=Path(config.DATA_RAW_DIR)/subj/('trialinfo_'+'_'.join(parts[2:])+'.mat')
        mat=loadmat(path,simplify_cells=True)['thistrial']
        dots=np.asarray(mat['dotArray']); steps=np.sqrt(np.sum((dots[1:]-dots[:-1])**2,axis=1)); moving=np.flatnonzero(steps>1e-6)
        onset=(int(moving[0])+1)/60; speed=float(np.median(steps[moving])*60/1920)
        if not np.isclose(onset,m['go_signal_s'],atol=1e-12,rtol=0) or not np.isclose(speed,m['target_speed_screen_s'],atol=1e-12,rtol=0):data_errors.append((m['trial_id'],'MAT motion'))
        stimulus=mat.get('thisstimulus',{}).get('dotsFilename','')
        expected=f'subject{int(subj[7:])}condition{nums[0]}sp{nums[1]}ss{nums[2]}repetition{nums[3]}.csv'
        if stimulus!=expected:data_errors.append((m['trial_id'],'stimulus name',stimulus,expected))
        raw_speeds.append(speed)
        if i%1000==0:print('MAT metadata verified',i,flush=True)
    check('all retained raw MAT motion and stimulus labels match cached conditions',not data_errors,json.dumps(data_errors[:20]))
    folds={}
    for f in range(4):
        split=json.loads((CORRECTED/f'runs/cvae/fold{f}/cvae_z3_seed42/split.json').read_text())
        groups=[set(split[k]) for k in ['train_subjects','validation_subjects','test_subjects']]
        check(f'fold{f} separation',list(map(len,groups))==[17,4,7] and not(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2]))
        folds[f]=(split,*[[t for t in trials if t['metadata']['subject'] in g] for g in groups])
        for label,ts in zip(['train','validation','test'],folds[f][1:]):
            for sp in context_query_for_trials(ts,config.CONTEXT_QUERY_SEED):
                a,b=set(sp.context_indices),set(sp.query_indices)
                expected={j for j,t in enumerate(ts) if t['metadata']['subject']==sp.subject}
                check(f'fold{f}/{label}/{sp.subject} context-query',bool(a) and bool(b) and not a&b and a|b==expected)
    check('every participant held out exactly once',len([s for f in folds.values() for s in f[0]['test_subjects']])==len({s for f in folds.values() for s in f[0]['test_subjects']})==28)
    paths=sorted((SOURCE/'runs').glob('*/*/*/checkpoint.pt'))
    check('96 neural checkpoints',len(paths)==96,str(len(paths)))
    for idx,p in enumerate(paths):
        rel=p.parent.relative_to(SOURCE/'runs');family=rel.parts[0];f=int(rel.parts[1][4:]);split,train,val,test=folds[f]
        ck=torch.load(p,map_location='cpu',weights_only=False);mc=ck['config']['model'];tc=ck['config']['train']
        model,norm=load_per_trial_checkpoint(p,'cpu');model.eval()
        expected_flags=(family!='conditional_ae',family!='unconditional_vae')
        check(f'{rel}/effective flags',(model.variational,model.use_condition)==expected_flags and not model.encoder_uses_timing and model.input_dim==200 and model.condition_dim==5 and model.timing_dim==2 and mc['architecture']=='mlp')
        xs=np.stack([t['pos_norm'].reshape(-1) for t in train]).astype(np.float32)
        yt=np.log(raw_timing(train)+np.float32(.001))
        for key,expected in [('train_mean',xs.mean(0)),('train_std',xs.std(0)+1e-8),('timing_mean',yt.mean(0)),('timing_std',yt.std(0)+1e-8)]:
            close(f'{rel}/{key} train-only',ck[key],expected,atol=1e-7,rtol=1e-6)
        dest=CORRECTED/'runs'/rel
        hist=json.loads((dest/'history.json').read_text())
        expected_obj=np.array(hist['val_recon'])+tc['timing_weight']*np.array(hist['val_timing'])+tc['kl_weight']*np.array(hist['val_kl'])
        close(f'{rel}/validation objective',hist['val_objective'],expected_obj,atol=1e-10,rtol=1e-10)
        check(f'{rel}/selected epoch',ck['epoch']==int(np.argmin(expected_obj))+1)
        close(f'{rel}/KL schedule',hist['beta'],tc['kl_weight']*np.minimum(np.arange(len(hist['beta']))/50,1),atol=1e-12,rtol=0)
        close(f'{rel}/saved objective',ck['val_objective'],expected_obj[ck['epoch']-1],atol=1e-10,rtol=1e-10)
        xx=np.stack([t['pos_norm'].reshape(-1) for t in test]).astype(np.float32)
        x=torch.from_numpy((xx-norm.train_mean)/norm.train_std);cond=torch.from_numpy(conditions(test))
        with torch.no_grad():
            mu=manual_encode(ck,x,cond);actual_mu,lv=model.encode(x,cond)
            close(f'{rel}/independent encoder',mu.numpy(),actual_mu.numpy(),atol=2e-6)
            rz,tz=manual_decode(ck,mu,cond)
            pred=(rz.numpy()*norm.train_std+norm.train_mean)
            # Literal, independently written inverse of standardized log seconds.
            time=np.maximum(np.exp(tz.numpy()*norm.timing_std+norm.timing_mean)-.001,0)
            timing=pd.read_csv(dest/'timing_predictions.csv');rec=pd.read_csv(dest/'reconstruction_predictions.csv')
            ids=[t['metadata']['trial_id'] for t in test]
            check(f'{rel}/prediction IDs',ids==timing.trial_id.tolist()==rec.trial_id.tolist())
            close(f'{rel}/true timing',timing[['movement_time_true_s','initiation_time_true_s']],raw_timing(test),atol=1e-7)
            terr=close(f'{rel}/independent timing predictions',timing[['movement_time_pred_s','initiation_time_pred_s']],time,atol=2e-6,rtol=2e-5)
            mse=np.mean((pred-xx)**2,axis=1)
            merr=close(f'{rel}/independent trajectory MSE',rec.trajectory_mse_tracker_units2,mse,atol=2e-6,rtol=2e-5)
            a,b=model.decode(mu,cond)
            close(f'{rel}/independent decoder trajectory',a.numpy(),rz.numpy(),atol=2e-6)
            close(f'{rel}/independent decoder timing',b.numpy(),tz.numpy(),atol=2e-6)
            poisoned=model.encode(x[:24],cond[:24],torch.full((24,2),999.))[0]
            close(f'{rel}/timing withheld',poisoned.numpy(),model.encode(x[:24],cond[:24])[0].numpy(),atol=0,rtol=0)
            z1=model.reparameterize(mu[:24],lv[:24]);z2=model.reparameterize(mu[:24],lv[:24])
            check(f'{rel}/variational sampling behaviour',not torch.equal(z1,z2) if model.variational else torch.equal(z1,mu[:24]) and torch.equal(z2,mu[:24]))
        sensitivity={}
        for j in range(5):
            cc=cond[:24].clone().requires_grad_(True)
            enc=model.encode(x[:24],cc)[0]
            egv=torch.autograd.grad(enc.square().sum(),cc,allow_unused=True)[0]
            eg=0. if egv is None else egv[:,j].abs().max().item()
            dc=cond[:24].clone().requires_grad_(True)
            dec,dt=model.decode(mu[:24].detach(),dc)
            dgv=torch.autograd.grad(dec.square().sum()+dt.square().sum(),dc,allow_unused=True)[0]
            dg=0. if dgv is None else dgv[:,j].abs().max().item()
            sensitivity[f'encoder_condition{j}_gradient']=eg;sensitivity[f'decoder_condition{j}_gradient']=dg
        check(f'{rel}/condition gradients',all(v>0 for v in sensitivity.values()) if model.use_condition else all(v==0 for v in sensitivity.values()),str(sensitivity))
        # Explicit valid side/category swaps, not just gradient reachability.
        swapped=cond[:24].clone();swapped[:,:3]=torch.roll(swapped[:,:3],1,1);swapped[:,3]=1-swapped[:,3];swapped[:,4]=torch.flip(swapped[:,4],[0])
        with torch.no_grad():
            encoder_change=float((model.encode(x[:24],cond[:24])[0]-model.encode(x[:24],swapped)[0]).abs().max())
            decoder_change=float((model.decode(mu[:24],cond[:24])[0]-model.decode(mu[:24],swapped)[0]).abs().max())
        check(f'{rel}/condition swap behaviour',encoder_change>0 and decoder_change>0 if model.use_condition else encoder_change==0 and decoder_change==0)
        neural_rows.append(dict(run=str(rel),family=family,dim=ck['latent_dim'],fold=f,seed=ck['config']['seed'],epoch=ck['epoch'],epochs_run=len(hist['beta']),checkpoint_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),max_timing_error=terr,max_mse_error=merr,encoder_swap_change=encoder_change,decoder_swap_change=decoder_change,**sensitivity))
        if idx%8==0:print('checkpoints verified',idx+1,'/',len(paths),flush=True)
    # Independently recompute subject errors from individual prediction CSVs.
    allruns=sorted((CORRECTED/'runs').glob('*/*/*/result.json'))
    check('124 corrected evaluations',len(allruns)==124)
    for rp in allruns:
        r=json.loads(rp.read_text());d=rp.parent;f=int(r['outer_fold']);split,train,val,test=folds[f]
        check(str(d.relative_to(CORRECTED))+'/same test cohort',r['test_subjects']==split['test_subjects'])
        timing=pd.read_csv(d/'timing_predictions.csv');rec=pd.read_csv(d/'reconstruction_predictions.csv');fid=pd.read_csv(d/'context_query_fidelity.csv').set_index('subject')
        ids=[t['metadata']['trial_id'] for t in test]
        check(str(d.relative_to(CORRECTED))+'/complete trial coverage',ids==timing.trial_id.tolist()==rec.trial_id.tolist())
        for subj,g in timing.groupby('subject'):
            ff=fid.loc[subj]
            participant_rows.append(dict(model_family=r['model_family'],latent_dim=int(r['latent_dim'] or 0),outer_fold=f,training_seed=r.get('training_seed') or r.get('generation_seed'),subject=subj,trajectory_mse=rec.loc[rec.subject==subj,'trajectory_mse_tracker_units2'].mean(),movement_time_mae_ms=np.mean(np.abs(g.movement_time_true_s-g.movement_time_pred_s))*1000,initiation_time_mae_ms=np.mean(np.abs(g.initiation_time_true_s-g.initiation_time_pred_s))*1000,mean_ks=np.mean([ff['ks_'+k] for k in kinematic_features_for_dim(2)]),ks_n_submovements=ff.ks_n_submovements,energy_distance=ff.energy_distance,mmd_rbf=ff.mmd_rbf))
    pp=pd.DataFrame(participant_rows);pp.to_csv(OUT/'independent_participant_metrics.csv',index=False)
    saved=pd.read_csv(CORRECTED/'results/analysis/participant_metrics_raw.csv')
    keys=['model_family','latent_dim','outer_fold','training_seed','subject']
    pp=pp.sort_values(keys).reset_index(drop=True);saved=saved.sort_values(keys).reset_index(drop=True)
    check('868 participant rows',len(pp)==len(saved)==868)
    close('participant metrics rebuilt from prediction CSVs',pp[METRICS],saved[METRICS],atol=1e-10,rtol=1e-10)
    avg=pp.groupby(['model_family','latent_dim','outer_fold','subject'])[METRICS].mean().reset_index()
    old=pd.read_csv(ROOT/'review_evidence/main_paired_model_comparisons.csv');pvalues=[]
    for _,r in old.iterrows():
        a=avg[(avg.model_family=='cvae')&(avg.latent_dim==r.latent_dim)].set_index('subject').sort_index()
        b=avg[(avg.model_family==r.comparator)&(avg.latent_dim==(0 if r.comparator=='condition_ridge' else r.latent_dim))].set_index('subject').loc[a.index]
        diff=np.round(b[r.metric].to_numpy()-a[r.metric].to_numpy(),12)
        w=wilcoxon(diff,zero_method='pratt',alternative='two-sided',method='auto') if np.any(diff) else (0.,1.)
        pvalues.append(float(w[1]));close(f'paired {r.latent_dim}/{r.comparator}/{r.metric}',[w[0],w[1],a[r.metric].mean(),b[r.metric].mean()],[r.wilcoxon_statistic,r.wilcoxon_p_uncorrected,r.cvae_mean,r.comparator_mean],atol=1e-10,rtol=1e-10)
    close('BH adjustment of all 56 tests',false_discovery_control(pvalues),old.wilcoxon_p_fdr_bh,atol=1e-12,rtol=1e-12)
    pd.DataFrame(neural_rows).to_csv(OUT/'checkpoint_audit.csv',index=False)
    # Refit classical models from train, select only on validation, compare test outputs.
    for f,(_,train,val,test) in folds.items():
        xx=np.stack([t['pos_norm'] for t in test]);vv=np.stack([t['pos_norm'] for t in val])
        for dim in [2,3,4,8]:
            d=CORRECTED/f'runs/spline_pca/fold{f}/spline_pca_z{dim}';r=json.loads((d/'result.json').read_text())
            reps=[SplinePCARepresentation(n_components=dim,include_timing=False,standardize_coefficients=st).fit(train) for st in [False,True]]
            vm=[float(np.mean((rep.decode(rep.encode(val))[0]-vv)**2)) for rep in reps]
            selected=int(np.argmin(vm));rep=reps[selected]
            check(f'spline fold{f}/z{dim} validation selection',bool(selected)==r['standardize_spline_coefficients'])
            close(f'spline fold{f}/z{dim} validation scores',vm,[r['validation_mse_raw_coefficients'],r['validation_mse_standardized_coefficients']],atol=1e-8)
            pred=rep.decode(rep.encode(test))[0]
            close(f'spline fold{f}/z{dim} MSE',np.mean((pred-xx)**2,axis=(1,2)),pd.read_csv(d/'reconstruction_predictions.csv').trajectory_mse_tracker_units2,atol=1e-8)
            ridge=TimingRidge.fit(rep,train,val);time=ridge.predict(rep.encode(test),conditions(test))
            close(f'spline fold{f}/z{dim} timing',time,pd.read_csv(d/'timing_predictions.csv')[['movement_time_pred_s','initiation_time_pred_s']],atol=1e-7)
            poisoned=[dict(t,move_start_idx=t['move_start_idx']+100) for t in test[:3]]
            close(f'spline fold{f}/z{dim} timing withheld',rep.encode(test[:3]),rep.encode(poisoned),atol=0,rtol=0)
        ridge=ConditionRidge.fit(train,val)
        pred,time=ridge.predict(test)
        for seed in [42,43,44]:
            d=CORRECTED/f'runs/condition_ridge/fold{f}/condition_ridge_seed{seed}'
            close(f'ridge fold{f}/seed{seed} MSE',np.mean((pred.reshape(-1,100,2)-xx)**2,axis=(1,2)),pd.read_csv(d/'reconstruction_predictions.csv').trajectory_mse_tracker_units2,atol=1e-7)
            close(f'ridge fold{f}/seed{seed} timing',time,pd.read_csv(d/'timing_predictions.csv')[['movement_time_pred_s','initiation_time_pred_s']],atol=1e-7)
        print('classical fold verified',f,flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        check('audit execution completed',False,traceback.format_exc());print(traceback.format_exc(),flush=True)
    finally:
        pd.DataFrame(checks).to_csv(OUT/'integrity_checks.csv',index=False)
        pd.DataFrame(neural_rows).to_csv(OUT/'checkpoint_audit.csv',index=False)
        summary=dict(checks=len(checks),passed=sum(r['passed'] for r in checks),failed=sum(not r['passed'] for r in checks),neural_checkpoints=len(neural_rows),python=sys.version,torch=torch.__version__,numpy=np.__version__)
        (OUT/'integrity_summary.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps(summary),flush=True)
