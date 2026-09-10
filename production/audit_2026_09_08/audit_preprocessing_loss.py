"""Independent cache transforms, selected raw CSV alignment and loss arithmetic."""
from pathlib import Path
import sys,json,pickle
sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch
from scipy.signal import butter,filtfilt
from scipy.interpolate import CubicSpline
import config
from src.vae_model import vae_loss,ConvCVAE
from src.preprocessing import lowpass_filter

def main():
    trials=pickle.load(open(ROOT/'studies/strategy_window_comparison/data/canonical_trials.pkl','rb'));errors=[];raw_count=0
    b,a=butter(4,10,fs=240);selected=set(np.random.default_rng(20260908).choice(len(trials),112,replace=False))
    for i,t in enumerate(trials):
        filtered=filtfilt(b,a,t['pos_raw'],axis=0)
        if not np.allclose(filtered,t['pos_filtered'],atol=1e-10):errors.append((i,'filter'))
        go=t['stim_onset_idx']+round(float(t['metadata']['go_signal_s'])*240)
        if go!=t['go_signal_idx'] or t['move_end_idx']!=len(filtered)-1:errors.append((i,'event index'))
        speed=np.sqrt(np.sum(np.gradient(filtered,1/240,axis=0)**2,axis=1))
        candidates=[j for j in range(go,len(filtered)-2) if np.all(speed[j:j+3]>5)]
        onset=candidates[0] if candidates else go
        if onset!=t['move_start_idx']:errors.append((i,'onset'))
        for key,start in [('pos_movement_norm',onset),('pos_go_to_arrival_norm',go)]:
            segment=filtered[start:];norm=CubicSpline(np.linspace(0,1,len(segment)),segment)(np.linspace(0,1,100));norm-=norm[0]
            if not np.allclose(norm,t[key],atol=1e-10):errors.append((i,key))
        if i in selected:
            m=t['metadata'];filename=m['trial_id'][len(m['subject'])+1:]+'.csv'
            raw=pd.read_csv(Path(config.DATA_RAW_DIR)/m['subject']/filename,header=None,names=config.CSV_COLUMNS)
            frame=raw[['frame','x','y','z','marker']].apply(pd.to_numeric,errors='coerce').dropna(subset=['frame','x','y','z'])
            frame['frame']=np.rint(frame.frame).astype(int);grouped=frame.groupby('frame').agg({'x':'mean','y':'mean','z':'mean','marker':'max'})
            grid=np.arange(grouped.index.min(),grouped.index.max()+1)
            pos=np.stack([np.interp(grid,grouped.index,grouped[k]) for k in ['x','y','z']],1)
            marker=int(grouped.index[grouped.marker==5][0]-grid[0])
            if not np.allclose(pos,t['pos_raw'],atol=1e-12) or marker!=t['stim_onset_idx']:errors.append((i,'raw CSV'))
            raw_count+=1
    rng=np.random.default_rng(19);arrays=[torch.tensor(rng.normal(size=s),dtype=torch.float64) for s in [(7,200),(7,200),(7,3),(7,3),(7,2),(7,2)]]
    recon,target,mu,lv,rt,tt=arrays
    actual=vae_loss(recon,target,mu,lv,.37,rt,tt,20)
    r=np.sum((recon.numpy()-target.numpy())**2,axis=1).mean();k=np.sum(.5*(np.exp(lv.numpy())+mu.numpy()**2-1-lv.numpy()),axis=1).mean();tm=np.sum((rt.numpy()-tt.numpy())**2,axis=1).mean()
    np.testing.assert_allclose([float(x) for x in actual],[r+20*tm+.37*k,r,k,tm],atol=1e-12)
    dormant={}
    try:ConvCVAE(variational=True,use_condition=True)
    except TypeError as e:dormant['cnn_constructor_rejects_trainer_flags']=str(e)
    try:lowpass_filter(np.zeros((15,2)))
    except ValueError as e:dormant['filter_length15_boundary']=str(e)
    result=dict(cached_trials_checked=len(trials),raw_csv_sample_checked=raw_count,raw_csv_sample_seed=20260908,errors=errors,independent_loss_formula_passed=True,min_cached_length=min(len(t['pos_raw']) for t in trials),dormant_defects=dormant)
    (OUT/'preprocessing_loss_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
