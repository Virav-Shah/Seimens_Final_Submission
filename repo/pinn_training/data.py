"""Full trajectories from the unmodified reference RHS, with tighter tolerances."""
import os
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json,time,hashlib
from pathlib import Path
from dataclasses import asdict
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import qmc
from joblib import Parallel,delayed
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mechanism_comparison.model import INPUT_SET,make_rhs,density,BOUNDS
OUT=Path(__file__).resolve().parents[2]/'output/pinn_small_trial'
TIMES=np.r_[0,np.geomspace(1e-6,2,64),np.linspace(2,24,65)[1:]]
SEEDS=[2026,8407,11939]

def trajectory(theta,condition):
    _,s,p=INPUT_SET[condition];y0=np.zeros(97);y0[7:87].reshape(5,16)[:,:2]=density(s)
    result=[];tt=[]
    for start,end,on in [(0,2,True),(2,24,False)]:
        material=make_rhs(theta,s,on)
        def rhs(t,y):
            conc=y[7:87].reshape(5,16)[:,14]
            return np.r_[material(t,y[:87]),np.array(p.maximum_rate_h)*conc/(np.array(p.c50_uM)+conc),conc]
        sol=solve_ivp(rhs,(start,end),y0,method='BDF',rtol=1e-8,atol=1e-11,dense_output=True)
        if not sol.success:raise RuntimeError(sol.message)
        ts=TIMES[(TIMES>=start)&(TIMES<=end)] if start==0 else TIMES[TIMES>start]
        result.extend(sol.sol(ts).T);tt.extend(ts);y0=sol.y[:,-1]
    y=np.array(result);z=y[:,7:87].reshape(-1,5,16)
    cap=s.total_cells*np.array(s.fractions)*s.cell_area*1e-4;vol=s.total_cells*np.array(s.fractions)*s.intracellular_volume
    ab=s.antibody_vc*(y[:,0]+y[:,2])+s.antibody_vp*(y[:,1]+y[:,3])+z[:,:,2:14].sum(-1)@cap+y[:,5]
    pl=s.antibody_vc*y[:,0]+s.antibody_vp*y[:,1]+(z[:,:,2:5]+z[:,:,8:11]).sum(-1)@cap+s.payload_volume*y[:,4]+z[:,:,14]@vol+y[:,6]
    dose=s.dose*np.minimum(TIMES/2,1)
    mass=max(abs(ab-dose).max(),abs(pl-dose).max())/s.dose
    if not np.isfinite(y).all() or y.min() < -2e-6 or mass>2e-6:raise RuntimeError(f'trajectory failed: min={y.min()} mass={mass}')
    return y

def one(theta):
    st=time.perf_counter()
    try:return {'y':np.stack([trajectory(theta,c) for c in range(3)]),'seconds':time.perf_counter()-st,'error':None}
    except Exception as e:return {'x':list(theta),'seconds':time.perf_counter()-st,'error':repr(e)}

def coordinates(n,seed):
    b=np.array(BOUNDS);return qmc.scale(qmc.LatinHypercube(4,seed=seed).random(n),b[:,0],b[:,1])

def generate(name,n,seed):
    path=OUT/f'{name}.npz'
    if path.exists():
        d=np.load(path);return d['x'],d['y']
    x=coordinates(n,seed);st=time.perf_counter()
    rows=Parallel(n_jobs=4)(delayed(one)(a) for a in x)
    errs=[r for r in rows if r['error']]
    timing={'wall_seconds':time.perf_counter()-st,'solve_seconds':[r['seconds'] for r in rows],'errors':errs}
    (OUT/f'{name}_generation.json').write_text(json.dumps(timing,indent=2))
    if errs:raise RuntimeError('Trajectory errors retained; generation stopped')
    y=np.stack([r['y'] for r in rows]);np.savez_compressed(path,x=x,y=y,t=TIMES)
    print(name,y.shape,'seconds',round(timing['wall_seconds'],2),flush=True)
    return x,y

def preflight():
    import torch
    from physics import Physics,tensor
    ph=Physics();torch.set_num_threads(1);rng=np.random.default_rng(744)
    max_abs=0.;num=0
    for theta in coordinates(3,2026):
        for c in range(3):
            y=trajectory(theta,c)
            for on,indices in [(True,[1,20,48,64]),(False,[65,88,128])]:
                states=y[indices];_,s,p=INPUT_SET[c];rhs=make_rhs(theta,s,on)
                expected=[]
                for ti,yi in zip(TIMES[indices],states):
                    pp=yi[7:87].reshape(5,16)[:,14]
                    expected.append(np.r_[rhs(ti,yi[:87]),np.array(p.maximum_rate_h)*pp/(np.array(p.c50_uM)+pp),pp])
                actual=ph.rhs(tensor(states),tensor(np.tile(theta,(len(states),1))),torch.full((len(states),),c),tensor(float(on))).numpy()
                np.testing.assert_allclose(actual,expected,rtol=1e-8,atol=1e-10)
                max_abs=max(max_abs,float(np.max(abs(actual-expected))));num+=len(states)
                # Physical material accounts agree over the whole trajectory.
                bal=ph.balance(tensor(y),tensor(TIMES),torch.full((len(y),),c))
                assert float(bal.abs().max())<2e-6
    out={'passed':True,'states_checked':num,'derivatives_per_state':97,'max_absolute_difference':max_abs,'rtol':1e-8,'atol':1e-10}
    (OUT/'equation_checks.json').write_text(json.dumps(out,indent=2));print('equation checks',out,flush=True)
    return out

if __name__=='__main__':
    OUT.mkdir(exist_ok=True,parents=True)
    import platform,torch,scipy
    ref=Path(__file__).resolve().parents[1]/'mechanism_comparison/model.py'
    cfg={'reference_sha256':hashlib.sha256(ref.read_bytes()).hexdigest(),'conditions':[{'name':n,'biology':asdict(s),'killing':asdict(p)} for n,s,p in INPUT_SET],'bounds_log10':BOUNDS,'seeds':SEEDS,'times_h':TIMES.tolist(),'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'scipy':scipy.__version__,'numpy':np.__version__,'device':'CPU','dtype':'float64','budget_seconds':7200,'initial_neural_seconds':360,'gate':{'tumour_mae_pp':5,'healthy_mae_pp':.25,'subgroup_mae_pp':3,'negative_scaled_max':.001,'mass_fraction_max':.01}}
    (OUT/'configuration.json').write_text(json.dumps(cfg,indent=2))
    preflight()
    generate('initial_2026',64,2026);generate('validation',32,5511);generate('test',64,9911)
    sets=[set(map(tuple,np.load(OUT/f'{n}.npz')['x'])) for n in ['initial_2026','validation','test']]
    assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
    print('Data partitions disjoint',flush=True)
