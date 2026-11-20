"""Bounded data-assisted PINN feasibility experiment (all predictions un-clipped)."""
import os
os.environ.setdefault('MPLBACKEND','Agg');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import time,json,copy,argparse
import numpy as np
import torch
from torch import nn
from physics import Physics,tensor
from data import OUT,TIMES,BOUNDS

torch.set_default_dtype(torch.float64);torch.set_num_threads(1)

def endpoints(y):
    # [design, condition, state] -> [design, condition, five populations]
    return -100*np.expm1(-y[...,87:92])
def aggregate(k):
    h=k[:,:,1:]@np.array([.79,.1,.1,.01])
    return np.c_[k[:,:,0].min(1),h.max(1),k[:,:,1:].max((1,2))]
def accuracy(actual,pred):
    from sklearn.metrics import r2_score
    a,b=aggregate(actual),aggregate(pred);out={}
    for j,name in enumerate(['tumour','overall_healthy','worst_subgroup']):
        d=b[:,j]-a[:,j]
        out[name]={'mae_pp':float(abs(d).mean()),'rmse_pp':float(np.sqrt((d*d).mean())),'r2':float(r2_score(a[:,j],b[:,j])),'max_error_pp':float(abs(d).max())}
    af=(a[:,1]<=.95)&(a[:,2]<=10);pf=(b[:,1]<=.95)&(b[:,2]<=10)
    out.update(n=len(a),actual_feasible=int(af.sum()),predicted_feasible=int(pf.sum()),false_feasible=int((pf&~af).sum()),false_infeasible=int((af&~pf).sum()),subgroup_mae_pp=abs(actual[:,:,1:]-pred[:,:,1:]).mean((0,1)).tolist())
    return out

class TrajectoryNet(nn.Module):
    def __init__(self,x,y):
        super().__init__();self.physics=Physics()
        self.register_buffer('mean',tensor(x.mean(0)));self.register_buffer('std',tensor(x.std(0)))
        self.register_buffer('scale',tensor(np.maximum(abs(y).max(axis=(0,1,2)),1e-8)))
        def net():
            layers=[nn.Linear(8,64),nn.Tanh()]
            for _ in range(3):layers.extend([nn.Linear(64,64),nn.Tanh()])
            layers.append(nn.Linear(64,97))
            return nn.Sequential(*layers)
        self.first=net();self.second=net()
        # Small nonzero output weights retain hidden-layer gradients initially.
        for n in [self.first,self.second]:nn.init.normal_(n[-1].weight,std=1e-3);nn.init.zeros_(n[-1].bias)
    def forward(self,t,x,c):
        context=torch.nn.functional.one_hot(c,3).to(torch.float64)
        inp=torch.cat([(x-self.mean)/self.std,context],-1)
        tau1=t/2;tau2=(t-2)/22
        start=self.physics.initial(c)
        first=start+tau1[:,None]*self.scale*self.first(torch.cat([tau1[:,None],inp],-1))
        bridge=start+self.scale*self.first(torch.cat([torch.ones_like(t[:,None]),inp],-1))
        second=bridge+tau2[:,None]*self.scale*self.second(torch.cat([tau2[:,None],inp],-1))
        return torch.where((t<=2)[:,None],first,second)
    def prediction(self,x,times=TIMES):
        # Bounded memory, full independent trajectories.
        parts=[]
        with torch.no_grad():
            for xx in x:
                tt=tensor(np.tile(times,3));cc=torch.arange(3).repeat_interleave(len(times));xt=tensor(np.tile(xx,(len(tt),1)))
                parts.append(self(tt,xt,cc).numpy().reshape(3,len(times),97))
        return np.stack(parts)
    def residual(self,t,x,c):
        y,dy=torch.func.jvp(lambda tt:self(tt,x,c),(t,),(torch.ones_like(t),))
        dt=torch.where(t<=2,2.,22.)
        r=(dy-self.physics.rhs(y,x,c,(t<=2).to(torch.float64)))*dt[:,None]/self.scale
        return y,r

def sample_collocation(n,rng):
    b=np.array(BOUNDS);x=rng.uniform(b[:,0],b[:,1],size=(n,4));c=rng.integers(0,3,n)
    # Equal allocation to intervals; mixture covers transient and late dynamics.
    t=np.r_[np.exp(rng.uniform(np.log(1e-6),np.log(2),n//2)),rng.uniform(2,24,n-n//2)]
    return tensor(t),tensor(x),torch.as_tensor(c)

def physical_metrics(model,x,y):
    pred=model.prediction(x);negative=float(np.maximum(-pred/model.scale.numpy(),0).max())
    mass=0.
    with torch.no_grad():
        for row in pred:
            z=tensor(row.reshape(-1,97));t=tensor(np.tile(TIMES,3));c=torch.arange(3).repeat_interleave(len(TIMES))
            mass=max(mass,float(model.physics.balance(z,t,c).abs().max()))
    d=accuracy(endpoints(y[:,:,-1]),endpoints(pred[:,:,-1]))
    d.update(finite=bool(np.isfinite(pred).all()),negative_scaled_max=negative,mass_fraction_max=mass,trajectory_scaled_mae=float(np.mean(abs(pred-y)/model.scale.numpy())))
    return d,pred

def fit(x,y,xv,yv,seconds=360,seed=2026,kind='PINN',prefix='pilot'):
    torch.manual_seed(seed);rng=np.random.default_rng(seed);model=TrajectoryNet(x,y)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    xt=tensor(x);yt=tensor(y);times=tensor(TIMES)
    tc,xc,cc=sample_collocation(128,rng)
    start=time.perf_counter();lastval=start;history=[];best=float('inf');beststate=None;bestrecord=None;step=0
    while time.perf_counter()-start<seconds:
        elapsed=time.perf_counter()-start;progress=elapsed/seconds
        weight=float(np.clip((progress-.2)/.2,0,1)) if kind=='PINN' else 0.
        i=torch.as_tensor(rng.integers(0,len(x),256));c=torch.as_tensor(rng.integers(0,3,256));ti=torch.as_tensor(rng.integers(0,len(TIMES),256))
        pred=model(times[ti],xt[i],c);data=((pred-yt[i,c,ti])/model.scale).square().mean()
        neg=torch.relu(-pred/model.scale).square().mean();phys=torch.zeros(());bal=torch.zeros(())
        if weight:
            py,r=model.residual(tc,xc,cc);phys=r.square().mean();bal=model.physics.balance(py,tc,cc).square().mean()
            neg=neg+torch.relu(-py/model.scale).square().mean()
        loss=data+neg+weight*(phys+bal)
        if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss at step {step}; no clipping or silent substitution')
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();step+=1
        if step%200==0:
            nt,nx,nc=sample_collocation(1024,rng)
            with torch.no_grad():
                # Forward AD remains enabled inside no_grad; no training graph retained.
                _,rr=model.residual(nt,nx,nc);ids=rr.square().mean(-1).topk(64).indices
            fresh=sample_collocation(64,rng)
            tc=torch.cat([nt[ids],fresh[0]]);xc=torch.cat([nx[ids],fresh[1]]);cc=torch.cat([nc[ids],fresh[2]])
        if time.perf_counter()-lastval>=20 or step==1:
            with torch.no_grad():vp=model.prediction(xv,np.array([24.]))
            met=accuracy(endpoints(yv[:,:,-1]),endpoints(vp[:,:,0]))
            score=float(np.mean([met[n]['mae_pp']/s for n,s in zip(['tumour','overall_healthy','worst_subgroup'],[100,.95,10])]))
            rec={'step':step,'seconds':time.perf_counter()-start,'physics_weight':weight,'data_loss':float(data.detach()),'ode_loss':float(phys.detach()),'balance_loss':float(bal.detach()),'negative_loss':float(neg.detach()),'validation_score':score,'validation_metrics':met}
            history.append(rec);lastval=time.perf_counter()
            # PINN checkpoint must have received the complete equation loss.
            if (kind!='PINN' or weight>=1) and score<best:
                best=score;beststate=copy.deepcopy(model.state_dict());bestrecord=rec
            print(prefix,'step',step,'s',round(rec['seconds']),'w',round(weight,2),'val',round(score,5),'ODE',f'{rec["ode_loss"]:.3g}',flush=True)
            (OUT/f'{prefix}_training.json').write_text(json.dumps(history,indent=2))
    if beststate is None:raise RuntimeError('No eligible full-physics checkpoint')
    model.load_state_dict(beststate);duration=time.perf_counter()-start
    torch.save({'state_dict':model.state_dict(),'seed':seed,'kind':kind,'selected':bestrecord,'training_seconds':duration,'steps':step},OUT/f'{prefix}_checkpoint.pt')
    np.savez(OUT/f'{prefix}_scaling.npz',mean=model.mean.numpy(),std=model.std.numpy(),state_scale=model.scale.numpy())
    return model,{'training_seconds':duration,'steps':step,'selected_checkpoint':bestrecord,'kind':kind}

def main(seconds=360):
    start=time.perf_counter();d=np.load(OUT/'initial_2026.npz');v=np.load(OUT/'validation.npz');x,y=d['x'],d['y'];xv,yv=v['x'],v['y']
    model,training=fit(x,y,xv,yv,seconds=seconds)
    # Hard initial condition, bridge continuity and time derivative tests.
    c=torch.arange(3);xx=tensor(np.tile(x[0],(3,1)));zero=model(torch.zeros(3),xx,c)
    torch.testing.assert_close(zero,model.physics.initial(c),rtol=0,atol=0)
    t=tensor([.7,1.1,9.]);_,dy=torch.func.jvp(lambda tt:model(tt,xx,c),(t,),(torch.ones_like(t),));eps=1e-5
    fd=(model(t+eps,xx,c)-model(t-eps,xx,c))/(2*eps)
    torch.testing.assert_close(dy,fd,rtol=1e-5,atol=1e-7)
    met,pred=physical_metrics(model,xv,yv)
    checks={'finite':met['finite'],'tumour':met['tumour']['mae_pp']<5,'overall_healthy':met['overall_healthy']['mae_pp']<.25,'worst_subgroup':met['worst_subgroup']['mae_pp']<3,'nonnegative':met['negative_scaled_max']<=.001,'material_balance':met['mass_fraction_max']<.01}
    result={'status':'passed' if all(checks.values()) else 'stopped_at_feasibility_gate','checks':checks,'validation':met,'training':training,'pilot_seconds':time.perf_counter()-start,'automatic_differentiation_check':True,'initial_condition_check':True,'checkpoint_policy':'Best validation endpoint score among checkpoints receiving full physics weight; no test data used'}
    np.savez_compressed(OUT/'pilot_validation_predictions.npz',x=xv,prediction=pred,actual=yv,t=TIMES)
    # Open held-out set only after checkpoint selection and gate decision.
    te=np.load(OUT/'test.npz');testmet,testpred=physical_metrics(model,te['x'],te['y']);result['test']=testmet
    np.savez_compressed(OUT/'pilot_test_predictions.npz',x=te['x'],prediction=testpred,actual=te['y'],t=TIMES)
    result['pilot_seconds']=time.perf_counter()-start
    (OUT/'pilot_results.json').write_text(json.dumps(result,indent=2))
    print('GATE',result['status'],checks,flush=True);print('VALIDATION',met,flush=True)
    if not all(checks.values()):print('Search comparison intentionally not run under predeclared stopping rule.',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seconds',type=int,default=360);a=ap.parse_args();main(a.seconds)
