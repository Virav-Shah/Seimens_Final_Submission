"""Independent checks of trajectory dynamics, beyond enforced accounting."""
import os
os.environ.setdefault('MPLBACKEND','Agg');os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json
import numpy as np
import torch
from balanced_trial import OUT,OLD,GROUPS
from balanced_model import BalancedNet,KNOTS
from train import tensor,endpoints,aggregate
from data import TIMES

def main():
    train=np.load(OLD/'initial_2026.npz');test=np.load(OUT/'fresh_test_data.npz');rng=np.random.default_rng(757)
    # Shared sample choices for the two trained networks.
    ids=rng.integers(0,64,256);c=torch.as_tensor(rng.integers(0,3,256));valid=np.where(~np.isin(TIMES,KNOTS))[0];ti=rng.choice(valid,256)
    t=tensor(TIMES[ti]);x=tensor(test['x'][ids]);actual=tensor(test['y'][ids,c.numpy(),ti]);audit={}
    dense=np.r_[0,np.geomspace(1e-6,2,512),np.linspace(2,24,513)[1:]];sample=np.linspace(0,63,16,dtype=int)
    for kind in ['PINN','Supervised']:
        m=BalancedNet(train['x'],train['y']);saved=torch.load(OUT/f'{kind}_checkpoint.pt',weights_only=False);m.load_state_dict(saved['state_dict'])
        with torch.no_grad():
            y,dy=torch.func.jvp(lambda tt:m(tt,x,c),(t,),(torch.ones_like(t),))
            rhs=m.physics.rhs(y,x,c,(t<2).to(torch.float64));truth=m.physics.rhs(actual,x,c,(t<2).to(torch.float64))
        residual={}
        for name,ix in GROUPS.items():
            denom=torch.maximum(torch.sqrt((truth[:,ix]**2).mean(0)),m.scale[ix]/24)
            err=((dy[:,ix]-rhs[:,ix])/denom).square().mean().sqrt()
            residual[name]=float(err)
        pred=m.prediction(test['x'][sample],dense);P=pred[:,:,:,7:87].reshape(16,3,len(dense),5,16)[:,:,:,:,14]
        hazardrate=m.physics.kmax.numpy()[None,:,None,:]*P/(m.physics.c50.numpy()[None,:,None,:]+P)
        integrated=np.trapezoid(hazardrate,dense,axis=2);recomputed=-100*np.expm1(-integrated);network=endpoints(pred[:,:,-1])
        delta=abs(aggregate(recomputed)-aggregate(network))
        audit[kind]={'sampled_equation_residual_relative_rms_by_group':residual,'residual_normalisation':'per-component RMS of reference derivative, floored at training state scale / 24 h; 256 shared fresh-test states','payload_to_killing_consistency':{'audit_design_indices':sample.tolist(),'integration_points':len(dense),'aggregate_mean_difference_pp':delta.mean(0).tolist(),'aggregate_max_difference_pp':delta.max(0).tolist(),'population_mean_difference_pp':abs(recomputed-network).mean((0,1)).tolist()},'loss_accounts_negative_derivative_count':int((dy[:,[5,6]]< -1e-8).sum()),'loss_accounts_derivatives_checked':512,'hazard_rate_above_max_count':int((dy[:,87:92]>m.physics.kmax[c]+1e-8).sum()),'hazard_rates_checked':1280}
        print(kind,audit[kind],flush=True)
    (OUT/'dynamics_audit.json').write_text(json.dumps(audit,indent=2))
if __name__=='__main__':main()
