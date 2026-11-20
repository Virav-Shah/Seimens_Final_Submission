"""Post-run reproducibility, shape, derivative and checkpoint integrity checks."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from data import OUT,TIMES
from train import TrajectoryNet,tensor,physical_metrics,sample_collocation

def main():
    cfg=json.loads((OUT/'configuration.json').read_text())
    ref=Path(__file__).resolve().parents[1]/'mechanism_comparison/model.py'
    assert hashlib.sha256(ref.read_bytes()).hexdigest()==cfg['reference_sha256']
    sets=[]
    for name,n in [('initial_2026',64),('validation',32),('test',64)]:
        d=np.load(OUT/f'{name}.npz');assert d['y'].shape==(n,3,129,97)
        assert np.isfinite(d['y']).all();assert len(np.unique(d['t']))==129
        sets.append(set(map(tuple,d['x'])))
    assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
    d=np.load(OUT/'initial_2026.npz');model=TrajectoryNet(d['x'],d['y'])
    saved=torch.load(OUT/'pilot_checkpoint.pt',weights_only=False);model.load_state_dict(saved['state_dict'])
    assert saved['selected']['physics_weight']==1
    x=tensor(d['x'][:3]);c=torch.arange(3)
    torch.testing.assert_close(model(torch.zeros(3),x,c),model.physics.initial(c),rtol=0,atol=0)
    # Algebraic bridge endpoint equals output at end of infusion.
    inp=torch.cat([(x-model.mean)/model.std,torch.eye(3)],-1)
    bridge=model.physics.initial(c)+model.scale*model.first(torch.cat([torch.ones((3,1)),inp],-1))
    torch.testing.assert_close(model(torch.full((3,),2.),x,c),bridge,rtol=0,atol=0)
    t=tensor([.1,1.8,12.]);_,res=model.residual(t,x,c)
    model.zero_grad();res.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    predfile=np.load(OUT/'pilot_test_predictions.npz');loaded=model.prediction(predfile['x'])
    np.testing.assert_allclose(loaded,predfile['prediction'],rtol=1e-12,atol=1e-12)
    # Expose which physical quantities produced largest validation failures.
    v=np.load(OUT/'pilot_validation_predictions.npz');scaled=v['prediction']/model.scale.numpy()
    ix=np.unravel_index(np.argmin(scaled),scaled.shape)
    names=['intact ADC central','intact ADC peripheral','payload-free antibody central','payload-free antibody peripheral','extracellular payload','antibody lost','payload lost']
    for group in ['tumour','healthy neither','healthy T1','healthy T2','healthy both']:
        for state in ['free T1','free T2','intact B1','intact B2','intact D','payload-free B1','payload-free B2','payload-free D','internalised intact B1','internalised intact B2','internalised intact D','internalised payload-free B1','internalised payload-free B2','internalised payload-free D','intracellular payload','cumulative targeted release']:names.append(group+': '+state)
    names += [g+': accumulated killing effect' for g in ['tumour','healthy neither','healthy T1','healthy T2','healthy both']]
    names += [g+': payload exposure integral' for g in ['tumour','healthy neither','healthy T1','healthy T2','healthy both']]
    assert len(names)==97
    diagnostic={'worst_negative':{'validation_design':int(ix[0]),'condition':int(ix[1]),'time_h':float(TIMES[ix[2]]),'state':names[ix[3]],'state_index':int(ix[3]),'value':float(v['prediction'][ix]),'scale':float(model.scale[ix[3]]),'scaled_value':float(scaled[ix])},'state_names':names}
    rng=np.random.default_rng(2026)
    ti=torch.as_tensor(rng.integers(0,129,256));di=torch.as_tensor(rng.integers(0,64,256));ci=torch.as_tensor(rng.integers(0,3,256))
    predictions=model(tensor(TIMES)[ti],tensor(d['x'])[di],ci)
    data_loss=((predictions-tensor(d['y'])[di,ci,ti])/model.scale).square().mean()
    ct,cx,cc=sample_collocation(128,rng);py,rr=model.residual(ct,cx,cc)
    losses={'trajectory':data_loss,'equations':rr.square().mean(),'balance':model.physics.balance(py,ct,cc).square().mean()}
    gradients={}
    for label,loss in losses.items():
        grads=torch.autograd.grad(loss,tuple(model.parameters()),retain_graph=True,allow_unused=True)
        norm=torch.sqrt(sum(g.square().sum() for g in grads if g is not None))
        gradients[label]={'loss':float(loss.detach()),'parameter_gradient_norm_before_clipping':float(norm)}
    diagnostic['checkpoint_loss_diagnostics']=gradients
    (OUT/'diagnostics.json').write_text(json.dumps(diagnostic,indent=2))
    result={'reference_unchanged':True,'datasets_disjoint':True,'97_states_129_times':True,'initial_conditions_exact':True,'bridge_continuity_exact':True,'residual_gradients_finite':True,'saved_checkpoint_reproduces_predictions':True,'checkpoint_full_physics_weight':True}
    (OUT/'artifact_checks.json').write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':main()
