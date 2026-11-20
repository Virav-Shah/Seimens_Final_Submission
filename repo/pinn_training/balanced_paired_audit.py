"""Compare both predictors on the same union of ODE-audited candidate designs."""
import json
import numpy as np
import torch
from balanced_trial import OUT,OLD
from balanced_model import BalancedNet
from train import endpoints,accuracy

def main():
    p=OUT/'candidate_audit.json';r=json.loads(p.read_text())
    if r['status']!='completed':return
    unique={}
    for method in r['methods'].values():
        for row in method['candidates']:unique[tuple(row['x'])]=row['actual']
    x=np.array(list(unique));actual=np.array(list(unique.values()));d=np.load(OLD/'initial_2026.npz')
    paired={'n':len(x),'note':'Shared union selected by both methods; not an unbiased global test sample','x':x.tolist(),'actual':actual.tolist(),'methods':{}}
    for name in ['PINN','Supervised']:
        m=BalancedNet(d['x'],d['y']);m.load_state_dict(torch.load(OUT/f'{name}_checkpoint.pt',weights_only=False)['state_dict'])
        pred=endpoints(m.prediction(x,np.array([24.]))[:,:,0])
        paired['methods'][name]={'metrics':accuracy(actual,pred),'prediction':pred.tolist()}
    r['paired_union']=paired;p.write_text(json.dumps(r,indent=2));print({k:v['metrics'] for k,v in paired['methods'].items()})
if __name__=='__main__':main()
