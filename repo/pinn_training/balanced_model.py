"""Continuous time-decomposed surrogate with nonnegative material accounting.
The two conserved inventories are antibody and payload, including clearance.
This representation enforces accounting, not the differential equations.
"""
import numpy as np
import torch
from torch import nn
from physics import Physics,tensor
from data import TIMES
from train import TrajectoryNet

KNOTS=np.array([0,1e-6,1e-4,.001,.01,.1,.5,1,2,4,8,16,24.])

class BalancedNet(nn.Module):
    def __init__(self,x,y):
        super().__init__();self.physics=Physics()
        self.register_buffer('mean',tensor(x.mean(0)));self.register_buffer('std',tensor(x.std(0)))
        self.register_buffer('scale',tensor(np.maximum(abs(y).max((0,1,2)),1e-8)))
        self.register_buffer('knots',tensor(KNOTS))
        self.register_buffer('knot_coord',tensor(2*np.log1p(KNOTS/1e-6)/np.log1p(24/1e-6)-1))
        def net(n):return nn.Sequential(nn.Linear(n,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,97))
        self.nodes=net(8);self.local=net(8+len(KNOTS)-1)
        for a in [self.nodes,self.local]:nn.init.normal_(a[-1].weight,std=.001);nn.init.zeros_(a[-1].bias)
        ab=[0,1,2,3,5];intact=[0,1];payload=[4,6];receptor=[];cumul=[]
        for j in range(5):
            base=7+16*j;ab+=list(range(base+2,base+14));intact+=list(range(base+2,base+5))+list(range(base+8,base+11));payload.append(base+14);receptor+=list(range(base,base+2));cumul.append(base+15)
        cumul+=list(range(87,97))
        self.ab=ab;self.intact=intact;self.payload=payload;self.receptor=receptor;self.cumul=cumul
        self.register_buffer('abmap',torch.eye(97)[ab]);self.register_buffer('pmap',torch.eye(97)[payload]);self.register_buffer('rmap',torch.eye(97)[receptor]);self.register_buffer('cumap',torch.eye(97)[cumul])
        self.register_buffer('intact_mask',tensor([float(i in intact) for i in ab]))
        coeff=np.ones((3,97));mask=np.ones((3,97),dtype=bool)
        for c in range(3):
            coeff[c,[0,2]]=float(self.physics.vc[c]);coeff[c,[1,3]]=float(self.physics.vp[c]);coeff[c,4]=float(self.physics.pv[c])
            for j in range(5):
                base=7+16*j;coeff[c,base+2:base+14]=float(self.physics.cap[c,j]);coeff[c,base+14]=float(self.physics.vol[c,j])
                present=self.physics.rho[c,j].numpy()>0
                allowed=np.array([present[0],present[1],present.all()])
                mask[c,base+2:base+14]=np.tile(allowed,4)
        self.register_buffer('coeff',tensor(coeff));self.register_buffer('allowed',torch.as_tensor(mask))
        # Amount-scale offsets avoid assigning similar probability to compartments
        # whose natural capacities differ by many orders of magnitude.
        self.register_buffer('log_amount_prior',torch.log(self.coeff*self.scale))
    def forward(self,t,x,c):
        b=len(t);context=torch.cat([(x-self.mean)/self.std,torch.nn.functional.one_hot(c,3).to(torch.float64)],-1)
        inputs=torch.cat([self.knot_coord[None,:,None].expand(b,-1,-1),context[:,None,:].expand(-1,len(KNOTS),-1)],-1)
        nodes=self.nodes(inputs)
        seg=torch.searchsorted(self.knots,t.contiguous(),right=True)-1;seg=seg.clamp(0,len(KNOTS)-2)
        dt=self.knots[seg+1]-self.knots[seg];u=(t-self.knots[seg])/dt
        idx=torch.arange(b);raw=(1-u[:,None])*nodes[idx,seg]+u[:,None]*nodes[idx,seg+1]
        loc=torch.cat([u[:,None],context,torch.nn.functional.one_hot(seg,len(KNOTS)-1).to(torch.float64)],-1)
        raw=raw+4*u[:,None]*(1-u[:,None])*self.local(loc)
        # First allocate all administered antibody, including antibody lost.
        dose=self.physics.dose[c]*torch.minimum(t/self.physics.infusion[c],torch.ones_like(t))
        logits=(raw+self.log_amount_prior[c])[:,self.ab]
        logits=logits.masked_fill(~self.allowed[c][:,self.ab],-torch.inf)
        fractions=torch.softmax(logits,-1);abamount=dose[:,None]*fractions
        abstate=abamount/self.coeff[c][:,self.ab]
        # Payload still attached to living intact antibody is shared inventory.
        # All remaining payload is explicitly free intracellular/extracellular or lost.
        free=dose*(fractions*(1-self.intact_mask)).sum(-1)
        pf=torch.softmax((raw+self.log_amount_prior[c])[:,self.payload],-1)
        pstate=free[:,None]*pf/self.coeff[c][:,self.payload]
        initial=self.physics.initial(c)[:,self.receptor]
        g=t/(t+1e-4)
        receptor=initial*((1-g[:,None])+g[:,None]*torch.nn.functional.softplus(raw[:,self.receptor])/np.log(2))
        # Positive piecewise-linear rates integrate to nondecreasing cumulative states.
        rates=torch.nn.functional.softplus(nodes[:,:,self.cumul])*self.scale[self.cumul]/24
        widths=self.knots[1:]-self.knots[:-1]
        elapsed=(t[:,None]-self.knots[:-1]).clamp_min(0)
        q=torch.minimum(elapsed/widths,torch.ones_like(elapsed))
        cumulative=(widths[None,:,None]*(rates[:,:-1]*q[:,:,None]+.5*(rates[:,1:]-rates[:,:-1])*q[:,:,None]**2)).sum(1)
        return abstate@self.abmap+pstate@self.pmap+receptor@self.rmap+cumulative@self.cumap
    prediction=TrajectoryNet.prediction
    def residual(self,t,x,c):
        y,dy=torch.func.jvp(lambda tt:self(tt,x,c),(t,),(torch.ones_like(t),))
        seg=(torch.searchsorted(self.knots,t.contiguous(),right=True)-1).clamp(0,len(KNOTS)-2)
        dt=self.knots[seg+1]-self.knots[seg]
        return y,(dy-self.physics.rhs(y,x,c,(t<2).to(torch.float64)))*dt[:,None]/self.scale
