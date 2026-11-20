"""Differentiable copy of the fixed three-condition reference equations.
No biological approximation; all states retain the reference numerical units.
"""
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mechanism_comparison.model import INPUT_SET, density

def tensor(x): return torch.as_tensor(x,dtype=torch.float64)

class Physics:
    def __init__(self):
        ss=[s for _,s,_ in INPUT_SET]; ps=[p for _,_,p in INPUT_SET]
        def arr(fn):return tensor(np.array([fn(s) for s in ss]))
        self.rho=arr(density)
        self.cap=arr(lambda s:s.total_cells*np.array(s.fractions)*s.cell_area*1e-4)
        self.vol=arr(lambda s:s.total_cells*np.array(s.fractions)*s.intracellular_volume)
        self.gain=arr(lambda s:s.cell_area*1e-4/s.intracellular_volume)
        for name,fn in {'ki':lambda s:s.ki,'k0':lambda s:s.k0,'kr':lambda s:s.recycling_rates,'kp':lambda s:s.processing_rates,'kg':lambda s:s.degradation_rates,'kd':lambda s:np.log(2)/s.half,'perm':lambda s:s.permeability,'met':lambda s:s.metabolism,'pv':lambda s:s.payload_volume,'pc':lambda s:s.payload_clearance,'vc':lambda s:s.antibody_vc,'vp':lambda s:s.antibody_vp,'qa':lambda s:s.antibody_q,'cl':lambda s:s.antibody_clearance,'dose':lambda s:s.dose,'infusion':lambda s:s.infusion_h,'l':lambda s:2.4e13*1e-10*3600*s.surface_factor}.items():setattr(self,name,arr(fn))
        self.c50=tensor([p.c50_uM for p in ps]);self.kmax=tensor([p.maximum_rate_h for p in ps])
    def initial(self,c):
        y=torch.zeros((len(c),97),dtype=torch.float64)
        z=y[:,7:87].reshape(-1,5,16);z[:,:,:2]=self.rho[c]
        return y
    def rhs(self,y,theta,c,on):
        a1,b1,a2,b2=(10**theta*3600).unbind(-1)
        corr=torch.sqrt(a2/a1);q1=b1*corr;q2=b2/corr
        z=y[:,7:87].reshape(-1,5,16);R=z[:,:,:2];S=z[:,:,2:8].reshape(-1,5,2,3);E=z[:,:,8:14].reshape(-1,5,2,3)
        ki=self.ki[c,None,:];kr=self.kr[c,None,:];kp=self.kp[c,None,:];kg=self.kg[c,None,:]
        dr=self.k0[c,None,:]*(self.rho[c]-R);ds=[];outs=[]
        for x in range(2):
            b=S[:,:,x,:];cc=y[:,2*x,None]
            v1=a1[:,None]*cc*1e-6*R[:,:,0]-b1[:,None]*b[:,:,0]
            v2=a2[:,None]*cc*1e-6*R[:,:,1]-b2[:,None]*b[:,:,1]
            w1=self.l[c,None]*b[:,:,1]*R[:,:,0]-q1[:,None]*b[:,:,2]
            w2=self.l[c,None]*b[:,:,0]*R[:,:,1]-q2[:,None]*b[:,:,2]
            dr=dr-torch.stack([v1+w1,v2+w2],-1)
            ds.append(torch.stack([v1-w2,v2-w1,w1+w2],-1)-ki*b)
            flow=self.qa[c]*(y[:,2*x]-y[:,2*x+1])
            outs.extend([(-self.cl[c]*y[:,2*x]-flow-(self.cap[c]*(v1+v2)).sum(-1))/self.vc[c],flow/self.vp[c]])
        kd=self.kd[c];shed=kd[:,None,None]*S[:,:,0,:]
        ds0=ds[0]-shed+kr*E[:,:,0,:];ds1=ds[1]+shed+kr*E[:,:,1,:]
        de0=ki*S[:,:,0,:]-(kr+kp+kg)*E[:,:,0,:]
        de1=ki*S[:,:,1,:]+kp*E[:,:,0,:]-(kr+kg)*E[:,:,1,:]
        release=((kp+kg)*E[:,:,0,:]).sum(-1)
        abloss=(kg*(E[:,:,0,:]+E[:,:,1,:])).sum(-1)
        production=self.gain[c,None]*release
        exchange=self.perm[c,None]*(y[:,4,None]-z[:,:,14])
        dp=production+exchange-self.met[c,None]*z[:,:,14]
        generated=kd*(self.vc[c]*y[:,0]+self.vp[c]*y[:,1])+(self.cap[c]*shed.sum(-1)).sum(-1)
        out4=(generated-self.pc[c]*y[:,4]-(self.vol[c]*exchange).sum(-1))/self.pv[c]
        out5=self.cl[c]*(y[:,0]+y[:,2])+(self.cap[c]*abloss).sum(-1)
        out6=self.cl[c]*y[:,0]+self.pc[c]*y[:,4]+self.met[c]*(self.vol[c]*z[:,:,14]).sum(-1)
        dz=torch.cat([dr,ds0,ds1,de0,de1,dp[:,:,None],production[:,:,None]],-1)
        out=torch.stack([outs[0]-kd*y[:,0]+on*self.dose[c]/self.infusion[c]/self.vc[c],outs[1]-kd*y[:,1],outs[2]+kd*y[:,0],outs[3]+kd*y[:,1],out4,out5,out6],-1)
        hazard=self.kmax[c]*z[:,:,14]/(self.c50[c]+z[:,:,14])
        return torch.cat([out,dz.reshape(-1,80),hazard,z[:,:,14]],-1)
    def balance(self,y,t,c):
        z=y[:,7:87].reshape(-1,5,16)
        ab=self.vc[c]*(y[:,0]+y[:,2])+self.vp[c]*(y[:,1]+y[:,3])+(z[:,:,2:14].sum(-1)*self.cap[c]).sum(-1)+y[:,5]
        payload=self.vc[c]*y[:,0]+self.vp[c]*y[:,1]+((z[:,:,2:5]+z[:,:,8:11]).sum(-1)*self.cap[c]).sum(-1)+self.pv[c]*y[:,4]+(z[:,:,14]*self.vol[c]).sum(-1)+y[:,6]
        dose=self.dose[c]*torch.minimum(t/self.infusion[c],torch.ones_like(t))
        return torch.stack([ab-dose,payload-dose],-1)/self.dose[c,None]
