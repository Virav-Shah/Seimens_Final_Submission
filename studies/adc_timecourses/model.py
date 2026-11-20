from dataclasses import dataclass, replace, asdict
from pathlib import Path
import itertools, json, random, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from platypus import Problem, Real, Solution, InjectedPopulation, NSGAII
from IPython.display import display, Markdown

# Hours; solution and intracellular concentrations in uM; amounts in umol;
# surface/endosomal states in units of 1e-10 mol/dm2.
F = np.array([.0003, .79, .10, .10, .01])
NAMES = ['T', 'H0', 'H1', 'H2', 'H12']
NCELL = 7043996587630.85
VI = 2.689720494498626e-13
AREA = 3.141592653589793e-8
CAP = NCELL * F * AREA * 1e-4  # umol per scaled surface unit
CELLVOL = NCELL * F * VI     # L
GAIN = AREA * 1e-4 / VI      # uM per scaled surface unit
VC, VP, QA = 11.3, 41.5, 1.05
CLA = 0.015540799881601949
BOUNDS = [(2.,7.),(-5.,2.),(2.,7.),(-5.,2.)]
LIMIT = .0095
RATES = ['kon1','koff1','kon2','koff2']

@dataclass(frozen=True)
class Scenario:
    mode: str = 'bispecific'
    rho: tuple = (1., 1.)  # multiples of 1e-10 mol/dm2
    k0: tuple = (2., 2.)
    ki: tuple = (2., 2., 2.)
    half: float = np.inf
    permeability: float = 0.  # PS / VI, h^-1; bidirectional, K=fu=1
    payload_volume: float = 52.8  # one WELL-MIXED extracellular pool, exploratory
    payload_clearance: float = 5.28  # L/h; clearance rate 0.1 h^-1
    metabolism: float = 0.
    tau: float = 0.  # 0: instantaneous internal degradation and release
    recycle: float = 0.
    process_share: float = .5  # remaining exit split between processing/degradation
    closure: str = 'symmetric'
    surface_factor: float = 1.
    dose: float = 4.  # umol
    target_densities: tuple = ()  # optional 5 x 2 matrix, multiples of 1e-10 mol/dm2
    fractions: tuple = (.0003,.79,.10,.10,.01)
    total_cells: float = NCELL
    intracellular_volume: float = VI
    cell_area: float = AREA
    antibody_vc: float = VC
    antibody_vp: float = VP
    antibody_q: float = QA
    antibody_clearance: float = CLA
    infusion_h: float = 2.
    horizon_h: float = 24.
    recycling_rates: tuple = ()
    processing_rates: tuple = ()
    degradation_rates: tuple = ()


def density(s):
    if s.target_densities:
        return np.asarray(s.target_densities, dtype=float)
    return np.array([[1,1],[0,0],[.25,0],[0,.25],[.25,.25]]) * s.rho


def make_rhs(theta, s, on):
    CAP=s.total_cells*np.array(s.fractions)*s.cell_area*1e-4
    CELLVOL=s.total_cells*np.array(s.fractions)*s.intracellular_volume
    GAIN=s.cell_area*1e-4/s.intracellular_volume
    VC,VP,QA,CLA=s.antibody_vc,s.antibody_vp,s.antibody_q,s.antibody_clearance
    a1,b1,a2,b2 = 10.**np.asarray(theta) * 3600
    l = 2.4e13 * 1e-10 * 3600 * s.surface_factor
    corr = np.sqrt(a2/a1) if s.closure == 'symmetric' else 1.
    q1,q2 = b1*corr,b2/corr
    kd = np.log(2)/s.half
    rho,k0,ki = density(s), np.array(s.k0), np.array(s.ki)
    kr = s.recycle/s.tau if s.tau else 0.
    kp = (1-s.recycle)*s.process_share/s.tau if s.tau else 0.
    kg = (1-s.recycle)*(1-s.process_share)/s.tau if s.tau else 0.
    if s.recycling_rates:
        kr,kp,kg=map(np.asarray,(s.recycling_rates,s.processing_rates,s.degradation_rates))
    def fun(t,y):
        # 0:4 Ic,Ip,Nc,Np; 4 extracellular payload; 5 Ab lost; 6 payload lost.
        # per cell: R(2), intact S(3), naked S(3), intact E(3), naked E(3),
        # P instantaneous, A cumulative productive targeted release (both uM).
        z=y[7:].reshape(5,16); dz=np.zeros_like(z); out=np.zeros_like(y)
        R=z[:,:2]; S=z[:,2:8].reshape(5,2,3); E=z[:,8:14].reshape(5,2,3)
        dS=dz[:,2:8].reshape(5,2,3); dE=dz[:,8:14].reshape(5,2,3)
        dz[:,:2]=k0*(rho-R)
        for x,c in enumerate([y[0],y[2]]):
            b=S[:,x,:]
            v1=a1*c*1e-6*R[:,0]-b1*b[:,0]
            v2=a2*c*1e-6*R[:,1]-b2*b[:,1]
            w1=l*b[:,1]*R[:,0]-q1*b[:,2]
            w2=l*b[:,0]*R[:,1]-q2*b[:,2]
            if s.mode == 'monospecific':
                # Bivalent binding to T1, following the original reconstruction.
                # State 2 is T1-ADC-T1, NOT a T1T2 complex.
                v1=2*a1*c*1e-6*R[:,0]-b1*b[:,0]
                v2=np.zeros(5)
                w1=l*b[:,0]*R[:,0]-2*b1*b[:,2]
                dz[:,0]-=v1+w1
                dS[:,x,:]=np.column_stack([v1-w1,np.zeros(5),w1])-ki*b
            else:
                dz[:,0]-=v1+w1; dz[:,1]-=v2+w2
                dS[:,x,:]=np.column_stack([v1-w2,v2-w1,w1+w2])-ki*b
            cix=2*x; pix=cix+1
            flow=QA*(y[cix]-y[pix])
            out[cix]=(-CLA*y[cix]-flow-CAP@(v1+v2))/VC
            out[pix]=flow/VP
        shed=kd*S[:,0,:]
        dS[:,0,:]-=shed; dS[:,1,:]+=shed
        out[0]-=kd*y[0]; out[1]-=kd*y[1]
        out[2]+=kd*y[0]; out[3]+=kd*y[1]
        out[0]+=(s.dose/s.infusion_h/VC if on else 0.)
        if s.tau or s.recycling_rates:
            dS[:]+=kr*E
            dE[:,0,:]=ki*S[:,0,:]-(kr+kp+kg)*E[:,0,:]
            dE[:,1,:]=ki*S[:,1,:]+kp*E[:,0,:]-(kr+kg)*E[:,1,:]
            release=((kp+kg)*E[:,0,:]).sum(axis=1)
            ab_loss=(kg*E.sum(axis=1)).sum(axis=1)
        else:
            release=(ki*S[:,0,:]).sum(axis=1)
            ab_loss=(ki*S.sum(axis=1)).sum(axis=1)
        production=GAIN*release
        exchange=s.permeability*(y[4]-z[:,14])
        dz[:,14]=production+exchange-s.metabolism*z[:,14]
        dz[:,15]=production
        # Both antibody compartments release to the explicitly pooled extracellular
        # payload volume; this is a lumping assumption, not central-only transport.
        generated=kd*(VC*y[0]+VP*y[1])+CAP@shed.sum(axis=1)
        out[4]=(generated-s.payload_clearance*y[4]-CELLVOL@exchange)/s.payload_volume
        out[5]=CLA*(y[0]+y[2])+CAP@ab_loss
        out[6]=CLA*y[0]+s.payload_clearance*y[4]+s.metabolism*(CELLVOL@z[:,14])
        out[7:]=dz.ravel()
        return out
    return fun



@dataclass(frozen=True)
class KillModel:
    c50_uM: tuple = (4.,4.,4.,4.,4.)
    maximum_rate_h: tuple = (.08,.08,.08,.08,.08)

# Explicit rates replace tau/branching shortcuts in the combined model.
BIOLOGY=Scenario(ki=(.2,4.,2.), half=96., permeability=.1, metabolism=.01,
                 recycling_rates=(.6,.6,.6), processing_rates=(.7,.7,.7),
                 degradation_rates=(.7,.7,.7))
PD=KillModel()
INPUT_SET=[('Nominal',BIOLOGY,PD),
 ('Variant A',replace(BIOLOGY,ki=(.16,3.2,1.6),half=80.,permeability=.12),
  replace(PD,c50_uM=(4.8,3.2,3.2,3.2,3.2))),
 ('Variant B',replace(BIOLOGY,ki=(.24,4.8,2.4),half=120.,permeability=.08),
  replace(PD,c50_uM=(3.2,4.8,4.8,4.8,4.8)))]
GLOBAL_LIMIT=.0095
GROUP_LIMITS=np.repeat(.10,4)
SEEDS=(8407,2026)
GLOBAL_EVALUATIONS=256
LOCAL_ITERATIONS=40


def check_inputs(s,p):
    assert s.closure == "symmetric", "This method retains V1 symmetric surface closure"
    assert np.asarray(density(s)).shape==(5,2) and np.all(density(s)>=0)
    assert len(s.fractions)==5 and np.all(np.array(s.fractions)>=0)
    assert np.isclose(sum(s.fractions[1:]),1.)
    assert 0<s.infusion_h<s.horizon_h and s.dose>=0
    assert min(s.total_cells,s.intracellular_volume,s.cell_area,s.antibody_vc,s.antibody_vp,s.payload_volume)>0
    assert min(s.antibody_q,s.antibody_clearance,s.payload_clearance,s.permeability,s.metabolism)>=0
    assert s.half>0 and np.all(np.array(s.ki)>=0) and np.all(np.array(s.k0)>=0)
    for a in (s.recycling_rates,s.processing_rates,s.degradation_rates):
        assert not a or (len(a)==3 and np.all(np.array(a)>=0))
    assert all(bool(a)==bool(s.recycling_rates) for a in (s.processing_rates,s.degradation_rates))
    assert len(p.c50_uM)==len(p.maximum_rate_h)==5
    assert np.all(np.array(p.c50_uM)>0) and np.all(np.array(p.maximum_rate_h)>=0)

def integrate(theta,s=BIOLOGY,p=PD,tight=False,points=3,method='BDF'):
    check_inputs(s,p)
    y0=np.zeros(97)  # 87 material states + 5 cumulative hazards + 5 AUCs
    y0[7:87].reshape(5,16)[:,:2]=density(s)
    tt=[]; yy=[]
    times=np.unique(np.r_[np.linspace(0,s.horizon_h,points),s.infusion_h])
    for start,end,on in [(0,s.infusion_h,True),(s.infusion_h,s.horizon_h,False)]:
        material=make_rhs(theta,s,on)
        def rhs(t,y):
            d=np.zeros_like(y);d[:87]=material(t,y[:87])
            conc=y[7:87].reshape(5,16)[:,14]
            d[87:92]=np.array(p.maximum_rate_h)*conc/(np.array(p.c50_uM)+conc)
            d[92:97]=conc
            return d
        sol=solve_ivp(rhs,(start,end),y0,method=method,rtol=1e-8 if tight else 2e-6,
                      atol=1e-11 if tight else 2e-9,dense_output=True)
        if not sol.success:raise RuntimeError(sol.message)
        ts=times[(times>=start)&(times<=end)] if start==0 else times[(times>start)&(times<=end)]
        tt.extend(ts);yy.extend(sol.sol(ts).T);y0=sol.y[:,-1]
    y=np.array(yy);t=np.array(tt);z=y[:,7:87].reshape(-1,5,16)
    if y.min() < -2e-6:
        if not tight:return integrate(theta,s,p,True,points,method)
        if method=='BDF':return integrate(theta,s,p,True,points,'Radau')
        raise RuntimeError('Unresolved negative state')
    kill=-np.expm1(-y[:,87:92])
    cap=s.total_cells*np.array(s.fractions)*s.cell_area*1e-4
    vols=s.total_cells*np.array(s.fractions)*s.intracellular_volume
    ab=s.antibody_vc*(y[:,0]+y[:,2])+s.antibody_vp*(y[:,1]+y[:,3])+z[:,:,2:14].sum(axis=2)@cap+y[:,5]
    payload=s.antibody_vc*y[:,0]+s.antibody_vp*y[:,1]+(z[:,:,2:5]+z[:,:,8:11]).sum(axis=2)@cap+s.payload_volume*y[:,4]+z[:,:,14]@vols+y[:,6]
    dose=s.dose*np.minimum(t/s.infusion_h,1.)
    mass=max(abs(ab-dose).max(),abs(payload-dose).max())/max(s.dose,1.)
    if mass>2e-6:raise RuntimeError(f'Mass error exceeds tolerance: {mass}')
    return dict(t=t,y=y,z=z,kill=kill,healthy=kill[:,1:]@np.array(s.fractions[1:]),mass=mass)

def assess(theta,tight=False):
    rows=[]
    for name,s,p in INPUT_SET:
        r=integrate(theta,s,p,tight=tight); k=r['kill'][-1]
        rows.append([k[0],r['healthy'][-1],*k[1:]])
    a=np.array(rows)
    # Every constraint applies to every listed input variant, not an average variant.
    margins=np.r_[GLOBAL_LIMIT-a[:,1],(GROUP_LIMITS-a[:,2:]).ravel()]
    return dict(tumour=100*a[:,0].min(),healthy=100*a[:,1].max(),
                margins=100*margins,outcomes=100*a)
