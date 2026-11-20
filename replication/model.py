"""Auditable reconstruction, NOT recovered Stoddard source code.
Units: physical rates in seconds, M, mol/dm²; solver states scaled for conditioning.
See output/model_analysis.tex for equations and provenance.
"""
from dataclasses import dataclass, replace, asdict
import numpy as np
from scipy.integrate import solve_ivp

@dataclass(frozen=True)
class Parameters:
    kon1: float = 4.6e5
    koff1: float = 3.4e-4
    kon2: float = 4.6e5
    koff2: float = 3.4e-4
    rho1: float = 1e-10
    rho2: float = 1e-10
    k2d1: float = 2.4e13  # binding T1 after T2; Sengers-derived
    k2d2: float = 2.4e13  # binding T2 after T1
    turnover: float = 2/3600  # interpretation of "twice-hourly"
    dual_turnover_factor: float = 1.0  # unpublished closure
    ec50: float = 4e-6
    dose: float = 4e-6
    infusion_s: float = 7200
    end_s: float = 86400  # inferred from figure traces
    vc: float = 11.3  # source-reference value, adoption unconfirmed
    vp: float = 41.5
    q_l_h: float = 1.05
    cl_l_h: float = 0.1  # DEMONSTRATION ONLY; not reported
    total_cells: float = 3e13  # DEMONSTRATION ONLY; not reported
    area_dm2: float = 5e-8  # Table S1 conversion assumption, adoption unconfirmed
    cell_volume_l: float = 1.0513052175e-12  # sphere from 500 um²; reconstruction
    surface_off_rule: str = 'sengers'  # or 'same_bond'; adoption unconfirmed
    mode: str = 'bispecific'

NAMES = ('tumor','healthy_T1','healthy_T2','healthy_T12')
FREQ = np.array([0.0003,0.1,0.1,0.01])
HEALTHY_WEIGHTS = np.array([0.1,0.1,0.01])  # no-target remainder 0.79
SCALE_DENSITY = 1e-10
SCALE_CONC = 1e-6

def densities(p):
    return np.array([[p.rho1,p.rho2],[.25*p.rho1,0],[0,.25*p.rho2],[.25*p.rho1,.25*p.rho2]])

def initial(p):
    # Cc,Cp (uM), eliminated (umol); then R1,R2,B1,B2,D,I for each population
    y = np.zeros(27)
    y[3:].reshape(4,6)[:,:2] = densities(p)/SCALE_DENSITY
    if p.mode == 'monospecific': y[3:].reshape(4,6)[:,1] = 0
    return y

def rhs(t,y,p,infusing):
    cc, cp = y[:2]*SCALE_CONC
    z = y[3:].reshape(4,6)
    r1,r2,b1,b2,d = (z[:,i]*SCALE_DENSITY for i in range(5))
    out = np.zeros_like(y); dz = out[3:].reshape(4,6)
    k = p.turnover; kd = k*p.dual_turnover_factor
    rho = densities(p)
    if p.mode == 'bispecific':
        # q1 loses T1 from D -> B2; q2 loses T2 from D -> B1.
        q2 = p.koff2
        q1 = p.koff1 if p.surface_off_rule == 'same_bond' else p.koff1*p.kon2*p.k2d2/(p.kon1*p.k2d1)
        v1 = p.kon1*cc*r1-p.koff1*b1
        v2 = p.kon2*cc*r2-p.koff2*b2
        w1 = p.k2d1*b2*r1-q1*d
        w2 = p.k2d2*b1*r2-q2*d
        dz[:,0] = k*(rho[:,0]-r1)-v1-w1
        dz[:,1] = k*(rho[:,1]-r2)-v2-w2
        dz[:,2] = v1-w2-k*b1
        dz[:,3] = v2-w1-k*b2
        dz[:,4] = w1+w2-kd*d
        uptake = k*(b1+b2)+kd*d
        exchange = v1+v2
    else:
        # B1 = singly-bound antibody; D = doubly-bound, identical-arm combinatorics.
        v1 = 2*p.kon1*cc*r1-p.koff1*b1
        w1 = p.k2d1*b1*r1-2*p.koff1*d
        dz[:,0] = k*(rho[:,0]-r1)-v1-w1
        dz[:,2] = v1-w1-k*b1
        dz[:,4] = w1-kd*d
        uptake = k*b1+kd*d
        exchange = v1
    dz[:,:5] /= SCALE_DENSITY
    dz[:,5] = p.area_dm2/p.cell_volume_l*uptake/SCALE_CONC
    # Table 1 frequencies are used literally; sum including tumor is 1.0003.
    surface_area = p.total_cells*FREQ*p.area_dm2
    elim = p.cl_l_h/3600*cc
    distribution = p.q_l_h/3600*(cc-cp)
    dose_rate = p.dose/p.infusion_s if infusing else 0.
    out[0] = (dose_rate-elim-distribution-np.dot(surface_area,exchange))/p.vc/SCALE_CONC
    out[1] = distribution/p.vp/SCALE_CONC
    out[2] = elim/SCALE_CONC
    return out

def simulate(p=Parameters(), points=121, rtol=1e-7, atol=1e-10, method="BDF"):
    if p.surface_off_rule not in ('sengers','same_bond'): raise ValueError('invalid off rule')
    if p.mode not in ('bispecific','monospecific'): raise ValueError('invalid mode')
    if not 0 < p.infusion_s < p.end_s: raise ValueError('require 0 < infusion < horizon')
    y0 = initial(p)
    times=np.unique(np.r_[np.linspace(0,p.end_s,points),p.infusion_s])
    ts=[]; ys=[]
    for a,b,on in [(0,p.infusion_s,True),(p.infusion_s,p.end_s,False)]:
        sol=solve_ivp(lambda t,y:rhs(t,y,p,on),(a,b),y0,method=method,rtol=rtol,atol=atol,dense_output=True)
        if not sol.success: raise RuntimeError(sol.message)
        sel=times[(times>=a)&(times<=b)] if a==0 else times[(times>a)&(times<=b)]
        ts.extend(sel); ys.extend(sol.sol(sel).T); y0=sol.y[:,-1]
    y=np.asarray(ys); t=np.asarray(ts); z=y[:,3:].reshape(-1,4,6)
    # Never clip inside equations; check solver excursions explicitly.
    if y.min() < -1e-6: raise RuntimeError(f'negative scaled state {y.min()}')
    intracellular=np.maximum(z[:,:,5]*SCALE_CONC,0)
    kills=intracellular/(p.ec50+intracellular)
    healthy=kills[:,1:]@HEALTHY_WEIGHTS
    tumor=kills[:,0]
    rho=densities(p)
    if p.mode=='bispecific':
        bound=np.stack((z[:,:,2]+z[:,:,4],z[:,:,3]+z[:,:,4]),axis=-1)*SCALE_DENSITY
    else:
        bound=np.stack((z[:,:,2]+2*z[:,:,4],np.zeros_like(z[:,:,4])),axis=-1)*SCALE_DENSITY
        rho[:,1]=0
    occ=np.divide(bound,rho[None,:,:],out=np.zeros_like(bound),where=rho[None,:,:]>0)
    surface=(z[:,:,2]+z[:,:,3]+z[:,:,4])*SCALE_DENSITY
    m_surface=surface@(p.total_cells*FREQ*p.area_dm2)
    m_internal=intracellular@(p.total_cells*FREQ*p.cell_volume_l)
    total=y[:,0]*SCALE_CONC*p.vc+y[:,1]*SCALE_CONC*p.vp+y[:,2]*SCALE_CONC+m_surface+m_internal
    administered=p.dose*np.minimum(t/p.infusion_s,1)
    return dict(t=t,y=y,occupancy=occ,intracellular=intracellular,tumor=tumor,healthy=healthy,
                mass_error=total-administered, parameters=asdict(p))

def endpoints(p):
    r=simulate(p,points=3)
    return float(r['tumor'][-1]),float(r['healthy'][-1])

SCENARIOS={
 'figure4_bispecific':Parameters(),
 'figure4_monospecific':Parameters(mode='monospecific'),
 'figure6':Parameters(kon1=1e4,kon2=1e4,koff1=1,koff2=1),
 'figure8':Parameters(kon1=1e4,kon2=5e4,koff1=1,koff2=1,rho1=1e-9,rho2=5e-11),
}
