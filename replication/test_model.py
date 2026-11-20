import unittest
from dataclasses import replace
import numpy as np
from model import *

class ModelTests(unittest.TestCase):
    def test_drug_mass_balance(self):
        for p in SCENARIOS.values():
            r=simulate(p);self.assertLess(max(abs(r['mass_error']))/p.dose,1e-6)
    def test_no_dose_no_effect(self):
        r=simulate(replace(Parameters(),dose=0))
        np.testing.assert_allclose(r['intracellular'],0,atol=1e-15)
    def test_no_target_no_effect(self):
        r=simulate(replace(Parameters(),rho1=0,rho2=0))
        np.testing.assert_allclose(r['intracellular'],0,atol=1e-15)
    def test_equal_target_symmetry(self):
        r=simulate(SCENARIOS['figure6'])
        np.testing.assert_allclose(r['intracellular'][:,1],r['intracellular'][:,2],atol=1e-12)
    def test_receptor_conservation_and_occupancy(self):
        for p in SCENARIOS.values():
            r=simulate(p);z=r['y'][:,3:].reshape(-1,4,6)*SCALE_DENSITY
            total1=z[:,:,0]+z[:,:,2]+(2 if p.mode=='monospecific' else 1)*z[:,:,4]
            np.testing.assert_allclose(total1,np.broadcast_to(densities(p)[:,0],total1.shape),atol=1e-17)
            if p.mode=='bispecific':
                total2=z[:,:,1]+z[:,:,3]+z[:,:,4]
                np.testing.assert_allclose(total2,np.broadcast_to(densities(p)[:,1],total2.shape),atol=1e-17)
            self.assertLessEqual(r['occupancy'].max(),1+1e-6)
    def test_tolerance_convergence(self):
        for p in SCENARIOS.values():
            a=simulate(p);b=simulate(p,rtol=1e-9,atol=1e-12)
            self.assertLess(abs(a['tumor'][-1]-b['tumor'][-1]),1e-5)
            self.assertLess(abs(a['healthy'][-1]-b['healthy'][-1]),1e-5)
    def test_surface_detailed_balance(self):
        p=SCENARIOS['figure8'];q2=p.koff2;q1=p.koff1*p.kon2*p.k2d2/(p.kon1*p.k2d1)
        self.assertAlmostEqual(p.kon1*p.koff2*p.k2d2*q1/(p.koff1*p.kon2*q2*p.k2d1),1.)
    def test_emax_weighting(self):
        r=simulate(Parameters());A=r['intracellular'][-1,1:]
        self.assertAlmostEqual(r['healthy'][-1],sum(HEALTHY_WEIGHTS*A/(Parameters().ec50+A)))
        self.assertLess(r['healthy'][-1],.21)
if __name__=='__main__':unittest.main()
