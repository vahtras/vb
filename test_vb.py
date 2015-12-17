"""Test module of VB derivatives with numerical differentiation"""
import os
import numpy as np
import random, unittest
from .daltools import one
from .daltools.util import full, timing
from .vb import WaveFunction, Structure, is_two_electron
from .nod import Nod
from num_diff import findif as fd


class WaveFunction(WaveFunction):
    """Extended class with numerical derivatives"""

    def numnormgrad(self, delta=1e-3):
        """Numerical norm gradient"""
        return self.numgrad(self.norm, delta)

    def numenergygrad(self, delta=1e-3):
        """Return energy numerical gradient"""
        return self.numgrad(self.energy, delta)

    def numgrad(self, func, delta=1e-3):
        """
        # Numerical gradient
        """
        deltah = delta/2
        #
        # structure coefficients
        #
        structgrad = full.matrix(len(self.structs))
        #
        #
        for s in range(structgrad.shape[0]):
            self.coef[s] += deltah
            ep = func()
            self.coef[s] -= delta
            em = func()
            self.coef[s] += deltah
            structgrad[s] = (ep - em)/delta
        #
        # orbital gradient
        #
        r, c = Nod.C.shape
        orbgrad = full.matrix((r, c))
        for m in range(c):
            for t in range(r):
                Nod.C[t, m] += deltah
                ep = func()
                Nod.C[t, m] -= delta
                em = func()
                orbgrad[t, m] = (ep - em)/delta
                Nod.C[t, m] += deltah
        return (structgrad, orbgrad[:, :])

    def numnormhess(self, delta=1e-3):
        """Numerical norm Hessian"""
        return self.numhess(self.norm, delta)

    def numenergyhess(self, delta=1e-3):
        """Numerical energy Hessian"""
        return self.numhess(self.energy, delta)

    def numhess(self, func, delta=1e-3):
        """Generic numerical Hessian of input function func"""
        #
        # Numerical norm of Hessian for chosen elements
        #
        # Numerical gradient
        deltah = delta/2
        delta2 = delta*delta
        #
        # structure coefficients
        #
        ls = len(self.structs)
        ao, mo = Nod.C.shape
        strstrhess = full.matrix((ls, ls))
        orbstrhess = full.matrix((ao, mo, ls))
        orborbhess = full.matrix((ao, mo, ao, mo))
        #
        # Structure-structure
        #
        for p in range(ls):
            for q in range(ls):
                self.coef[p] += deltah
                self.coef[q] += deltah
                epp = func()
                self.coef[q] -= delta
                epm = func()
                self.coef[p] -= delta
                emm = func()
                self.coef[q] += delta
                emp = func()
                strstrhess[p, q] = (epp + emm - epm - emp)/delta2
                #
                # Reset
                #
                self.coef[p] += deltah
                self.coef[q] -= deltah
        #
        # Orbital-structure
        #
        for p in range(ls):
            for mu in range(ao):
                for m in range(mo):
                    self.coef[p] += deltah
                    Nod.C[mu, m] += deltah
                    epp = func()
                    Nod.C[mu, m] -= delta
                    epm = func()
                    self.coef[p] -= delta
                    emm = func()
                    Nod.C[mu, m] += delta
                    emp = func()
                    orbstrhess[mu, m, p] = (epp + emm - epm - emp)/delta2
                    #
                    # Reset
                    #
                    self.coef[p] += deltah
                    Nod.C[mu, m] -= deltah

        #
        # Orbital-orbital
        #
        for mu in range(ao):
            for m in range(mo):
                for nu in range(ao):
                    for n in range(mo):
                        Nod.C[mu, m] += deltah
                        Nod.C[nu, n] += deltah
                        epp = func()
                        Nod.C[nu, n] -= delta
                        epm = func()
                        Nod.C[mu, m] -= delta
                        emm = func()
                        Nod.C[nu, n] += delta
                        emp = func()
                        orborbhess[mu, m, nu, n] = \
                            (epp + emm - epm - emp)/delta2
                        #
                        # Reset
                        #
                        Nod.C[mu, m] += deltah
                        Nod.C[nu, n] -= deltah

        return (
            strstrhess,
            orbstrhess,
            orborbhess
            )



class VBTest(unittest.TestCase):
    def setUp(self):
        """
         Model the fci result as a wb wave function
      
         WF = cg(1sg|1sg) + cu(1su|1su)
      
         1sg=Ng(a+b)  norm Ng=1/sqrt(2(1+Sab))
         1su=Nu(a-b)  norm Nu=1/sqrt(2(1-Sab))
      
         WF = cg*Ng**2(a+b|a+b) + cu*Nu**2(a-b|a-b)
                                                                                           = (cg*Ng**2+cu*Nu**2)[(a|a) + (b|b)]
                                                                                           + (cg*Ng**2-cu*Nu**2)[(a|b) + (b|a)]
        """
      
        self.tmp = os.path.join(os.path.dirname(__file__), 'test_data')
        def tmp(fil):
            return os.path.join(self.tmp, fil)
      
        self.molinp=tmp("MOLECULE.INP")
        self.dalinp=tmp("DALTON.INP")
        self.one=tmp("AOONEINT")
        self.two=tmp("AOTWOINT")
#
# A common dalton input to calculate integrals
#
        dalinp=open(self.dalinp,'w')
        dalinp.write("""**DALTON
.INTEGRAL
**END OF
""")
        dalinp.close()
        #
        # Molecule input
        #
        molinp=open(self.molinp,'w')
        molinp.write("""BASIS
STO-3G
1
2
    1    0         A
        1.    2    1    1
A   0.0  0.0  0.0
B   0.0  0.0  0.7428
""")
        molinp.close()
#
# Setup VB wave function
#
        Nod.C=full.matrix((2,2)).random()
        Nod.S=one.read("OVERLAP",tmp('AOONEINT')).unpack().unblock()
        Nod.h=one.read("ONEHAMI",tmp('AOONEINT')).unpack().unblock()
        Nod.Z=one.readhead(tmp('AOONEINT'))['potnuc']
        ion=Structure( [Nod([0],[0]),Nod([1],[1])], [1,1] )
        cov=Structure( [Nod([0],[1]),Nod([1],[0])], [1,1] )
        cg=random.random()
        cu=random.random()
        Sab=0.13533528
        Ng2=1/(2*(1+Sab))
        Nu2=1/(2*(1-Sab))
        cion=cg*Ng2+cu*Nu2
        ccov=cg*Ng2-cu*Nu2
        self.WF=WaveFunction(
          [ion,cov],[cion,ccov],
          tmpdir=self.tmp
          )
        #
        # Threshold for numerical differentiation
        #
        self.delta=1e-4

    def test_str(self):
        import re
        self.assertTrue("(1)" in str(self.WF) and "(2)" in str(self.WF))

    def test_energy_orb_hessian(self):
        """Energy orbital Hessian"""
        _, _, numorbhess = self.WF.numenergyhess()
        _, _, anaorbhess = self.WF.energyhess()
        _, _, np.testing.assert_allclose(numorbhess, anaorbhess, self.delta)

    def test_energy_mixed_hessian(self):
        """Energy mixed Hessian"""
        _, nummixhess, _ = self.WF.numenergyhess(self.delta)
        _, anamixhess, _ = self.WF.energyhess()
        np.testing.assert_allclose(nummixhess, anamixhess, rtol=self.delta , atol=self.delta)

    def test_energy_struct_hessian(self):
        """Energy structure Hessian"""
        numstrhess, _, _ = self.WF.numenergyhess(self.delta)
        anastrhess, _, _ = self.WF.energyhess()
        np.testing.assert_allclose(numstrhess, anastrhess, rtol=self.delta, atol=self.delta)

    def test_energy_orb_gradient(self):
        """Energy orbital gradient"""
        _, numorbgr = self.WF.numenergygrad()
        _, anaorbgr = self.WF.energygrad()
        np.testing.assert_allclose(numorbgr, anaorbgr, rtol=self.delta)

    def test_energy_struct_gradient(self):
        """Energy structure gradient"""
        numstrgr, _ = self.WF.numenergygrad()
        anastrgr, _ = self.WF.energygrad()
        np.testing.assert_allclose(numstrgr, anastrgr, rtol=self.delta)

    def test_norm_orb_hessian(self):
        """Norm orbital hessian"""
        _, _, numorbhess = self.WF.numnormhess()
        _, _, anaorbhess = self.WF.normhess()
        _, _, np.testing.assert_allclose(numorbhess, anaorbhess, self.delta)

    def test_norm_mixed_hessian(self):
        """Norm mixed hessian"""
        _, nummixhess, _ = self.WF.numnormhess()
        _, anamixhess, _ = self.WF.normhess()
        np.testing.assert_allclose(nummixhess, anamixhess, self.delta)

    def test_norm_struct_hessian(self):
        """Norm structure hessian"""
        numstrhess = fd.clhess(self.WF, 'norm', 'coef')()
        anastrhess, _, _ = self.WF.normhess()
        np.testing.assert_allclose(numstrhess, anastrhess, self.delta)

    def test_norm_orb_gradient(self):
        """Norm orbital gradient"""
        _, numorbgr = self.WF.numnormgrad()
        _, anaorbgr = self.WF.normgrad()
        np.testing.assert_allclose(numorbgr, anaorbgr, rtol=self.delta)

    def test_norm_struct_gradient(self):
        """Norm structure gradient"""
        numstrgr = fd.clgrad(self.WF, 'norm', 'coef')()
        anastrgr, _ = self.WF.normgrad()
        np.testing.assert_allclose(numstrgr, anastrgr, rtol=self.delta)

    def test_nel(self):
        """Number of electrons"""
        nel=self.WF.nel()
        self.failUnlessAlmostEqual(nel,2.0,6,'Wrong electron number %g != %g'%(nel,2))

    def test_verify_not_implemented_exception(self):
        with self.assertRaises(NotImplementedError):
            is_two_electron()



if __name__ == "__main__":
   unittest.main()
