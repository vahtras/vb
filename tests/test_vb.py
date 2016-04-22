"""Test module of VB derivatives with numerical differentiation"""
import os
import math
import numpy as np
import random, unittest
from daltools import one
from util import full, timing
from findifftool import core as fd
from . import vb
from vb.core import WaveFunction, Structure
from vb.nod import Nod


class WaveFunctionND(WaveFunction):
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

    def setUp(self, arg=None):
        np.random.seed(0)
        self.set_tmpdir(arg)
        self.set_tmpfiles()
        self.init_nod()

    def tmp(self, fil):
        return os.path.join(self.tmpdir, fil)

    def set_tmpdir(self, tail):
        self.tmpdir = os.path.join(os.path.dirname(__file__), tail)
        
    def set_tmpfiles(self):
        self.molinp=self.tmp("MOLECULE.INP")
        self.dalinp=self.tmp("DALTON.INP")
        self.one=self.tmp("AOONEINT")
        self.two=self.tmp("AOTWOINT")

    def init_nod(self):
        Nod.S=one.read("OVERLAP", self.one).unpack().unblock()
        Nod.h=one.read("ONEHAMI", self.one).unpack().unblock()
        Nod.Z=one.readhead(self.one)['potnuc']


class VBTestH2A(VBTest):
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
      
        VBTest.setUp(self, 'test_h2_ab')
      
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

        ion=Structure( [Nod([0],[0]),Nod([1],[1])], [1.0, 1.0] )
        cov=Structure( [Nod([0],[1]),Nod([1],[0])], [1.0, 1.0] )
        cg=random.random()
        cu=random.random()
        Sab=0.13533528
        Ng2=1/(2*(1+Sab))
        Nu2=1/(2*(1-Sab))
        cion=cg*Ng2+cu*Nu2
        ccov=cg*Ng2-cu*Nu2
        self.WF=WaveFunctionND(
          [ion,cov],[cion,ccov],
          tmpdir=self.tmpdir
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
        np.testing.assert_allclose(numorbgr, anaorbgr, rtol=self.delta, atol=self.delta)

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
        self.assertAlmostEqual(nel, 2.0)


class VBTestH2B(VBTest):
    
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
      
        VBTest.setUp(self, 'test_h2_ab')
#
# Setup VB wave function
#
        Nod.C=full.unit(2)

        ion_a = Structure([Nod([0],[0])], [1.0], normalize=False)
        ion_b = Structure([Nod([1],[1])], [1.0], normalize=False)
        import math
        N = math.sqrt(0.5)
        cov=Structure([Nod([0],[1]),Nod([1],[0])], [N, N], normalize=False)
        cg = 0.99364675
        cu = -0.11254389
        Sab = 0.65987313
        Ng2=1/(2*(1+Sab))
        Nu2=1/(2*(1-Sab))
        cion = cg*Ng2 + cu*Nu2
        ccov = (cg*Ng2 - cu*Nu2)/N
        self.WF=WaveFunctionND(
          [cov, ion_a, ion_b],[ccov, cion, cion],
          tmpdir=self.tmpdir
          )
        #

    def test_final_energy(self):
        self.assertAlmostEqual(self.WF.energy() + Nod.Z, -1.137283835)

    def test_norm(self):
        self.assertAlmostEqual(self.WF.norm(), 1.0)

    def test_normalize_mo(self):
        self.WF.C = Nod.C*2
        self.WF.normalize_mo()
        np.testing.assert_allclose(self.WF.C, Nod.C)

    def test_vb_vector(self):
        self.WF.normalize_structures()
        np.testing.assert_allclose(
            self.WF.coef,
            (0.787469, 0.133870, 0.133870),
            rtol=1e-5
            )

    def test_weights(self):
        self.WF.normalize_structures()
        np.testing.assert_allclose(
            self.WF.structure_weights(),
            (0.784329, 0.107836, 0.107836),
            rtol=5e-6
            )
            

    def test_structure_overlap(self):
        self.WF.normalize()
        np.testing.assert_allclose(
            self.WF.structure_overlap(), 
            [[1.00000000, 0.77890423, 0.77890423],
             [0.77890423, 1.00000000, 0.43543258],
             [0.77890423, 0.43543258, 1.00000000]]
        )

    def test_structure_hamiltonian(self):
        self.WF.normalize()
        np.testing.assert_allclose(
            self.WF.structure_hamiltonian(),
            [[-1.12438723, -0.92376625, -0.92376625],
             [-0.92376625, -0.75220865, -0.65716238],
             [-0.92376625, -0.65716238, -0.75220865]]
        )

    def test_eigenvalues(self):
        self.WF.normalize()
        e, _ = self.WF.eigenvalues_vectors()
        np.testing.assert_allclose(
            e, [-1.137284, -0.168352, 0.483143],
            rtol=1e-5
        )

    def test_eigenvectors(self):
        _, v = self.WF.eigenvalues_vectors()
        # fix phase
        if v[0, 0] < 0: v[:, 0] *= -1
        if v[1, 1] > 0: v[:, 1] *= -1
        if v[2, 2] > 0: v[:, 2] *= -1
        np.testing.assert_allclose(
            v, [
            [0.787469, 0.000000, 2.417515],
            [0.133870, -0.941081, -1.494602],
            [0.133870, 0.941081, -1.494602]
            ],
            rtol=1e-5, atol=1e-5
        )

class VBTestH2C(VBTest):
    
    def setUp(self):
        """
        """
      
        VBTest.setUp(self, 'test_h2_c')
#
# Setup VB wave function
#
        Nod.C=full.init(
            [
                [0.7633862173, 0.3075441467, 
                0.0000000000, 0.0000000000, 0.0328937818, 
                0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 
                0.7633862173, 0.3075441467,
                0.0000000000, 0.0000000000, -0.0328937818]
            ]
        )

        ion_a = Structure([Nod([0],[0])], [1.0])
        ion_b = Structure([Nod([1],[1])], [1.0])
        cov = Structure([Nod([0],[1]),Nod([1],[0])], [1.0, 1.0])
        self.WF=WaveFunctionND(
          [cov, ion_a, ion_b],[0.83675, 0.09850, 0.09850],
          tmpdir=self.tmpdir
          )
        self.WF.normalize()
        #

    def test_Z(self):
        self.assertAlmostEqual(self.WF.Z, 0.715104, places=6)

    def test_final_energy(self):
        self.assertAlmostEqual(self.WF.energy() + Nod.Z, -1.14660543)

    def test_norm(self):
        self.assertAlmostEqual(self.WF.norm(), 1.0)

    def test_orbital_norm(self):
        S_mo = full.matrix.diag(self.WF.C.T*Nod.S*self.WF.C)
        np.testing.assert_allclose(S_mo, [1.0, 1.0])

    def test_normalize_mo(self):
        self.WF.C = Nod.C*2
        self.WF.normalize_mo()
        np.testing.assert_allclose(self.WF.C, Nod.C)

    def test_vb_vector(self):
        np.testing.assert_allclose(
            self.WF.coef,
            [0.83675, 0.09850, 0.09850],
            rtol=1e-5
            )

    def test_determinant_coef(self):
        coef = []
        for s, cs in zip(self.WF.structs, self.WF.coef):
            for d, cd in zip(s.nods, s.coef):
                coef.append(cs*cd)
        np.testing.assert_allclose(
            coef, [0.48184, 0.48184, 0.09850, 0.09850],
            atol=1e-5)



    def test_weights(self):
        self.WF.normalize_structures()
        np.testing.assert_allclose(
            self.WF.structure_weights(),
            (0.83545, 0.08228, 0.08228),
            atol=1e-5
            )
            

    def test_first_excited(self):
        self.WF.normalize()
        e, _ = self.WF.eigenvalues_vectors()
        np.testing.assert_allclose(
            e[1], -0.256277,
            rtol=1e-5
        )


class VBTestFH(VBTest):

    def setUp(self):
        VBTest.setUp(self, 'test_fh')

        Nod.C = np.loadtxt(self.tmp('orb')).view(full.matrix)
        # Fix d-function normalization
        Nod.C[14, :] *= math.sqrt(1.0/3.0)
        Nod.C[17, :] *= math.sqrt(1.0/3.0)
        Nod.C[19, :] *= math.sqrt(1.0/3.0)

        cov = Structure(
            [Nod([0, 1, 2, 3, 4],[0, 1, 2, 3, 5]),
             Nod([0, 1, 2, 3, 5],[0, 1, 2, 3, 4])],
            [1.0, 1.0]
            )
        ion_a = Structure([Nod([0, 1, 2, 3, 4],[0, 1, 2, 3, 4])], [1.0])
        ion_b = Structure([Nod([0, 1, 2, 3, 5],[0, 1, 2, 3, 5])], [1.0])

        self.wf = WaveFunctionND([cov, ion_a, ion_b], [0.66526, -0.36678, -0.07321], tmpdir=self.tmpdir)
        self.wf.normalize()


    def test_Z(self):
        self.assertAlmostEqual(self.wf.Z, 5.193669438059)

    @unittest.skip('hold')
    def test_energy(self):
        self.assertAlmostEqual(self.wf.energy() + self.wf.Z, -100.03323961)

    #@unittest.skip('hold')
    def test_orb_normalized(self):
        C = self.wf.C.copy()
        self.wf.normalize_mo()
        np.testing.assert_allclose(C, self.wf.C)

    #@unittest.skip('hold')
    def test_orbital_overlap(self):
        ref_overlap = full.init([
        [ 1.000000,  0.000000,   0.000000,   0.000000,   0.074261,  0.064405],
        [-0.000000,  1.000000,   0.000000,   0.000000,   0.214777,  0.361800],
        [ 0.000000,  0.000000,   1.000000,   0.000000,   0.000000,  0.000000],
        [ 0.000000,  0.000000,   0.000000,   1.000000,   0.000000,  0.000000],
        [ 0.074261,  0.214777,   0.000000,   0.000000,   1.000000, -0.419052],
        [ 0.064405,  0.361800,   0.000000,   0.000000,  -0.419052,  1.000000]
        ])
        self.wf.normalize_mo()
        overlap = self.wf.C.T*Nod.S*self.wf.C
        print(overlap); print(ref_overlap)
        np.testing.assert_allclose(overlap, ref_overlap, atol=1e-6)


    @unittest.skip('off-diagonal still wrong')
    def test_structure_overlap(self):
        np.testing.assert_allclose(
            self.wf.structure_overlap(), [
            [1.000000, -0.685108, -0.685108],
            [-0.685108, 1.000000, 0.306654],
            [-0.685108, 0.306654, 1.000000]
            ]
        )


    def test_non_VBSCF(self):
        with self.assertRaises(NotImplementedError):
            wf = WaveFunction([], [], VBSCF=False, tmpdir=self.tmpdir)

if __name__ == "__main__":
   unittest.main()
