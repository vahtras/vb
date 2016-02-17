import unittest
import os
import scipy.optimize
import vb
import daltools





class VBTestBase(unittest.TestCase):

    @staticmethod
    def wf_energy(coef, wf):
        wf.coef = coef
        total_energy = wf.energy() + wf.Z
        return total_energy

    @staticmethod
    def structure_gradient(coef, wf):
        wf.coef = coef
        return wf.energygrad()[0]

    @staticmethod
    def constraint_norm(coef, wf):
        wf.coef = coef
        return wf.norm() - 1.0

    @staticmethod
    def constraint_norm_grad(coef, wf):
        wf.coef = coef
        return wf.normgrad()[0]

class VBTestH2(VBTestBase):

    def setUp(self):
        self.tmp = os.path.join(os.path.dirname(__file__), 'test_h2_ab')
        def tmp(fil):
            return os.path.join(self.tmp, fil)

        vb.Nod.tmpdir = self.tmp
        vb.Nod.C = daltools.util.full.unit(2)
        vb.Nod.S = daltools.one.read("OVERLAP", tmp("AOONEINT")).unpack().unblock()
        self.wf = vb.WaveFunction(
            [vb.Structure(
                [vb.Nod([0], [0]), vb.Nod([1], [1])],
                [1.0, 1.0]
                ),
             vb.Structure(
                [vb.Nod([0], [1]), vb.Nod([1], [0])],
                [1.0, 1.0]
                )
            ],
            [1.0, 1.0],
            tmpdir = self.tmp
        )

        self.constraints = (
            {'type': 'eq',
             'fun': self.constraint_norm,
             'jac': self.constraint_norm_grad,
             'args': (self.wf,)
            },
        )
        

    def tearDown(self):
        pass

    def test_solver_start_ionic(self):
        result = scipy.optimize.minimize(
            self.wf_energy, [1.0, 0.0], jac=self.structure_gradient, args=(self.wf,),
            constraints=self.constraints, method='SLSQP'
        )
        self.assertAlmostEqual(result.fun, -1.13728383)

    def test_solver_start_covalent(self):
        result = scipy.optimize.minimize(
            self.wf_energy, [0.0, 1.0], jac=self.structure_gradient, args=(self.wf,),
            constraints=self.constraints, method='SLSQP'
        )
        self.assertAlmostEqual(result.fun, -1.13728383)

    def test_constraint(self):
        self.wf.normalize()
        for c in self.constraints:
            self.assertAlmostEqual(c['fun'](self.wf.coef, self.wf), 0.0)

class VBTestH2C(VBTestBase):

    def setUp(self):
        self.tmp = os.path.join(os.path.dirname(__file__), 'test_h2_c')
        def tmp(fil):
            return os.path.join(self.tmp, fil)

        vb.Nod.tmpdir = self.tmp
        vb.Nod.C = daltools.util.full.matrix((10, 2))
        vb.Nod.C[0, 0] = 1.0
        vb.Nod.C[5, 1] = 1.0
        vb.Nod.S = daltools.one.read("OVERLAP", tmp("AOONEINT")).unpack().unblock()
        self.wf = vb.WaveFunction(
            [vb.Structure(
                [vb.Nod([0], [1]), vb.Nod([1], [0])],
                [1.0, 1.0]
                ),
             vb.Structure(
                [vb.Nod([0], [0]), vb.Nod([1], [1])],
                [1.0, 1.0]
                )
            ],
            [1.0, 1.0],
            tmpdir = self.tmp
        )

        self.constraints = (
            {'type': 'eq',
             'fun': self.constraint_norm,
             'jac': self.constraint_norm_grad,
             'args': (self.wf,)
            },
        )
        

    def tearDown(self):
        pass

    def test_constraint(self):
        self.wf.normalize()
        for c in self.constraints:
            self.assertAlmostEqual(c['fun'](self.wf.coef, self.wf), 0.0)

if __name__ == "__main__":
    unittest.main()
