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
    def energy_gradient(coef, wf):
        wf.coef = coef[:len(wf.coef)]
        wf.C = wf[len(wf.coef):].reshape(wf.C.shape)
        return wf.energygrad()

    @staticmethod
    def constraint_norm(coef, wf):
        wf.coef = coef
        return wf.norm() - 1.0

    @staticmethod
    def generate_structure_constraint(i):
        def fun(coef, wf):
            S = wf.structs[i]
            return S*S - 1.0
        return fun

    @staticmethod
    def generate_structure_constraint_gradient(i):
        def fun(coef, wf):
            S = wf.structs[i]
            return S*S - 1.0
        return fun

    @staticmethod
    def generate_orbital_constraint(i):
        def fun(coef, wf):
            cmo = wf.C[:, i]
            return cmo.T*vb.Nod.S*cmo - 1.0
        return fun

    @staticmethod
    def constraint_norm_structure_grad(coef, wf):
        wf.coef = coef
        return wf.normgrad()[0]

    @staticmethod
    def constraint_norm_grad(coef, wf):
        wf.coef = coef[:len(wf.coef)]
        wf.C = coef[len(wf.coef):].reshape(wf.C.shape)
        structgrad, orbgrad = wf.normgrad()
        norm_grad = daltools.util.full.matrix(structgrad.size + orbgrad.size)
        norm_grad[:structgrad.size] = structgrad
        norm_grad[structgrad.size:] = orbgrad.ravel()
        return norm_grad

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
             'jac': self.constraint_norm_structure_grad,
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
            {'type': 'eq',
             'fun': self.generate_structure_constraint(0),
             'jac': None,
             'args': (self.wf,)
            },
            {'type': 'eq',
             'fun': self.generate_structure_constraint(1),
             'jac': None,
             'args': (self.wf,)
            },
            {'type': 'eq',
             'fun': self.generate_orbital_constraint(0),
             'jac': None,
             'args': (self.wf,)
            },
            {'type': 'eq',
             'fun': self.generate_orbital_constraint(1),
             'jac': None,
             'args': (self.wf,)
            },
        )
        

    def tearDown(self):
        pass

    def test_constraint(self):
        self.wf.normalize()
        for c in self.constraints:
            self.assertAlmostEqual(c['fun'](self.wf.coef, self.wf), 0.0)

    @unittest.skip('wait')
    def test_solver_start_covalent(self):
        start_covalent = daltools.util.full.init(
            [1.0, 0.0] + [
                1., 0., 0., 0., 0., 0., 0., 0., 0., 0,
                0., 0., 0., 0., 0., 1., 0., 0., 0., 0
                ]
            )
        result = scipy.optimize.minimize(
            self.wf_energy, start_covalent, jac=self.structure_gradient, args=(self.wf,),
            constraints=self.constraints, method='SLSQP'
        )
        self.assertAlmostEqual(result.fun, -1.13728383)

if __name__ == "__main__":
    unittest.main()
