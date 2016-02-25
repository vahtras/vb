import unittest
import numpy
import os
import scipy.optimize
import vb
import daltools
import abc
from daltools.util import full
from num_diff import findif

def update_wf(coef, wf):
    ncoef = len(wf.coef)
    wf.C[:, :] = coef[ncoef:].reshape(wf.C.shape, order='F')
    #wf.normalize_structures()
    wf.coef[:] = coef[:ncoef]

def update_and_reset_wf(coef, wf):
    ncoef = len(wf.coef)
    wf.C[:, :] = coef[ncoef:].reshape(wf.C.shape, order='F')
    wf.normalize_structures()
    wf.coef[:] = coef[:ncoef]

def extract_wf_coef(wf):
    coef = full.matrix(wf.coef.size + wf.C.size)
    coef[:wf.coef.size] = wf.coef
    coef[wf.coef.size:] = wf.C.ravel(order='F')
    return coef

class VBTestBase(unittest.TestCase):
    __metaclass__ = abc.ABCMeta


    @abc.abstractmethod
    def wf_energy(coef, wf):
        pass

    @abc.abstractmethod
    def wf_gradient(coef, wf):
        pass
    
    @abc.abstractmethod
    def constraint_norm(coef, wf):
        pass

    @abc.abstractmethod
    def constraint_norm_grad(coef, wf):
        pass

    @abc.abstractmethod
    def generate_structure_constraint(i):
        pass

    @abc.abstractmethod
    def generate_structure_constraint_gradient(i):
        pass

    @abc.abstractmethod
    def generate_orbital_constraint(i):
        pass

    @abc.abstractmethod
    def generate_orbital_constraint_gradient(i):
        pass

class VBTestH2(VBTestBase):

    @staticmethod
    def wf_energy(coef, wf):
        wf.coef[:] = coef
        total_energy = wf.energy() + wf.Z
        return total_energy

    @staticmethod
    def wf_gradient(coef, wf):
        wf.coef[:] = coef
        return wf.energygrad()[0]

    @staticmethod
    def constraint_norm(coef, wf):
        wf.coef[:] = coef
        return wf.norm() - 1.0

    @staticmethod
    def constraint_norm_grad(coef, wf):
        wf.coef[:] = coef
        return wf.normgrad()[0]

    @staticmethod
    def generate_structure_constraint(i):
        pass

    @staticmethod
    def generate_structure_constraint_gradient(i):
        pass

    @staticmethod
    def generate_orbital_constraint(i):
        pass

    @staticmethod
    def generate_orbital_constraint_gradient(i):
        pass

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
            self.wf_energy, [1.0, 0.0], jac=self.wf_gradient, args=(self.wf,),
            constraints=self.constraints, method='SLSQP'
        )
        self.assertAlmostEqual(result.fun, -1.13728383)

    def test_solver_start_covalent(self):
        result = scipy.optimize.minimize(
            self.wf_energy, [0.0, 1.0], jac=self.wf_gradient, args=(self.wf,),
            constraints=self.constraints, method='SLSQP'
        )
        self.assertAlmostEqual(result.fun, -1.13728383)

    def test_constraint(self):
        self.wf.normalize()
        for c in self.constraints:
            self.assertAlmostEqual(c['fun'](self.wf.coef, self.wf), 0.0)

class VBTestH2C(VBTestBase):

    @staticmethod
    def wf_energy(coef, wf):
        update_wf(coef, wf)
        total_energy = wf.energy() + wf.Z
        return total_energy

    @staticmethod
    def wf_gradient(coef, wf):
        update_wf(coef, wf)
        grad = full.matrix(wf.coef.size + wf.C.size)
        sg, og = wf.energygrad()
        grad[:wf.coef.size] = sg
        grad[wf.coef.size:] = og.ravel()
        return grad

    @staticmethod
    def constraint_norm(coef, wf):
        update_wf(coef, wf)
        return wf.norm() - 1.0

    @staticmethod
    def constraint_norm_grad(coef, wf):
        update_wf(coef, wf)
        grad = full.matrix(wf.coef.size + wf.C.size)
        sg, og = wf.normgrad()
        grad[:wf.coef.size] = sg
        grad[wf.coef.size:] = og.ravel(order='F')
        return grad

    @staticmethod
    def generate_structure_constraint(i):
        def fun(coef, wf):
            update_wf(coef, wf)
            return wf.structs[i].overlap() - 1.0
        return fun

    @staticmethod
    def generate_structure_constraint_gradient(i):
        def fun(coef, wf):
            update_wf(coef, wf)
            ds2 = wf.structs[i].overlap_gradient()
            grad = full.matrix(coef.shape)
            grad[wf.coef.size:] = ds2.ravel(order='F')
            return grad
        return fun

    @staticmethod
    def generate_orbital_constraint(i):
        def fun(coef, wf):
            update_wf(coef, wf)
            mo = wf.C[:, i]
            return (mo.T & (vb.Nod.S*mo)) - 1.0
        return fun

    @staticmethod
    def generate_orbital_constraint_gradient(i):
        def fun(coef, wf):
            update_wf(coef, wf)
            mo = wf.C[:, i]
            dc2 = 2*mo.T*vb.Nod.S
            grad = full.matrix(coef.shape)
            grad[wf.coef.size + mo.size*i: wf.coef.size + mo.size*(i+1)] = dc2.ravel()
            return grad
        return fun

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
             vb.Structure([vb.Nod([0], [0])], [1.0]),
             vb.Structure([vb.Nod([1], [1])], [1.0]),
            ],
            [1.0, 0.0, 0.0],
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
             'jac': self.generate_structure_constraint_gradient(0),
             'args': (self.wf,)
            },
            {'type': 'eq',
             'fun': self.generate_structure_constraint(1),
             'jac': self.generate_structure_constraint_gradient(1),
             'args': (self.wf,)
            },
            {'type': 'eq',
             'fun': self.generate_structure_constraint(2),
             'jac': self.generate_structure_constraint_gradient(2),
             'args': (self.wf,)
            },
            {'type': 'eq',
             'fun': self.generate_orbital_constraint(0),
             'jac': self.generate_orbital_constraint_gradient(0),
             'args': (self.wf,)
            },
            {'type': 'eq',
             'fun': self.generate_orbital_constraint(1),
             'jac': self.generate_orbital_constraint_gradient(1),
             'args': (self.wf,)
            },
        )

        self.final = full.matrix(23)
        self.final[:3] = [0.83675, 0.09850, 0.09850]
        self.final[3:8] = [0.7633862173, 0.3075441467, 0.0, 0.0, 0.0328947818]
        self.final[18:23] = [0.7633862173, 0.3075441467, 0.0, 0.0, -0.0328947818]

        update_and_reset_wf(self.final, self.wf)
        

    def tearDown(self):
        pass

    def test_Z(self):
        self.assertAlmostEqual(self.wf.Z, 0.715104, 6)

    def test_final_energy(self):
        energy = self.wf_energy(self.final, self.wf)
        self.assertAlmostEqual(energy, -1.14660543, places=4)

    def test_final_energy_gradient(self):
        gradient = self.wf_gradient(self.final, self.wf)
        numpy.testing.assert_allclose(gradient, 0*gradient, atol=5e-3)

    def test_final_constraints_norm(self):
        self.wf.normalize_structures()
        constraint = self.constraints[0]['fun'](self.final, self.wf)
        self.assertAlmostEqual(constraint, 0.0, delta=5e-5)

    def test_final_constraints_norm_grad(self):
        constraint_numgrad = findif.ndgrad(self.constraints[0]['fun'])(self.final, self.wf).view(full.matrix)
        constraint_grad = self.constraints[0]['jac'](self.final, self.wf)
        numpy.testing.assert_allclose(constraint_grad, constraint_numgrad)

    def test_final_constraints_orbital_1(self):
        constraint = self.constraints[4]['fun'](self.final, self.wf)
        self.assertAlmostEqual(constraint, 0.0, delta=1e-5)

    def test_final_constraints_orbital_1_grad(self):
        constraint_numgrad = findif.ndgrad(self.constraints[4]['fun'])(self.final, self.wf).view(full.matrix)
        constraint_grad = self.constraints[4]['jac'](self.final, self.wf)
        numpy.testing.assert_allclose(constraint_grad, constraint_numgrad)


    def test_final_constraints_orbital_2(self):
        constraint = self.constraints[5]['fun'](self.final, self.wf)
        self.assertAlmostEqual(constraint, 0.0, delta=1e-5)

    def test_final_constraints_orbital_2_grad(self):
        constraint_numgrad = findif.ndgrad(self.constraints[5]['fun'])(self.final, self.wf).view(full.matrix)
        constraint_grad = self.constraints[5]['jac'](self.final, self.wf)
        numpy.testing.assert_allclose(constraint_grad, constraint_numgrad)

    def test_final_constraints_structure_1(self):
        constraint = self.constraints[1]['fun'](self.final, self.wf)
        self.assertAlmostEqual(constraint, 0.0, delta=1e-5)

    def test_final_constraints_structure_1_grad(self):
        constraint_numgrad = findif.ndgrad(self.constraints[1]['fun'])(self.final, self.wf).view(full.matrix)
        constraint_grad = self.constraints[1]['jac'](self.final, self.wf)
        numpy.testing.assert_allclose(constraint_grad, constraint_numgrad)

    def test_final_constraints_structure_2(self):
        constraint = self.constraints[2]['fun'](self.final, self.wf)
        self.assertAlmostEqual(constraint, 0.0, delta=1e-5)

    def test_final_constraints_structure_2_grad(self):
        constraint_numgrad = findif.ndgrad(self.constraints[2]['fun'])(self.final, self.wf).view(full.matrix)
        constraint_grad = self.constraints[2]['jac'](self.final, self.wf)
        numpy.testing.assert_allclose(constraint_grad, constraint_numgrad)

    def test_final_constraints_structure_3(self):
        constraint = self.constraints[3]['fun'](self.final, self.wf)
        self.assertAlmostEqual(constraint, 0.0, delta=1e-5)

    def test_final_constraints_structure_3_grad(self):
        constraint_numgrad = findif.ndgrad(self.constraints[3]['fun'])(self.final, self.wf).view(full.matrix)
        constraint_grad = self.constraints[3]['jac'](self.final, self.wf)
        numpy.testing.assert_allclose(constraint_grad, constraint_numgrad)

    @unittest.skip('goes away')
    def test_solver_start_final(self):
        result = scipy.optimize.minimize(
            self.wf_energy, self.final, jac=self.wf_gradient, args=(self.wf,),
            constraints=self.constraints, method='SLSQP'
        )
        print result
        self.assertAlmostEqual(result.fun, -1.14660543, delta=1e-5)

    @unittest.skip('converges wrong')
    def test_solver_start_covalent(self):
        start_covalent = daltools.util.full.init(
            [1.0, 0.1, 0.1] + [
                1., .1, 0., 0., 0.1, 0., 0., 0., 0., 0,
                0., 0., 0., 0., 0., 1., .1, 0., 0., -0.1
                ]
            )
        result = scipy.optimize.minimize(
            self.wf_energy, start_covalent, jac=self.wf_gradient, args=(self.wf,),
            constraints=self.constraints, method='SLSQP'
        )
        self.assertAlmostEqual(result.fun, -1.14660543)

if __name__ == "__main__":
    unittest.main()
