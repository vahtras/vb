import unittest
import os
import numpy
import vb
from .daltools.util import full
from .daltools import one
from scipyifc import *
from num_diff import findif

class MinTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_parabola(self):
        x0 = numpy.array((2.0, 0.0))
        f = lambda x, dummy : (x[0] - 1)**2 + (x[1] - 2)**2
        g = lambda x, dummy: numpy.array([2*(x[0] - 1), 2*(x[1] - 2)])
        xfg = Minimizer(x0, f, g, method='L-BFGS-B')
        xfg.minimize()
        numpy.testing.assert_allclose(xfg.x, (1.0, 2.0))

    def test_constraint(self):
        f = lambda x, dummy : (x[0] - 1)**2 + (x[1] - 2.5)**2
        g = None
        c = (
            {'type': 'ineq', 'fun': lambda x: x[0] - 2*x[1] + 2},
            {'type': 'ineq', 'fun': lambda x: x[0] - 2*x[1] + 6},
            {'type': 'ineq', 'fun': lambda x: -x[0] + 2*x[1] + 2},
            )
        b = ((0, None), (0, None))

        x0 = numpy.array((0.0, 0.0))
        xfg = Minimizer(x0, f, g, method='SLSQP', constraints=c, bounds=b)
        xfg.minimize()
        numpy.testing.assert_allclose(xfg.x, (1.4, 1.7))

class TestLagrangianMinTest(unittest.TestCase):

    def setUp(self):
        """Opimize x+y constrained to a circle"""
        # Initial values
        self.p0 = (1.0, 0.0)
        self.l0 = (1.0,)
        self.x0 = self.p0 + self.l0 # (p, l)
        #
        self.f = lambda x: x[0] + x[1]
        g = lambda x: full.init((1, 1))
        self.c = (
            {'type': 'eq',
             'fun': lambda x: x[0]**2 + x[1]**2  - 1.0,
             'jac': lambda x: full.init((2*x[0], 2*x[1]))
            },
            )
        self.xfg = LagrangianMinimizer(self.p0, self.f, g, method='BFGS', args=(self,), constraints=self.c)

    def test_setup_start_point(self):
        numpy.testing.assert_allclose(self.xfg.p, self.p0)

    def test_setup_lagrange_variable(self):
        self.assertEqual(self.xfg.l, (1.0))

    def test_setup_constraint(self):
        self.assertIs(self.xfg.c, self.c)

    def test_setup_composite_varibale(self):
        numpy.testing.assert_allclose(self.xfg.x, tuple(self.p0) + tuple(self.xfg.l))

    def test_setup_lagrangian_function(self):
        L = self.xfg.lagrangian()
        x = self.xfg.x
        self.assertAlmostEqual(L(x, self.xfg), self.f(self.p0))



    @unittest.skip('wait')
    def test_constraint(self):

        x0 = numpy.array((0.0, 0.0))
        xfg = LagrangianMinimizer(x0, f, g, method='BFGS', constraints=c)
        xfg.minimize()
        numpy.testing.assert_allclose(xfg.x, (0.7071067811865476, 0.7071067811865476))

class TestVBMin(unittest.TestCase):

    def setUp(self):
        self.tmp = os.path.join(os.path.dirname(__file__), 'test_h2_c')
        def tmp(fil):
            return os.path.join(self.tmp, fil)

        vb.Nod.tmpdir = self.tmp
        vb.Nod.C = full.matrix((10, 2))
        vb.Nod.C[0, 0] = 1.0
        vb.Nod.C[5, 1] = 1.0
        vb.Nod.S = one.read("OVERLAP", tmp("AOONEINT")).unpack().unblock()
        self.blockdims = ((5, 5), (1, 1))
        self.wf = vb.WaveFunction(
            [vb.Structure(
                [vb.Nod([0], [1]), vb.Nod([1], [0])],
                [1.0, 1.0]
                ),
             vb.Structure([vb.Nod([0], [0])], [1.0]),
             vb.Structure([vb.Nod([1], [1])], [1.0]),
            ],
            [1.0, 0.0, 0.0],
            tmpdir = self.tmp,
            blockdims=self.blockdims
        )

        # In this setup we have local expansions of mos, leading to a block diagonal C
        self.final = full.matrix(13)
        self.final_coef =[0.83675, 0.09850, 0.09850]
        self.final[:3] = self.final_coef
        self.final_C = full.init([
            [0.7633862173, 0.3075441467, 0.0, 0.0, 0.0328947818,0,0,0,0,0],
            [0,0,0,0,0, 0.7633862173, 0.3075441467, 0.0, 0.0, -0.0328947818]
            ])
        self.final[3:8] = self.final_C[:5, 0]
        self.final[8:13] = self.final_C[5:, 1]

        self.wf.normalize_structures()
        self.xfg = VBMinimizer(self.wf)

    def test_updater_structure_coefficiets(self):
        self.xfg.x = self.final
        numpy.testing.assert_allclose(self.xfg.coef, self.final_coef)

    def test_updater_orbital_coefficients(self):
        self.xfg.x = self.final
        numpy.testing.assert_allclose(self.xfg.C, self.final_C)

    def test_Z(self):
        self.assertAlmostEqual(self.xfg.Z, 0.715104, 6)

    def test_final_energy(self):
        energy = self.xfg.f(self.final, self.xfg)
        self.assertAlmostEqual(energy, -1.14660543, places=4)

    def test_final_energy_gradient(self):
        constraint_numgrad = findif.ndgrad(self.xfg.f)(self.final, self.xfg).view(full.matrix)
        constraint_grad = self.xfg.g(self.final, self.xfg)
        numpy.testing.assert_allclose(constraint_grad, constraint_numgrad, atol=1e-7)

    def test_final_constraints_norm(self):
        self.xfg.wf.C[:, :] = self.final_C
        self.xfg.wf.normalize_structures()
        constraint = VBMinimizer.constraint_norm(self.final, self.xfg)
        self.assertAlmostEqual(constraint, 0.0, delta=5e-5)

    def test_final_constraints_norm_grad(self):
        constraint_numgrad = findif.ndgrad(VBMinimizer.constraint_norm)
        constraint_grad = VBMinimizer.constraint_norm_grad(self.final, self.xfg)
        numpy.testing.assert_allclose(constraint_grad, constraint_numgrad, atol=1e-7)

    def test_final_constraints_norm_grad(self):
        constraint_grad = VBMinimizer.constraint_norm_grad(self.final, self.xfg)
        gradf = findif.ndgrad(VBMinimizer.constraint_norm)
        constraint_numgrad = gradf(self.final, self.xfg)
        numpy.testing.assert_allclose(constraint_grad, constraint_numgrad)

    def test_final_constraints_orbital_0(self):
        constraint = VBMinimizer.constraint_orbital_norm(0)(self.final, self.xfg)
        self.assertAlmostEqual(constraint, 0.0, delta=1e-5)

    def test_final_constraints_orbital_1(self):
        constraint = VBMinimizer.constraint_orbital_norm(1)(self.final, self.xfg)
        self.assertAlmostEqual(constraint, 0.0, delta=1e-5)

    def test_final_constraints_orbital_norm_grad_0(self):
        constraint_grad = VBMinimizer.constraint_orbital_norm_grad(0)(self.final, self.xfg)
        gradf = findif.ndgrad(VBMinimizer.constraint_orbital_norm(0))
        constraint_numgrad = gradf(self.final, self.xfg)
        numpy.testing.assert_allclose(constraint_grad, constraint_numgrad)

    def test_final_constraints_orbital_norm_grad_1(self):
        constraint_grad = VBMinimizer.constraint_orbital_norm_grad(1)(self.final, self.xfg)
        gradf = findif.ndgrad(VBMinimizer.constraint_orbital_norm(1))
        constraint_numgrad = gradf(self.final, self.xfg)
        numpy.testing.assert_allclose(constraint_grad, constraint_numgrad)

    def test_final_constraints_structure_0(self):
        self.xfg.wf.C[:, :] = self.final_C
        self.xfg.normalize_structures()
        constraint = VBMinimizer.constraint_structure_norm(0)(self.final, self.xfg)
        self.assertAlmostEqual(constraint, 0.0, delta=1e-5)

    def test_final_constraints_structure_1(self):
        self.xfg.wf.C[:, :] = self.final_C
        self.xfg.normalize_structures()
        constraint = VBMinimizer.constraint_structure_norm(1)(self.final, self.xfg)
        self.assertAlmostEqual(constraint, 0.0, delta=1e-5)

    def test_final_constraints_structure_2(self):
        self.xfg.wf.C[:, :] = self.final_C
        self.xfg.normalize_structures()
        constraint = VBMinimizer.constraint_structure_norm(2)(self.final, self.xfg)
        self.assertAlmostEqual(constraint, 0.0, delta=1e-5)

    def test_final_constraints_structure_norm_grad_0(self):
        constraint_grad = VBMinimizer.constraint_structure_norm_grad(0)(self.final, self.xfg)
        gradf = findif.ndgrad(VBMinimizer.constraint_structure_norm(0))
        constraint_numgrad = gradf(self.final, self.xfg)
        numpy.testing.assert_allclose(constraint_grad, constraint_numgrad)

    def test_final_constraints_structure_norm_grad_1(self):
        constraint_grad = VBMinimizer.constraint_structure_norm_grad(1)(self.final, self.xfg)
        gradf = findif.ndgrad(VBMinimizer.constraint_structure_norm(1))
        constraint_numgrad = gradf(self.final, self.xfg)
        numpy.testing.assert_allclose(constraint_grad, constraint_numgrad)

    def test_final_constraints_structure_norm_grad_2(self):
        constraint_grad = VBMinimizer.constraint_structure_norm_grad(2)(self.final, self.xfg)
        gradf = findif.ndgrad(VBMinimizer.constraint_structure_norm(2))
        constraint_numgrad = gradf(self.final, self.xfg)
        numpy.testing.assert_allclose(constraint_grad, constraint_numgrad)

    def test_solver_start_final(self):
        self.xfg.x = self.final
        self.xfg.minimize()
        self.assertAlmostEqual(self.xfg.value, -1.14660543, places=4)

if __name__ == "__main__":
    unittest.main()
