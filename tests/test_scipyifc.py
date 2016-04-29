import unittest
import os
import numpy
from util import full
from qcifc.core import QuantumChemistry
from findifftool import core as findif
from . import vb
from vb.scipyifc import *
from vb import core as vb
import math
SQRTH = math.sqrt(0.5)
SQRT2 = math.sqrt(2.0)

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
        g = lambda x, dummy : numpy.array([2*(x[0]-1), 2*(x[1]-2.5)])
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
        self.p1 = (SQRTH, SQRTH)
        self.l0 = (1.0,)
        self.l1 = (-SQRTH,)
        self.x0 = self.p0 + self.l0 # (p, l)
        self.x1 = self.p1 + self.l1 # (p, l)
        #
        self.f = lambda p: -(p[0] + p[1])
        self.g = lambda p: full.init((-1.0, -1.0))
        self.h = lambda p: full.matrix((2, 2))
        self.c = (
            {'type': 'eq',
             'fun': lambda p: p[0]**2 + p[1]**2  - 1.0,
             'jac': lambda p: full.init((2*p[0], 2*p[1])),
             'hes': lambda p: full.init([[2, 0], [0, 2]])
            },
            )
        self.pfg = LagrangianMinimizer(
            self.p0, self.f, self.g, hessian=self.h, constraints=self.c
            )

    def test_setup_initial_point(self):
        self.pfg.x = full.init(self.p0 + self.l0)
        numpy.testing.assert_allclose(self.pfg.p, self.p0)

    def test_setup_final_point(self):
        self.pfg.x = full.init(self.p1 + self.l1)
        numpy.testing.assert_allclose(self.pfg.p, self.p1)

    def test_setup_random_point(self):
        xrand = numpy.random.random(3)
        self.pfg.x = xrand
        numpy.testing.assert_allclose(self.pfg.p, xrand[:2])

    def test_setup_initial_multiplier(self):
        self.pfg.x = full.init(self.p0 + self.l0)
        numpy.testing.assert_allclose(self.pfg.l, self.l0)

    def test_setup_final_multiplier(self):
        self.pfg.x = full.init(self.p1 + self.l1)
        numpy.testing.assert_allclose(self.pfg.l, self.l1)

    def test_setup_random_multiplier(self):
        xrand = numpy.random.random(3)
        self.pfg.x = xrand
        numpy.testing.assert_allclose(self.pfg.l, xrand[2:])

    def test_setup_initial_function(self):
        self.pfg.x = full.init(self.p0 + self.l0)
        numpy.testing.assert_allclose(self.pfg.fp(self.pfg.p), -1.0)

    def test_setup_final_function(self):
        self.pfg.x = full.init(self.p1 + self.l1)
        numpy.testing.assert_allclose(self.pfg.fp(self.pfg.p), -SQRT2)

    def test_setup_random_function(self):
        xrand = numpy.random.random(3)
        self.pfg.x = xrand
        numpy.testing.assert_allclose(self.f(self.pfg.p), -xrand[0] - xrand[1])

    def test_setup_constraint(self):
        self.assertIs(self.pfg.cp, self.c)

    def test_setup_initial_composite_varibale(self):
        self.pfg.x = full.init(self.p0 + self.l0)
        numpy.testing.assert_allclose(self.pfg.x, self.p0 + self.l0)

    def test_setup_final_composite_varibale(self):
        self.pfg.x = full.init(self.p1 + self.l1)
        numpy.testing.assert_allclose(self.pfg.x, self.p1 + self.l1)

    def test_setup_initial_lagrangian_function(self):
        self.pfg.x = full.init(self.p0 + self.l0)
        L = self.pfg.lagrangian()
        self.assertAlmostEqual(L(self.pfg.x, self.pfg), self.f(self.p0))

    def test_setup_final_lagrangian_function(self):
        self.pfg.x = full.init(self.p1 + self.l1)
        L = self.pfg.lagrangian()
        self.assertAlmostEqual(L(self.pfg.x, self.pfg), self.f(self.p1))

    def test_setup_random_lagrangian_function(self):
        xrand = numpy.random.random(3)
        self.pfg.x =  xrand
        p = xrand[:2]
        l = xrand[2:]
        L = self.pfg.lagrangian()
        self.assertAlmostEqual(L(self.pfg.x, self.pfg), self.f(p) - l*self.c[0]['fun'](p))

    def test_setup_lagrangian_initial_gradient(self):
        dL = self.pfg.lagrangian_derivative()
        x = self.pfg.x
        numpy.testing.assert_allclose(dL(x, self.pfg), (-3, -1, 0))

    def test_setup_lagrangian_final_gradient(self):
        self.pfg.x = full.init(self.p1 + self.l1)
        dL = self.pfg.lagrangian_derivative()
        numpy.testing.assert_allclose(dL(self.pfg.x, self.pfg), (0, 0, 0), atol=1e-7)

    def test_setup_random_lagrangian_gradient(self):
        xrand = numpy.random.random(3)
        self.pfg.x =  xrand
        p = xrand[:2]
        l = xrand[2:]
        dL = self.pfg.lagrangian_derivative()
        dLdp = tuple(self.g(p) - l*self.c[0]['jac'](p))
        dLdl = (-self.c[0]['fun'](p),)
        dLdx = dLdp + dLdl
        numpy.testing.assert_allclose(dL(self.pfg.x, self.pfg), dLdx)

    def test_minimize_from_final(self):
        self.pfg.x = full.init(self.p1 + self.l1) 
        self.pfg.minimize()
        numpy.testing.assert_allclose(self.pfg.p, self.p1)

    def test_no_side_effect(self):
        self.pfg.x = self.x1
        x = full.init(numpy.random.random(3))
        Lx = self.pfg.lagrangian()(x, self.pfg)
        numpy.testing.assert_allclose(self.pfg.x, self.x1)

    def test_no_side_effect(self):
        self.pfg.x = self.x1
        x = full.init(numpy.random.random(3))
        Lx = self.pfg.lagrangian_derivative()(x, self.pfg)
        numpy.testing.assert_allclose(self.pfg.x, self.x1)

    def test_newton_step(self):
        x0 = [0.75133256,0.74762074, -0.69209032]
        self.pfg.x = x0
        dL = self.pfg.lagrangian_derivative()
        d2L = self.pfg.lagrangian_hessian()
        x1 = self.pfg.x - dL(x0, self.pfg)/d2L(x0, self.pfg)
        ref_x1 = full.init([0.70827179, 0.70834183, -0.70514972])
        numpy.testing.assert_allclose(x1, ref_x1)

    def test_newton_step_with_update(self):
        x0 = [0.75133256,0.74762074, -0.69209032]
        self.pfg.x = x0
        dL = self.pfg.lagrangian_derivative()
        d2L = self.pfg.lagrangian_hessian()
        self.pfg.x = self.pfg.x - dL(x0, self.pfg)/d2L(x0, self.pfg)
        ref_x1 = full.init([0.70827179, 0.70834183, -0.70514972])
        numpy.testing.assert_allclose(self.pfg.x, ref_x1)
        

    def test_minimize_my_newton(self):
        self.pfg.x = full.init(self.p1 + self.l1) + 0.1*numpy.random.random(3)
        self.pfg.set_method('MyNewton')
        self.pfg.minimize()
        numpy.testing.assert_allclose(self.pfg.p, self.p1, atol=1e-5)

    @unittest.skip('hold')
    def test_minimize_from_final_plus_noise(self):
        self.pfg.x = full.init(self.p1 + self.l1) + 1e-4*numpy.random.random(3)
        self.pfg.minimize()
        numpy.testing.assert_allclose(self.pfg.p, self.p1, atol=1e-4)

    @unittest.skip('hold')
    def test_minimize(self):
        self.pfg.minimize()
        numpy.testing.assert_allclose(self.pfg.p, self.p1)


class TestVBMin(unittest.TestCase):

    def setUp(self):
        self.tmp = os.path.join(os.path.dirname(__file__), 'test_h2_c')
        def tmp(fil):
            return os.path.join(self.tmp, fil)

        self.qcifc = QuantumChemistry.get_factory('Dalton', tmpdir=self.tmp)
        vb.Nod.tmpdir = self.tmp
        vb.Nod.C = full.matrix((10, 2))
        vb.Nod.C[0, 0] = 1.0
        vb.Nod.C[5, 1] = 1.0
        vb.Nod.S = self.qcifc.get_overlap()
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
