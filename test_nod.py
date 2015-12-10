import unittest
import numpy as np
import os
from vb import Nod, DKL, BraKet
from vb import Structure, StructError
from daltools.util.full import init, matrix
from num_diff.findif import clgrad, clhess, clmixhess, DELTA

class NodTest(unittest.TestCase):

    def setUp(self):
        Nod.S = init([[1.0, 0.1], [0.1, 1.0]])
        Nod.C = init([[0.7, 0.7], [0.7, -0.7]])

    def tearDown(self):
        pass

    def test_vac_empty(self):
        vac = Nod([], [])
        self.assertEqual(vac.electrons(), 0)

    def test_overlap_class_variable(self):
        one = Nod([0], [])
        np.testing.assert_allclose(one.S, Nod.S)

    def test_retrive_alpha_orbitals(self):
        det = Nod([1, 2], [2, 3])
        self.assertListEqual(det(0), [1, 2])

    def test_retrive_beta_orbitals(self):
        det = Nod([1, 2], [2, 3])
        self.assertListEqual(det(1), [2, 3])

    def test_repr(self):
        det = Nod([1, 2], [2, 3])
        self.assertEqual(str(det), '(1 2|2 3)')

    def test_empty_determinant_returns_none(self):
        det = Nod([0], [])
        self.assertIsNone(det.orbitals()[1])

    def test_alpha_orbitals(self):
        det = Nod([0], [1])
        alpha_orbitals, _ = det.orbitals()
        np.testing.assert_allclose(alpha_orbitals, [[.7], [.7]])

    def test_beta_orbitals(self):
        det = Nod([0], [1])
        _, beta_orbitals = det.orbitals()
        np.testing.assert_allclose(beta_orbitals, [[.7], [-.7]])

    def test_vac_normalized(self):
        det = Nod([], [])
        self.assertEqual(det*det, 1)

    def test_alpha_beta_orthogonal(self):
        alpha = Nod([0], [])
        beta = Nod([], [0])
        self.assertEqual(alpha*beta, 0)

    def test_single_norm(self):
        alpha = Nod([0], [])
        self.assertEqual(alpha*alpha, 2*0.7*0.77)

    def test_closed_shell_norm(self):
        sigmag = Nod([0], [0])
        self.assertEqual(sigmag*sigmag, (2*0.7*0.77)**2)

    def test_high_spin_norm(self):
        sigmagu = Nod([0, 1], [])
        self.assertAlmostEqual(sigmagu*sigmagu, 2*.7*(1+.1)*.7*2*.7*(1-.1)*.7)

    def test_vac_ao_density(self):
        vac = Nod([], [])
        np.testing.assert_allclose(vac.ao_density(), [[[0, 0], [0, 0]],  [[0, 0], [0, 0]]])

    def test_vac_mo_density(self):
        vac = Nod([], [])
        self.assertEqual(vac.mo_density(), [None, None])

    def test_alpha_mo_density(self):
        alpha = Nod([0], [])
        DKL_a, _ = DKL(alpha, alpha, mo=1)
        np.testing.assert_allclose(DKL_a, [[1./1.078]])

    def test_alpha_ao_density(self):
        alpha = Nod([0], [])
        DKL_a, _ = DKL(alpha, alpha, mo=0)
        d_a = 0.49/1.078
        np.testing.assert_allclose(DKL_a, [[d_a, d_a], [d_a, d_a]])

class BraKetTest(unittest.TestCase):

    def setUp(self):
        BraKet.tmpdir = os.path.join(os.path.dirname(__file__), 'test_data')
        Nod.S = init([[1.0, 0.1], [0.1, 1.0]])
        Nod.h = init([[-0.5, 0.1], [0.1, -0.25]])
        Nod.C = init([[0.7, 0.6], [0.6, -0.7]])
        self.a0a0 = BraKet(Nod([0], []), Nod([0], []))
        self.a0a1 = BraKet(Nod([0], []), Nod([1], []))
        self.a1a0 = BraKet(Nod([1], []), Nod([0], []))
        self.a1a1 = BraKet(Nod([1], []), Nod([1], []))
        self.b0b0 = BraKet(Nod([], [0]), Nod([], [0]))
        self.b0b1 = BraKet(Nod([], [0]), Nod([], [1]))
        self.b1b0 = BraKet(Nod([], [1]), Nod([], [0]))
        self.b1b1 = BraKet(Nod([], [1]), Nod([], [1]))
        self.K00_L00  = BraKet(Nod([0], [0]), Nod([0], [0]))
        self.K00_L01  = BraKet(Nod([0], [0]), Nod([0], [1]))
        self.K00_L10  = BraKet(Nod([0], [0]), Nod([1], [0]))
        self.K00_L11  = BraKet(Nod([0], [0]), Nod([1], [1]))
        self.K01_L00  = BraKet(Nod([0], [1]), Nod([0], [0]))
        self.K01_L01  = BraKet(Nod([0], [1]), Nod([0], [1]))
        self.K01_L10  = BraKet(Nod([0], [1]), Nod([1], [0]))
        self.K01_L11  = BraKet(Nod([0], [1]), Nod([1], [1]))
        self.K10_L00  = BraKet(Nod([1], [0]), Nod([0], [0]))
        self.K10_L01  = BraKet(Nod([1], [0]), Nod([0], [1]))
        self.K10_L10  = BraKet(Nod([1], [0]), Nod([1], [0]))
        self.K10_L11  = BraKet(Nod([1], [0]), Nod([1], [1]))
        self.K11_L00  = BraKet(Nod([1], [1]), Nod([0], [0]))
        self.K11_L01  = BraKet(Nod([1], [1]), Nod([0], [1]))
        self.K11_L10  = BraKet(Nod([1], [1]), Nod([1], [0]))
        self.K11_L11  = BraKet(Nod([1], [1]), Nod([1], [1]))
        self.f = (Nod.h, Nod.h)

    def tearDown(self):
        pass

    def test_str(self):
        self.assertEqual(str(self.K00_L00), "<(0|0)|...|(0|0)>")

    def test_nod_pair_setup_aa(self):
        self.assertAlmostEqual(self.a0a0.K*self.a0a0.L, self.a0a0.overlap())

    def test_nod_pair_setup_bb(self):
        self.assertAlmostEqual(self.b0b0.K*self.b0b0.L, self.b0b0.overlap())


    def test_a0a0_right_numerical_differential_overlap(self):
        KdL = clgrad(self.a0a0, 'overlap', 'L.C')()
        np.testing.assert_allclose(KdL, [[0.76, 0.0], [0.67, 0.0]])

    def test_a0a0_right_analytical_differential_overlap(self):
        KL00 = self.a0a0.right_overlap_gradient()
        np.testing.assert_allclose(KL00, [[0.76, 0.0], [0.67, 0]])

    def test_b0b0_right_numerical_differential_overlap(self):
        KdL = clgrad(self.b0b0, 'overlap', 'L.C')()
        np.testing.assert_allclose(KdL, [[0.76, 0.0], [0.67, 0.0]])

    def test_b0b0_right_analytical_differential_overlap(self):
        KL00 = self.b0b0.right_overlap_gradient()
        np.testing.assert_allclose(KL00, [[0.76, 0.0], [0.67, 0]])

    def test_a0a0_energy(self):
        self.assertAlmostEqual(self.a0a0*self.f, -0.251)

    def test_b0b0_energy(self):
        self.assertAlmostEqual(self.a0a0*self.f, -0.251)

    def test_a0a0_one_energy_right_differential(self):
        num_diff = clgrad(self.a0a0, '__mul__', 'L.C', )(self.f)
        ana_diff = self.a0a0.right_1el_energy_gradient(self.f)
        np.testing.assert_allclose(ana_diff, num_diff)

    def test_b0b0_one_energy_right_differential(self):
        num_diff = clgrad(self.b0b0, '__mul__', 'L.C', )(self.f)
        ana_diff = self.b0b0.right_1el_energy_gradient(self.f)
        np.testing.assert_allclose(ana_diff, num_diff)

    def test_a0a0_one_energy_mixed_differential(self):
        num_diff = clmixhess(self.a0a0, '__mul__', 'L.C', 'K.C')(self.f)
        ana_diff = self.a0a0.mixed_1el_energy_hessian(self.f)
        np.testing.assert_allclose(ana_diff, num_diff, rtol=DELTA, atol=DELTA)

    def test_b0b0_one_energy_mixed_differential(self):
        num_diff = clmixhess(self.b0b0, '__mul__', 'L.C', 'K.C')(self.f)
        ana_diff = self.b0b0.mixed_1el_energy_hessian(self.f)
        np.testing.assert_allclose(ana_diff, num_diff, rtol=DELTA, atol=DELTA)

    def test_a0a0_2el_energy(self):
        self.assertAlmostEqual(self.a0a0.twoel_tme(), 0)



###

    def test_a0a1_right_numerical_differential_overlap(self):
        KdL = clgrad(self.a0a1, 'overlap', 'L.C')()
        np.testing.assert_allclose(KdL, [[0.0, 0.76], [0.0, 0.67]])

    def test_b0b1_right_numerical_differential_overlap(self):
        KdL = clgrad(self.b0b1, 'overlap', 'L.C')()
        np.testing.assert_allclose(KdL, [[0.0, 0.76], [0.0, 0.67]])

    def test_a0a1_right_analytical_differential_overlap(self):
        KL01 = self.a0a1.right_overlap_gradient()
        np.testing.assert_allclose(KL01, [[0, 0.76], [0.0, 0.67]])

    def test_b0b1_right_analytical_differential_overlap(self):
        KL01 = self.b0b1.right_overlap_gradient()
        np.testing.assert_allclose(KL01, [[0, 0.76], [0.0, 0.67]])

    def test_a0a1_one_energy_right_differential(self):
        num_diff = clgrad(self.a0a0, '__mul__', 'L.C', )(self.f)
        ana_diff = self.a0a0.right_1el_energy_gradient(self.f)
        np.testing.assert_allclose(ana_diff, num_diff)

    def test_a0a1_one_energy_left_differential(self):
        num_diff = clgrad(self.a0a0, '__mul__', 'K.C', )(self.f)
        ana_diff = self.a0a0.left_1el_energy_gradient(self.f)
        np.testing.assert_allclose(ana_diff, num_diff)

    def test_a0a1_one_energy_right_differential(self):
        num_diff = clgrad(self.a0a1, '__mul__', 'L.C', )(self.f)
        ana_diff = self.a0a1.right_1el_energy_gradient(self.f)
        np.testing.assert_allclose(ana_diff, num_diff)

    def test_b0b1_one_energy_right_differential(self):
        num_diff = clgrad(self.b0b1, '__mul__', 'L.C', )(self.f)
        ana_diff = self.b0b1.right_1el_energy_gradient(self.f)
        np.testing.assert_allclose(ana_diff, num_diff)

    def test_a0a1_one_energy_mixed_differential(self):
        num_diff = clmixhess(self.a0a1, '__mul__', 'K.C', 'L.C')(self.f)
        ana_diff = self.a0a1.mixed_1el_energy_hessian(self.f)
        np.testing.assert_allclose(ana_diff, num_diff, rtol=DELTA, atol=DELTA)

    def test_b0b1_one_energy_mixed_differential(self):
        num_diff = clmixhess(self.b0b1, '__mul__', 'K.C', 'L.C')(self.f)
        ana_diff = self.b0b1.mixed_1el_energy_hessian(self.f)
        np.testing.assert_allclose(ana_diff, num_diff, rtol=DELTA, atol=DELTA)

###

    def test_a1a0_right_numerical_differential_overlap(self):
        KdL = clgrad(self.a1a0, 'overlap', 'L.C')()
        np.testing.assert_allclose(KdL, [[0.53, 0.0], [-0.64, 0.0]])

    def test_b1b0_right_numerical_differential_overlap(self):
        KdL = clgrad(self.b1b0, 'overlap', 'L.C')()
        np.testing.assert_allclose(KdL, [[0.53, 0.0], [-0.64, 0.0]])

    def test_a1a0_right_analytical_differential_overlap(self):
        KL10 = self.a1a0.right_overlap_gradient()
        np.testing.assert_allclose(KL10, [[0.53, 0.0], [-0.64, 0]])

    def test_b1b0_right_analytical_differential_overlap(self):
        KL10 = self.b1b0.right_overlap_gradient()
        np.testing.assert_allclose(KL10, [[0.53, 0.0], [-0.64, 0]])

    def test_a1a0_one_energy_right_differential(self):
        num_diff = clgrad(self.a1a0, '__mul__', 'L.C', )(self.f)
        ana_diff = self.a1a0.right_1el_energy_gradient(self.f)
        np.testing.assert_allclose(ana_diff, num_diff)

    def test_b1b0_one_energy_right_differential(self):
        num_diff = clgrad(self.b1b0, '__mul__', 'L.C', )(self.f)
        ana_diff = self.b1b0.right_1el_energy_gradient(self.f)
        np.testing.assert_allclose(ana_diff, num_diff)

    def test_a1a0_one_energy_mixed_differential(self):
        num_diff = clmixhess(self.a1a0, '__mul__', 'K.C', 'L.C')(self.f)
        ana_diff = self.a1a0.mixed_1el_energy_hessian(self.f)
        np.testing.assert_allclose(ana_diff, num_diff, rtol=DELTA, atol=DELTA)

    def test_b1b0_one_energy_mixed_differential(self):
        num_diff = clmixhess(self.b1b0, '__mul__', 'K.C', 'L.C')(self.f)
        ana_diff = self.b1b0.mixed_1el_energy_hessian(self.f)
        np.testing.assert_allclose(ana_diff, num_diff, rtol=DELTA, atol=DELTA)

###

    def test_a1a1_right_numerical_differential_overlap(self):
        KdL = clgrad(self.a1a1, 'overlap', 'L.C')()
        np.testing.assert_allclose(KdL, [[0.0, 0.53], [0.0, -0.64]])

    def test_b1b1_right_numerical_differential_overlap(self):
        KdL = clgrad(self.a1a1, 'overlap', 'L.C')()
        np.testing.assert_allclose(KdL, [[0.0, 0.53], [0.0, -0.64]])

    def test_a1a1_right_analytical_differential_overlap(self):
        KL11 = self.a1a1.right_overlap_gradient()
        np.testing.assert_allclose(KL11, [[0., 0.53], [0.0, -0.64]])

    def test_b1b1_right_analytical_differential_overlap(self):
        KL11 = self.b1b1.right_overlap_gradient()
        np.testing.assert_allclose(KL11, [[0., 0.53], [0.0, -0.64]])

    def test_a1a1_one_energy_right_differential(self):
        num_diff = clgrad(self.a1a1, '__mul__', 'L.C', )(self.f)
        ana_diff = self.a1a1.right_1el_energy_gradient(self.f)
        np.testing.assert_allclose(ana_diff, num_diff)

    def test_b1b1_one_energy_right_differential(self):
        num_diff = clgrad(self.b1b1, '__mul__', 'L.C', )(self.f)
        ana_diff = self.b1b1.right_1el_energy_gradient(self.f)
        np.testing.assert_allclose(ana_diff, num_diff)

    def test_a1a1_one_energy_mixed_differential(self):
        num_diff = clmixhess(self.a1a1, '__mul__', 'K.C', 'L.C')(self.f)
        ana_diff = self.a1a1.mixed_1el_energy_hessian(self.f)
        np.testing.assert_allclose(ana_diff, num_diff, rtol=DELTA, atol=DELTA)

    def test_b1b1_one_energy_mixed_differential(self):
        num_diff = clmixhess(self.b1b1, '__mul__', 'K.C', 'L.C')(self.f)
        ana_diff = self.b1b1.mixed_1el_energy_hessian(self.f)
        np.testing.assert_allclose(ana_diff, num_diff, rtol=DELTA, atol=DELTA)

###

    def test_a0a0_left_numerical_differential_overlap(self):
        dKL = clgrad(self.a0a0, 'overlap', 'K.C')().T
        np.testing.assert_allclose(dKL, [[0.76, 0.67], [0.0, 0.0]])

    def test_b0b0_left_numerical_differential_overlap(self):
        dKL = clgrad(self.b0b0, 'overlap', 'K.C')().T
        np.testing.assert_allclose(dKL, [[0.76, 0.67], [0.0, 0.0]])

    def test_a0a0_left_analytical_differential_overlap(self):
        KL00 = self.a0a0.left_overlap_gradient()
        np.testing.assert_allclose(KL00, [[0.76, 0.], [0.67, 0]])

    def test_b0b0_left_analytical_differential_overlap(self):
        KL00 = self.b0b0.left_overlap_gradient()
        np.testing.assert_allclose(KL00, [[0.76, 0], [0.67, 0]])

    def test_a0a0_one_energy_left_differential(self):
        num_diff = clgrad(self.a0a0, '__mul__', 'K.C', )(self.f)
        ana_diff = self.a0a0.left_1el_energy_gradient(self.f)
        np.testing.assert_allclose(ana_diff, num_diff)

    def test_b0b0_one_energy_left_differential(self):
        num_diff = clgrad(self.b0b0, '__mul__', 'K.C', )(self.f)
        ana_diff = self.b0b0.left_1el_energy_gradient(self.f)
        np.testing.assert_allclose(ana_diff, num_diff)
###

    def test_a0a1_left_numerical_differential_overlap_00(self):
        dKL = clgrad(self.a0a1, 'overlap', 'K.C')().T
        np.testing.assert_allclose(dKL, [[0.53, -0.64], [0.0, 0.0]])

    def test_b0b1_left_numerical_differential_overlap_00(self):
        dKL = clgrad(self.b0b1, 'overlap', 'K.C')().T
        np.testing.assert_allclose(dKL, [[0.53, -0.64], [0.0, 0.0]])

    def test_a0a1_left_analytical_differential_overlap(self):
        KL01 = self.a0a1.left_overlap_gradient()
        np.testing.assert_allclose(KL01, [[0.53, 0.], [-0.64, 0]])

    def test_b0b1_left_analytical_differential_overlap(self):
        KL01 = self.b0b1.left_overlap_gradient()
        np.testing.assert_allclose(KL01, [[0.53, 0], [-0.64, 0]])

    def test_a0a1_one_energy_left_differential(self):
        num_diff = clgrad(self.a0a1, '__mul__', 'K.C', )(self.f)
        ana_diff = self.a0a1.left_1el_energy_gradient(self.f)
        np.testing.assert_allclose(ana_diff, num_diff)

    def test_b0b1_one_energy_left_differential(self):
        num_diff = clgrad(self.b0b1, '__mul__', 'K.C', )(self.f)
        ana_diff = self.b0b1.left_1el_energy_gradient(self.f)
        np.testing.assert_allclose(ana_diff, num_diff)

###

    def test_a1a0_left_numerical_differential_overlap(self):
        dKL = clgrad(self.a1a0, 'overlap', 'K.C')().T
        np.testing.assert_allclose(dKL, [[0, .0], [0.76, 0.67]])

    def test_b1b0_left_numerical_differential_overlap(self):
        dKL = clgrad(self.b1b0, 'overlap', 'K.C')().T
        np.testing.assert_allclose(dKL, [[0, .0], [0.76, 0.67]])

    def test_a1a0_left_analytical_differential_overlap(self):
        KL10 = self.a1a0.left_overlap_gradient()
        np.testing.assert_allclose(KL10, [[0, 0.76], [0, 0.67]])

    def test_b1b0_left_analytical_differential_overlap(self):
        KL10 = self.b1b0.left_overlap_gradient()
        np.testing.assert_allclose(KL10, [[0, .76], [0., 0.67]])

    def test_a1a0_one_energy_left_differential(self):
        num_diff = clgrad(self.a1a0, '__mul__', 'K.C', )(self.f)
        ana_diff = self.a1a0.left_1el_energy_gradient(self.f)
        np.testing.assert_allclose(ana_diff, num_diff)

    def test_b1b0_one_energy_left_differential(self):
        num_diff = clgrad(self.b1b0, '__mul__', 'K.C', )(self.f)
        ana_diff = self.b1b0.left_1el_energy_gradient(self.f)
        np.testing.assert_allclose(ana_diff, num_diff)
###

    def test_a1a1_left_numerical_differential_overlap(self):
        dKL = clgrad(self.a1a1, 'overlap', 'K.C')().T
        np.testing.assert_allclose(dKL, [[0, 0], [0.53, -.64]])

    def test_b1b1_left_numerical_differential_overlap(self):
        dKL = clgrad(self.b1b1, 'overlap', 'K.C')().T
        np.testing.assert_allclose(dKL, [[0, 0], [0.53, -.64]])

    def test_a1a1_left_analytical_differential_overlap(self):
        KL11 = self.a1a1.left_overlap_gradient()
        np.testing.assert_allclose(KL11, [[0, 0.53], [0, -.64]])

    def test_b1b1_left_analytical_differential_overlap(self):
        KL11 = self.b1b1.left_overlap_gradient()
        np.testing.assert_allclose(KL11, [[0, 0.53], [0, -.64]])

    def test_a1a1_one_energy_left_differential(self):
        num_diff = clgrad(self.a1a1, '__mul__', 'K.C', )(self.f)
        ana_diff = self.a1a1.left_1el_energy_gradient(self.f)
        np.testing.assert_allclose(ana_diff, num_diff)

    def test_b1b1_one_energy_left_differential(self):
        num_diff = clgrad(self.b1b1, '__mul__', 'K.C', )(self.f)
        ana_diff = self.b1b1.left_1el_energy_gradient(self.f)
        np.testing.assert_allclose(ana_diff, num_diff)

### (0|0)

    def test_K00_L00_norm(self):
        NKL = self.K00_L00.overlap()
        self.assertAlmostEqual(NKL, 0.872356)

    def test_K00_L00_energy(self):
        HKL = self.K00_L00.oneel_energy(self.f)
        self.assertAlmostEqual(HKL, -0.5374732)

    def notest_K00_L00_overlap_numerical_right_gradient(self):
        KdL = clgrad(self.K00_L00, 'overlap', 'L.C')()
        np.testing.assert_allclose(KdL, [[1.41968, 0.0], [1.25156, 0.0]])

    def notest_K00_L00_overlap_analytial_right_gradient(self):
        KdL = self.K00_L00.right_overlap_gradient()
        np.testing.assert_allclose(KdL, [[1.41968, 0.0], [1.25156, 0.0]])

    def test_K00_L00_right_numerical_overlap_hessian(self):
        Kd2L = clhess(self.K00_L00, 'overlap', 'L.C')()
        self.assertAlmostEqual(Kd2L[0, 0, 0, 0], 1.15520, delta=DELTA)

    def test_K00_L00_right_analytical_overlap_hessian(self):
        Kd2L = self.K00_L00.right_overlap_hessian()
        self.assertAlmostEqual(Kd2L[0, 0, 0, 0], 1.15520, delta=DELTA)

    def test_K00_L00_left_right_numerical_overlap_hessian(self):
        Kd2L = clmixhess(self.K00_L00, 'overlap', 'K.C', 'L.C')()
        self.assertAlmostEqual(Kd2L[0, 0, 0, 0], 3.0232, delta=DELTA)

    def test_K00_L00_right_overlap_hessian(self):
        Kd2L_num = clhess(self.K00_L00, 'overlap', 'L.C')()
        Kd2L_ana = self.K00_L00.right_overlap_hessian()
        np.testing.assert_allclose(Kd2L_num, Kd2L_ana, rtol=DELTA, atol=DELTA)

    def test_K00_L00_mixed_overlap_hessian(self):
        dKdL_num = clmixhess(self.K00_L00, 'overlap', 'K.C', 'L.C')()
        dKdL_ana = self.K00_L00.mixed_overlap_hessian()
        np.testing.assert_allclose(dKdL_num, dKdL_ana, rtol=DELTA, atol=DELTA)

    def test_K00_L00_one_energy_right_gradient(self):
        num_diff = clgrad(self.K00_L00, '__mul__', 'L.C', )(self.f)
        ana_diff = self.K00_L00.right_1el_energy_gradient(self.f)
        np.testing.assert_allclose(ana_diff, num_diff)

    def test_K00_L00_one_energy_left_gradient(self):
        num_diff = clgrad(self.K00_L00, '__mul__', 'K.C', )(self.f)
        ana_diff = self.K00_L00.left_1el_energy_gradient(self.f)
        np.testing.assert_allclose(ana_diff, num_diff)

    def test_K00_L00_2el_energy_right_gradient(self):
        num_diff = clgrad(self.K00_L00, 'twoel_tme', 'L.C', )()
        ana_diff = self.K00_L00.right_2el_energy_gradient()
        np.testing.assert_allclose(ana_diff, num_diff)

    def test_K00_L00_2el_energy_left_gradient(self):
        num_diff = clgrad(self.K00_L00, 'twoel_tme', 'K.C', )()
        ana_diff = self.K00_L00.left_2el_energy_gradient()
        np.testing.assert_allclose(ana_diff, num_diff)

    def test_K00_L00_left_right_one_energy_hessian(self):
        dKdL_num = clmixhess(self.K00_L00, '__mul__', 'K.C', 'L.C')(self.f)
        dKdL_ana = self.K00_L00.mixed_1el_energy_hessian(self.f)
        np.testing.assert_allclose(dKdL_num, dKdL_ana, rtol=DELTA, atol=DELTA)

    def test_K00_L01_left_right_one_energy_hessian(self):
        dKdL_num = clmixhess(self.K00_L01, '__mul__', 'K.C', 'L.C')(self.f)
        dKdL_ana = self.K00_L01.mixed_1el_energy_hessian(self.f)
        np.testing.assert_allclose(dKdL_num, dKdL_ana, rtol=DELTA, atol=DELTA)

    def test_K00_L10_left_right_one_energy_hessian(self):
        dKdL_num = clmixhess(self.K00_L10, '__mul__', 'K.C', 'L.C')(self.f)
        dKdL_ana = self.K00_L10.mixed_1el_energy_hessian(self.f)
        np.testing.assert_allclose(dKdL_num, dKdL_ana, rtol=DELTA, atol=DELTA)

    def test_K00_L11_left_right_one_energy_hessian(self):
        dKdL_num = clmixhess(self.K00_L11, '__mul__', 'K.C', 'L.C')(self.f)
        dKdL_ana = self.K00_L11.mixed_1el_energy_hessian(self.f)
        np.testing.assert_allclose(dKdL_num, dKdL_ana, rtol=DELTA, atol=DELTA)

    def test_K01_L00_left_right_one_energy_hessian(self):
        dKdL_num = clmixhess(self.K01_L00, '__mul__', 'K.C', 'L.C')(self.f)
        dKdL_ana = self.K01_L00.mixed_1el_energy_hessian(self.f)
        np.testing.assert_allclose(dKdL_num, dKdL_ana, rtol=DELTA, atol=DELTA)

    def test_K01_L01_left_right_one_energy_hessian(self):
        dKdL_num = clmixhess(self.K01_L01, '__mul__', 'K.C', 'L.C')(self.f)
        dKdL_ana = self.K01_L01.mixed_1el_energy_hessian(self.f)
        np.testing.assert_allclose(dKdL_num, dKdL_ana, rtol=DELTA, atol=DELTA)

    def test_K01_L10_left_right_one_energy_hessian(self):
        dKdL_num = clmixhess(self.K01_L10, '__mul__', 'K.C', 'L.C')(self.f)
        dKdL_ana = self.K01_L10.mixed_1el_energy_hessian(self.f)
        np.testing.assert_allclose(dKdL_num, dKdL_ana, rtol=DELTA, atol=DELTA)

    def test_K01_L11_left_right_one_energy_hessian(self):
        dKdL_num = clmixhess(self.K01_L11, '__mul__', 'K.C', 'L.C')(self.f)
        dKdL_ana = self.K01_L11.mixed_1el_energy_hessian(self.f)
        np.testing.assert_allclose(dKdL_num, dKdL_ana, rtol=DELTA, atol=DELTA)

    def test_K10_L00_left_right_one_energy_hessian(self):
        dKdL_num = clmixhess(self.K10_L00, '__mul__', 'K.C', 'L.C')(self.f)
        dKdL_ana = self.K10_L00.mixed_1el_energy_hessian(self.f)
        np.testing.assert_allclose(dKdL_num, dKdL_ana, rtol=DELTA, atol=DELTA)

    def test_K10_L01_left_right_one_energy_hessian(self):
        dKdL_num = clmixhess(self.K10_L01, '__mul__', 'K.C', 'L.C')(self.f)
        dKdL_ana = self.K10_L01.mixed_1el_energy_hessian(self.f)
        np.testing.assert_allclose(dKdL_num, dKdL_ana, rtol=DELTA, atol=DELTA)

    def test_K10_L10_left_right_one_energy_hessian(self):
        dKdL_num = clmixhess(self.K10_L10, '__mul__', 'K.C', 'L.C')(self.f)
        dKdL_ana = self.K10_L10.mixed_1el_energy_hessian(self.f)
        np.testing.assert_allclose(dKdL_num, dKdL_ana, rtol=DELTA, atol=DELTA)

    def test_K10_L11_left_right_one_energy_hessian(self):
        dKdL_num = clmixhess(self.K10_L11, '__mul__', 'K.C', 'L.C')(self.f)
        dKdL_ana = self.K10_L11.mixed_1el_energy_hessian(self.f)
        np.testing.assert_allclose(dKdL_num, dKdL_ana, rtol=DELTA, atol=DELTA)

    def test_K11_L00_left_right_one_energy_hessian(self):
        dKdL_num = clmixhess(self.K11_L00, '__mul__', 'K.C', 'L.C')(self.f)
        dKdL_ana = self.K11_L00.mixed_1el_energy_hessian(self.f)
        np.testing.assert_allclose(dKdL_num, dKdL_ana, rtol=DELTA, atol=DELTA)

    def test_K11_L01_left_right_one_energy_hessian(self):
        dKdL_num = clmixhess(self.K11_L01, '__mul__', 'K.C', 'L.C')(self.f)
        dKdL_ana = self.K11_L01.mixed_1el_energy_hessian(self.f)
        np.testing.assert_allclose(dKdL_num, dKdL_ana, rtol=DELTA, atol=DELTA)

    def test_K11_L10_left_right_one_energy_hessian(self):
        dKdL_num = clmixhess(self.K11_L10, '__mul__', 'K.C', 'L.C')(self.f)
        dKdL_ana = self.K11_L10.mixed_1el_energy_hessian(self.f)
        np.testing.assert_allclose(dKdL_num, dKdL_ana, rtol=DELTA, atol=DELTA)

    def test_K11_L11_left_right_one_energy_hessian(self):
        dKdL_num = clmixhess(self.K11_L11, '__mul__', 'K.C', 'L.C')(self.f)
        dKdL_ana = self.K11_L11.mixed_1el_energy_hessian(self.f)
        np.testing.assert_allclose(dKdL_num, dKdL_ana, rtol=DELTA, atol=DELTA)

    def test_K00_L00_right_2el_energy_hessian(self):
        num_diff = clhess(self.K00_L00, 'twoel_tme', 'L.C', )()
        ana_diff = self.K00_L00.right_2el_energy_hessian()
        np.testing.assert_allclose(ana_diff, num_diff, rtol=DELTA, atol=DELTA)

    #@unittest.skip('hold')
    def test_K00_L00_mixed_2el_energy_hessian(self):
        num_diff = clmixhess(self.K00_L00, 'twoel_tme', 'K.C', 'L.C')()
        ana_diff = self.K00_L00.mixed_2el_energy_hessian()
        np.testing.assert_allclose(ana_diff, num_diff, rtol=DELTA, atol=DELTA)

class BraKetTest2(unittest.TestCase):

    def setUp(self):
        BraKet.tmpdir = os.path.join(os.path.dirname(__file__), 'test_data')
        Nod.S = init([[1.0, 0.2, 0.1], [0.2, 1.0, 0.2], [0.1, 0.2, 1.0]])
        Nod.h = init([[-0.5, 0.2, 0.1], [0.2, -0.25, 0.2], [0.1, -0.1, 1.0]])
        Nod.C = init([[0.7, 0.6, 0.5], [0.4, 0.3, 0.2]])
        self.f = (Nod.h, Nod.h)

        self.B00K00 = BraKet(Nod([0], [0]), Nod([0], [0]))
        self.B00K01 = BraKet(Nod([0], [0]), Nod([0], [1]))
        self.B00K10 = BraKet(Nod([0], [0]), Nod([1], [0]))
        self.B00K11 = BraKet(Nod([0], [0]), Nod([1], [1]))

        self.B01K00 = BraKet(Nod([0], [1]), Nod([0], [0]))
        self.B01K01 = BraKet(Nod([0], [1]), Nod([0], [1]))
        self.B01K10 = BraKet(Nod([0], [1]), Nod([1], [0]))
        self.B01K11 = BraKet(Nod([0], [1]), Nod([1], [1]))

        self.B10K00 = BraKet(Nod([1], [0]), Nod([0], [0]))
        self.B10K01 = BraKet(Nod([1], [0]), Nod([0], [0]))
        self.B10K10 = BraKet(Nod([1], [0]), Nod([1], [1]))
        self.B10K11 = BraKet(Nod([1], [0]), Nod([1], [1]))

        self.B11K00 = BraKet(Nod([1], [1]), Nod([0], [0]))
        self.B11K01 = BraKet(Nod([1], [1]), Nod([0], [1]))
        self.B11K10 = BraKet(Nod([1], [1]), Nod([1], [0]))
        self.B11K11 = BraKet(Nod([1], [1]), Nod([1], [1]))


##Gradient tests

# Right overlap gradient

    def test_00_d00(self):
        np.testing.assert_allclose(
            self.B00K00.right_overlap_gradient(),
            clgrad(self.B00K00, 'overlap', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_00_d01(self):
        np.testing.assert_allclose(
            self.B00K01.right_overlap_gradient(),
            clgrad(self.B00K01, 'overlap', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_00_d10(self):
        np.testing.assert_allclose(
            self.B00K10.right_overlap_gradient(),
            clgrad(self.B00K10, 'overlap', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_00_d11(self):
        np.testing.assert_allclose(
            self.B00K11.right_overlap_gradient(),
            clgrad(self.B00K11, 'overlap', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_01_d00(self):
        np.testing.assert_allclose(
            self.B01K00.right_overlap_gradient(),
            clgrad(self.B01K00, 'overlap', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_01_d01(self):
        np.testing.assert_allclose(
            self.B01K01.right_overlap_gradient(),
            clgrad(self.B01K01, 'overlap', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_01_d10(self):
        np.testing.assert_allclose(
            self.B01K10.right_overlap_gradient(),
            clgrad(self.B01K10, 'overlap', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_01_d11(self):
        np.testing.assert_allclose(
            self.B01K11.right_overlap_gradient(),
            clgrad(self.B01K11, 'overlap', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_10_d00(self):
        np.testing.assert_allclose(
            self.B10K00.right_overlap_gradient(),
            clgrad(self.B10K00, 'overlap', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_10_d01(self):
        np.testing.assert_allclose(
            self.B10K01.right_overlap_gradient(),
            clgrad(self.B10K01, 'overlap', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_10_d10(self):
        np.testing.assert_allclose(
            self.B10K10.right_overlap_gradient(),
            clgrad(self.B10K10, 'overlap', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_10_d11(self):
        np.testing.assert_allclose(
            self.B10K11.right_overlap_gradient(),
            clgrad(self.B10K11, 'overlap', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_11_d00(self):
        np.testing.assert_allclose(
            self.B11K00.right_overlap_gradient(),
            clgrad(self.B11K00, 'overlap', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_11_d01(self):
        np.testing.assert_allclose(
            self.B11K01.right_overlap_gradient(),
            clgrad(self.B11K01, 'overlap', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_11_d10(self):
        np.testing.assert_allclose(
            self.B11K10.right_overlap_gradient(),
            clgrad(self.B11K10, 'overlap', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_11_d11(self):
        np.testing.assert_allclose(
            self.B11K11.right_overlap_gradient(),
            clgrad(self.B11K11, 'overlap', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

# Right one-electron energy gradient

    def test_00_h_d00(self):
        np.testing.assert_allclose(
            self.B00K00.right_1el_energy_gradient(self.f),
            clgrad(self.B00K00, '__mul__', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_00_h_d01(self):
        np.testing.assert_allclose(
            self.B00K01.right_1el_energy_gradient(self.f),
            clgrad(self.B00K01, '__mul__', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_00_h_d10(self):
        np.testing.assert_allclose(
            self.B00K10.right_1el_energy_gradient(self.f),
            clgrad(self.B00K10, '__mul__', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_00_h_d11(self):
        np.testing.assert_allclose(
            self.B00K11.right_1el_energy_gradient(self.f),
            clgrad(self.B00K11, '__mul__', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_01_h_d00(self):
        np.testing.assert_allclose(
            self.B01K00.right_1el_energy_gradient(self.f),
            clgrad(self.B01K00, '__mul__', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_01_h_d01(self):
        np.testing.assert_allclose(
            self.B01K01.right_1el_energy_gradient(self.f),
            clgrad(self.B01K01, '__mul__', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_01_h_d10(self):
        np.testing.assert_allclose(
            self.B01K10.right_1el_energy_gradient(self.f),
            clgrad(self.B01K10, '__mul__', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_01_h_d11(self):
        np.testing.assert_allclose(
            self.B01K11.right_1el_energy_gradient(self.f),
            clgrad(self.B01K11, '__mul__', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_10_h_d00(self):
        np.testing.assert_allclose(
            self.B10K00.right_1el_energy_gradient(self.f),
            clgrad(self.B10K00, '__mul__', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_10_h_d01(self):
        np.testing.assert_allclose(
            self.B10K01.right_1el_energy_gradient(self.f),
            clgrad(self.B10K01, '__mul__', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_10_h_d10(self):
        np.testing.assert_allclose(
            self.B10K10.right_1el_energy_gradient(self.f),
            clgrad(self.B10K10, '__mul__', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_10_h_d11(self):
        np.testing.assert_allclose(
            self.B10K11.right_1el_energy_gradient(self.f),
            clgrad(self.B10K11, '__mul__', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_11_h_d00(self):
        np.testing.assert_allclose(
            self.B11K00.right_1el_energy_gradient(self.f),
            clgrad(self.B11K00, '__mul__', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_11_h_d01(self):
        np.testing.assert_allclose(
            self.B11K01.right_1el_energy_gradient(self.f),
            clgrad(self.B11K01, '__mul__', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_11_h_d10(self):
        np.testing.assert_allclose(
            self.B11K10.right_1el_energy_gradient(self.f),
            clgrad(self.B11K10, '__mul__', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_11_h_d11(self):
        np.testing.assert_allclose(
            self.B11K11.right_1el_energy_gradient(self.f),
            clgrad(self.B11K11, '__mul__', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

# Right two-electron energy gradient

    def test_00_g_d00(self):
        np.testing.assert_allclose(
            self.B00K00.right_2el_energy_gradient(),
            clgrad(self.B00K00, 'twoel_tme', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_00_g_d01(self):
        np.testing.assert_allclose(
            self.B00K01.right_2el_energy_gradient(),
            clgrad(self.B00K01, 'twoel_tme', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_00_g_d10(self):
        np.testing.assert_allclose(
            self.B00K10.right_2el_energy_gradient(),
            clgrad(self.B00K10, 'twoel_tme', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_00_g_d11(self):
        np.testing.assert_allclose(
            self.B00K11.right_2el_energy_gradient(),
            clgrad(self.B00K11, 'twoel_tme', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_01_g_d00(self):
        np.testing.assert_allclose(
            self.B01K00.right_2el_energy_gradient(),
            clgrad(self.B01K00, 'twoel_tme', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_01_g_d01(self):
        np.testing.assert_allclose(
            self.B01K01.right_2el_energy_gradient(),
            clgrad(self.B01K01, 'twoel_tme', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_01_g_d10(self):
        np.testing.assert_allclose(
            self.B01K10.right_2el_energy_gradient(),
            clgrad(self.B01K10, 'twoel_tme', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_01_g_d11(self):
        np.testing.assert_allclose(
            self.B01K11.right_2el_energy_gradient(),
            clgrad(self.B01K11, 'twoel_tme', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_10_g_d00(self):
        np.testing.assert_allclose(
            self.B10K00.right_2el_energy_gradient(),
            clgrad(self.B10K00, 'twoel_tme', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_10_g_d01(self):
        np.testing.assert_allclose(
            self.B10K01.right_2el_energy_gradient(),
            clgrad(self.B10K01, 'twoel_tme', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_10_g_d10(self):
        np.testing.assert_allclose(
            self.B10K10.right_2el_energy_gradient(),
            clgrad(self.B10K10, 'twoel_tme', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_10_g_d11(self):
        np.testing.assert_allclose(
            self.B10K11.right_2el_energy_gradient(),
            clgrad(self.B10K11, 'twoel_tme', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_11_g_d00(self):
        np.testing.assert_allclose(
            self.B11K00.right_2el_energy_gradient(),
            clgrad(self.B11K00, 'twoel_tme', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_11_g_d01(self):
        np.testing.assert_allclose(
            self.B11K01.right_2el_energy_gradient(),
            clgrad(self.B11K01, 'twoel_tme', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_11_g_d10(self):
        np.testing.assert_allclose(
            self.B11K10.right_2el_energy_gradient(),
            clgrad(self.B11K10, 'twoel_tme', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_11_g_d11(self):
        np.testing.assert_allclose(
            self.B11K11.right_2el_energy_gradient(),
            clgrad(self.B11K11, 'twoel_tme', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )



# Left overlap gradient

    def test_d00_00(self):
        np.testing.assert_allclose(
            self.B00K00.left_overlap_gradient(),
            clgrad(self.B00K00, 'overlap', 'K.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d00_01(self):
        np.testing.assert_allclose(
            self.B00K01.left_overlap_gradient(),
            clgrad(self.B00K01, 'overlap', 'K.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d00_10(self):
        np.testing.assert_allclose(
            self.B00K10.left_overlap_gradient(),
            clgrad(self.B00K10, 'overlap', 'K.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d00_11(self):
        np.testing.assert_allclose(
            self.B00K11.left_overlap_gradient(),
            clgrad(self.B00K11, 'overlap', 'K.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d01_00(self):
        np.testing.assert_allclose(
            self.B01K00.left_overlap_gradient(),
            clgrad(self.B01K00, 'overlap', 'K.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d01_01(self):
        np.testing.assert_allclose(
            self.B01K01.left_overlap_gradient(),
            clgrad(self.B01K01, 'overlap', 'K.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d01_10(self):
        np.testing.assert_allclose(
            self.B01K10.left_overlap_gradient(),
            clgrad(self.B01K10, 'overlap', 'K.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d01_11(self):
        np.testing.assert_allclose(
            self.B01K11.left_overlap_gradient(),
            clgrad(self.B01K11, 'overlap', 'K.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d10_00(self):
        np.testing.assert_allclose(
            self.B10K00.left_overlap_gradient(),
            clgrad(self.B10K00, 'overlap', 'K.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d10_01(self):
        np.testing.assert_allclose(
            self.B10K01.left_overlap_gradient(),
            clgrad(self.B10K01, 'overlap', 'K.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d10_10(self):
        np.testing.assert_allclose(
            self.B10K10.left_overlap_gradient(),
            clgrad(self.B10K10, 'overlap', 'K.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d10_11(self):
        np.testing.assert_allclose(
            self.B10K11.left_overlap_gradient(),
            clgrad(self.B10K11, 'overlap', 'K.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d11_00(self):
        np.testing.assert_allclose(
            self.B11K00.left_overlap_gradient(),
            clgrad(self.B11K00, 'overlap', 'K.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d11_01(self):
        np.testing.assert_allclose(
            self.B11K01.left_overlap_gradient(),
            clgrad(self.B11K01, 'overlap', 'K.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d11_10(self):
        np.testing.assert_allclose(
            self.B11K10.left_overlap_gradient(),
            clgrad(self.B11K10, 'overlap', 'K.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d11_11(self):
        np.testing.assert_allclose(
            self.B11K11.left_overlap_gradient(),
            clgrad(self.B11K11, 'overlap', 'K.C')(),
            rtol=DELTA, atol=DELTA
            )

# Left one-electron energy gradient

    def test_d00_h_00(self):
        np.testing.assert_allclose(
            self.B00K00.left_1el_energy_gradient(self.f),
            clgrad(self.B00K00, '__mul__', 'K.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_d00_h_01(self):
        np.testing.assert_allclose(
            self.B00K01.left_1el_energy_gradient(self.f),
            clgrad(self.B00K01, '__mul__', 'K.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_d00_h_10(self):
        np.testing.assert_allclose(
            self.B00K10.left_1el_energy_gradient(self.f),
            clgrad(self.B00K10, '__mul__', 'K.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_d00_h_11(self):
        np.testing.assert_allclose(
            self.B00K11.left_1el_energy_gradient(self.f),
            clgrad(self.B00K11, '__mul__', 'K.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_d01_h_00(self):
        np.testing.assert_allclose(
            self.B01K00.left_1el_energy_gradient(self.f),
            clgrad(self.B01K00, '__mul__', 'K.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_d01_h_01(self):
        np.testing.assert_allclose(
            self.B01K01.left_1el_energy_gradient(self.f),
            clgrad(self.B01K01, '__mul__', 'K.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_d01_h_10(self):
        np.testing.assert_allclose(
            self.B01K10.left_1el_energy_gradient(self.f),
            clgrad(self.B01K10, '__mul__', 'K.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_d01_h_11(self):
        np.testing.assert_allclose(
            self.B01K11.left_1el_energy_gradient(self.f),
            clgrad(self.B01K11, '__mul__', 'K.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_d10_h_00(self):
        np.testing.assert_allclose(
            self.B10K00.left_1el_energy_gradient(self.f),
            clgrad(self.B10K00, '__mul__', 'K.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_d10_h_01(self):
        np.testing.assert_allclose(
            self.B10K01.left_1el_energy_gradient(self.f),
            clgrad(self.B10K01, '__mul__', 'K.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_d10_h_10(self):
        np.testing.assert_allclose(
            self.B10K10.left_1el_energy_gradient(self.f),
            clgrad(self.B10K10, '__mul__', 'K.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_d10_h_11(self):
        np.testing.assert_allclose(
            self.B10K11.left_1el_energy_gradient(self.f),
            clgrad(self.B10K11, '__mul__', 'K.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_d11_h_00(self):
        np.testing.assert_allclose(
            self.B11K00.left_1el_energy_gradient(self.f),
            clgrad(self.B11K00, '__mul__', 'K.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_d11_h_01(self):
        np.testing.assert_allclose(
            self.B11K01.left_1el_energy_gradient(self.f),
            clgrad(self.B11K01, '__mul__', 'K.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_d11_h_10(self):
        np.testing.assert_allclose(
            self.B11K10.left_1el_energy_gradient(self.f),
            clgrad(self.B11K10, '__mul__', 'K.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_d11_h_11(self):
        np.testing.assert_allclose(
            self.B11K11.left_1el_energy_gradient(self.f),
            clgrad(self.B11K11, '__mul__', 'K.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

# Left two-electron energy gradient

    def test_d00_g_00(self):
        np.testing.assert_allclose(
            self.B00K00.left_2el_energy_gradient(),
            clgrad(self.B00K00, 'twoel_tme', 'K.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d00_g_01(self):
        np.testing.assert_allclose(
            self.B00K01.left_2el_energy_gradient(),
            clgrad(self.B00K01, 'twoel_tme', 'K.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d00_g_10(self):
        np.testing.assert_allclose(
            self.B00K10.left_2el_energy_gradient(),
            clgrad(self.B00K10, 'twoel_tme', 'K.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d00_g_11(self):
        np.testing.assert_allclose(
            self.B00K11.left_2el_energy_gradient(),
            clgrad(self.B00K11, 'twoel_tme', 'K.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d01_g_00(self):
        np.testing.assert_allclose(
            self.B01K00.left_2el_energy_gradient(),
            clgrad(self.B01K00, 'twoel_tme', 'K.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d01_g_01(self):
        np.testing.assert_allclose(
            self.B01K01.left_2el_energy_gradient(),
            clgrad(self.B01K01, 'twoel_tme', 'K.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d01_g_10(self):
        np.testing.assert_allclose(
            self.B01K10.left_2el_energy_gradient(),
            clgrad(self.B01K10, 'twoel_tme', 'K.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d01_g_11(self):
        np.testing.assert_allclose(
            self.B01K11.left_2el_energy_gradient(),
            clgrad(self.B01K11, 'twoel_tme', 'K.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d10_g_00(self):
        np.testing.assert_allclose(
            self.B10K00.left_2el_energy_gradient(),
            clgrad(self.B10K00, 'twoel_tme', 'K.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d10_g_01(self):
        np.testing.assert_allclose(
            self.B10K01.left_2el_energy_gradient(),
            clgrad(self.B10K01, 'twoel_tme', 'K.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d10_g_10(self):
        np.testing.assert_allclose(
            self.B10K10.left_2el_energy_gradient(),
            clgrad(self.B10K10, 'twoel_tme', 'K.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d10_g_11(self):
        np.testing.assert_allclose(
            self.B10K11.left_2el_energy_gradient(),
            clgrad(self.B10K11, 'twoel_tme', 'K.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d11_g_00(self):
        np.testing.assert_allclose(
            self.B11K00.left_2el_energy_gradient(),
            clgrad(self.B11K00, 'twoel_tme', 'K.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d11_g_01(self):
        np.testing.assert_allclose(
            self.B11K01.left_2el_energy_gradient(),
            clgrad(self.B11K01, 'twoel_tme', 'K.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d11_g_10(self):
        np.testing.assert_allclose(
            self.B11K10.left_2el_energy_gradient(),
            clgrad(self.B11K10, 'twoel_tme', 'K.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d11_g_11(self):
        np.testing.assert_allclose(
            self.B11K11.left_2el_energy_gradient(),
            clgrad(self.B11K11, 'twoel_tme', 'K.C')(),
            rtol=DELTA, atol=DELTA
            )


# Right overlap Hessian

    def test_00_dd00(self):
        np.testing.assert_allclose(
            self.B00K00.right_overlap_hessian(),
            clhess(self.B00K00, 'overlap', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_00_dd01(self):
        np.testing.assert_allclose(
            self.B00K01.right_overlap_hessian(),
            clhess(self.B00K01, 'overlap', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_00_dd10(self):
        np.testing.assert_allclose(
            self.B00K10.right_overlap_hessian(),
            clhess(self.B00K10, 'overlap', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_00_dd11(self):
        np.testing.assert_allclose(
            self.B00K11.right_overlap_hessian(),
            clhess(self.B00K11, 'overlap', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_01_dd00(self):
        np.testing.assert_allclose(
            self.B01K00.right_overlap_hessian(),
            clhess(self.B01K00, 'overlap', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_01_dd01(self):
        np.testing.assert_allclose(
            self.B01K01.right_overlap_hessian(),
            clhess(self.B01K01, 'overlap', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_01_dd10(self):
        np.testing.assert_allclose(
            self.B01K10.right_overlap_hessian(),
            clhess(self.B01K10, 'overlap', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_01_dd11(self):
        np.testing.assert_allclose(
            self.B01K11.right_overlap_hessian(),
            clhess(self.B01K11, 'overlap', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_10_dd00(self):
        np.testing.assert_allclose(
            self.B10K00.right_overlap_hessian(),
            clhess(self.B10K00, 'overlap', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_10_dd01(self):
        np.testing.assert_allclose(
            self.B10K01.right_overlap_hessian(),
            clhess(self.B10K01, 'overlap', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_10_dd10(self):
        np.testing.assert_allclose(
            self.B10K10.right_overlap_hessian(),
            clhess(self.B10K10, 'overlap', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_10_dd11(self):
        np.testing.assert_allclose(
            self.B10K11.right_overlap_hessian(),
            clhess(self.B10K11, 'overlap', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_11_dd00(self):
        np.testing.assert_allclose(
            self.B11K00.right_overlap_hessian(),
            clhess(self.B11K00, 'overlap', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_11_dd01(self):
        np.testing.assert_allclose(
            self.B11K01.right_overlap_hessian(),
            clhess(self.B11K01, 'overlap', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_11_dd10(self):
        np.testing.assert_allclose(
            self.B11K10.right_overlap_hessian(),
            clhess(self.B11K10, 'overlap', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_11_dd11(self):
        np.testing.assert_allclose(
            self.B11K11.right_overlap_hessian(),
            clhess(self.B11K11, 'overlap', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

# Right one-electron energy Hessian

    def test_00_h_dd00(self):
        np.testing.assert_allclose(
            self.B00K00.right_1el_energy_hessian(self.f),
            clhess(self.B00K00, '__mul__', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_00_h_dd01(self):
        np.testing.assert_allclose(
            self.B00K01.right_1el_energy_hessian(self.f),
            clhess(self.B00K01, '__mul__', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_00_h_dd10(self):
        np.testing.assert_allclose(
            self.B00K10.right_1el_energy_hessian(self.f),
            clhess(self.B00K10, '__mul__', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_00_h_dd11(self):
        np.testing.assert_allclose(
            self.B00K11.right_1el_energy_hessian(self.f),
            clhess(self.B00K11, '__mul__', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_01_h_dd00(self):
        np.testing.assert_allclose(
            self.B01K00.right_1el_energy_hessian(self.f),
            clhess(self.B01K00, '__mul__', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_01_h_dd01(self):
        np.testing.assert_allclose(
            self.B01K01.right_1el_energy_hessian(self.f),
            clhess(self.B01K01, '__mul__', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_01_h_dd10(self):
        np.testing.assert_allclose(
            self.B01K10.right_1el_energy_hessian(self.f),
            clhess(self.B01K10, '__mul__', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_01_h_dd11(self):
        np.testing.assert_allclose(
            self.B01K11.right_1el_energy_hessian(self.f),
            clhess(self.B01K11, '__mul__', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_10_h_dd00(self):
        np.testing.assert_allclose(
            self.B10K00.right_1el_energy_hessian(self.f),
            clhess(self.B10K00, '__mul__', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_10_h_dd01(self):
        np.testing.assert_allclose(
            self.B10K01.right_1el_energy_hessian(self.f),
            clhess(self.B10K01, '__mul__', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_10_h_dd10(self):
        np.testing.assert_allclose(
            self.B10K10.right_1el_energy_hessian(self.f),
            clhess(self.B10K10, '__mul__', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_10_h_dd11(self):
        np.testing.assert_allclose(
            self.B10K11.right_1el_energy_hessian(self.f),
            clhess(self.B10K11, '__mul__', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_11_h_dd00(self):
        np.testing.assert_allclose(
            self.B11K00.right_1el_energy_hessian(self.f),
            clhess(self.B11K00, '__mul__', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_11_h_dd01(self):
        np.testing.assert_allclose(
            self.B11K01.right_1el_energy_hessian(self.f),
            clhess(self.B11K01, '__mul__', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_11_h_dd10(self):
        np.testing.assert_allclose(
            self.B11K10.right_1el_energy_hessian(self.f),
            clhess(self.B11K10, '__mul__', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_11_h_dd11(self):
        np.testing.assert_allclose(
            self.B11K11.right_1el_energy_hessian(self.f),
            clhess(self.B11K11, '__mul__', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

# Right two-electron energy Hessian

    def test_00_g_dd00(self):
        np.testing.assert_allclose(
            self.B00K00.right_2el_energy_hessian(),
            clhess(self.B00K00, 'twoel_tme', 'L.C')(),
            rtol=DELTA, atol=DELTA*2
            )

    def test_00_g_dd01(self):
        np.testing.assert_allclose(
            self.B00K01.right_2el_energy_hessian(),
            clhess(self.B00K01, 'twoel_tme', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_00_g_dd10(self):
        np.testing.assert_allclose(
            self.B00K10.right_2el_energy_hessian(),
            clhess(self.B00K10, 'twoel_tme', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_00_g_dd11(self):
        np.testing.assert_allclose(
            self.B00K11.right_2el_energy_hessian(),
            clhess(self.B00K11, 'twoel_tme', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_01_g_dd00(self):
        np.testing.assert_allclose(
            self.B01K00.right_2el_energy_hessian(),
            clhess(self.B01K00, 'twoel_tme', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_01_g_dd01(self):
        np.testing.assert_allclose(
            self.B01K01.right_2el_energy_hessian(),
            clhess(self.B01K01, 'twoel_tme', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_01_g_dd10(self):
        np.testing.assert_allclose(
            self.B01K10.right_2el_energy_hessian(),
            clhess(self.B01K10, 'twoel_tme', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_01_g_dd11(self):
        np.testing.assert_allclose(
            self.B01K11.right_2el_energy_hessian(),
            clhess(self.B01K11, 'twoel_tme', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_10_g_dd00(self):
        np.testing.assert_allclose(
            self.B10K00.right_2el_energy_hessian(),
            clhess(self.B10K00, 'twoel_tme', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_10_g_dd01(self):
        np.testing.assert_allclose(
            self.B10K01.right_2el_energy_hessian(),
            clhess(self.B10K01, 'twoel_tme', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_10_g_dd10(self):
        np.testing.assert_allclose(
            self.B10K10.right_2el_energy_hessian(),
            clhess(self.B10K10, 'twoel_tme', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_10_g_dd11(self):
        np.testing.assert_allclose(
            self.B10K11.right_2el_energy_hessian(),
            clhess(self.B10K11, 'twoel_tme', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_11_g_dd00(self):
        np.testing.assert_allclose(
            self.B11K00.right_2el_energy_hessian(),
            clhess(self.B11K00, 'twoel_tme', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_11_g_dd01(self):
        np.testing.assert_allclose(
            self.B11K01.right_2el_energy_hessian(),
            clhess(self.B11K01, 'twoel_tme', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_11_g_dd10(self):
        np.testing.assert_allclose(
            self.B11K10.right_2el_energy_hessian(),
            clhess(self.B11K10, 'twoel_tme', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_11_g_dd11(self):
        np.testing.assert_allclose(
            self.B11K11.right_2el_energy_hessian(),
            clhess(self.B11K11, 'twoel_tme', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

# Mixed overlap Hessian

    def test_d00_d00(self):
        np.testing.assert_allclose(
            self.B00K00.mixed_overlap_hessian(),
            clmixhess(self.B00K00, 'overlap', 'K.C', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d00_d01(self):
        np.testing.assert_allclose(
            self.B00K01.mixed_overlap_hessian(),
            clmixhess(self.B00K01, 'overlap', 'K.C', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d00_d10(self):
        np.testing.assert_allclose(
            self.B00K10.mixed_overlap_hessian(),
            clmixhess(self.B00K10, 'overlap', 'K.C', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d00_d11(self):
        np.testing.assert_allclose(
            self.B00K11.mixed_overlap_hessian(),
            clmixhess(self.B00K11, 'overlap', 'K.C', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d01_d00(self):
        np.testing.assert_allclose(
            self.B01K00.mixed_overlap_hessian(),
            clmixhess(self.B01K00, 'overlap', 'K.C', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d01_d01(self):
        np.testing.assert_allclose(
            self.B01K01.mixed_overlap_hessian(),
            clmixhess(self.B01K01, 'overlap', 'K.C', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d01_d10(self):
        np.testing.assert_allclose(
            self.B01K10.mixed_overlap_hessian(),
            clmixhess(self.B01K10, 'overlap', 'K.C', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d01_d11(self):
        np.testing.assert_allclose(
            self.B01K11.mixed_overlap_hessian(),
            clmixhess(self.B01K11, 'overlap', 'K.C', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d10_d00(self):
        np.testing.assert_allclose(
            self.B10K00.mixed_overlap_hessian(),
            clmixhess(self.B10K00, 'overlap', 'K.C', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d10_d01(self):
        np.testing.assert_allclose(
            self.B10K01.mixed_overlap_hessian(),
            clmixhess(self.B10K01, 'overlap', 'K.C', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d10_d10(self):
        np.testing.assert_allclose(
            self.B10K10.mixed_overlap_hessian(),
            clmixhess(self.B10K10, 'overlap', 'K.C', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d10_d11(self):
        np.testing.assert_allclose(
            self.B10K11.mixed_overlap_hessian(),
            clmixhess(self.B10K11, 'overlap', 'K.C', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d11_d00(self):
        np.testing.assert_allclose(
            self.B11K00.mixed_overlap_hessian(),
            clmixhess(self.B11K00, 'overlap', 'K.C', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d11_d01(self):
        np.testing.assert_allclose(
            self.B11K01.mixed_overlap_hessian(),
            clmixhess(self.B11K01, 'overlap', 'K.C', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d11_d10(self):
        np.testing.assert_allclose(
            self.B11K10.mixed_overlap_hessian(),
            clmixhess(self.B11K10, 'overlap', 'K.C', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d11_d11(self):
        np.testing.assert_allclose(
            self.B11K11.mixed_overlap_hessian(),
            clmixhess(self.B11K11, 'overlap', 'K.C', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

# Mixed one-electron energy Hessian

    def test_d00_h_d00(self):
        np.testing.assert_allclose(
            self.B00K00.mixed_1el_energy_hessian(self.f),
            clmixhess(self.B00K00, '__mul__', 'K.C', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_d00_h_d01(self):
        np.testing.assert_allclose(
            self.B00K01.mixed_1el_energy_hessian(self.f),
            clmixhess(self.B00K01, '__mul__', 'K.C', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_d00_h_d10(self):
        np.testing.assert_allclose(
            self.B00K10.mixed_1el_energy_hessian(self.f),
            clmixhess(self.B00K10, '__mul__', 'K.C', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_d00_h_d11(self):
        np.testing.assert_allclose(
            self.B00K11.mixed_1el_energy_hessian(self.f),
            clmixhess(self.B00K11, '__mul__', 'K.C', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_d01_h_d00(self):
        np.testing.assert_allclose(
            self.B01K00.mixed_1el_energy_hessian(self.f),
            clmixhess(self.B01K00, '__mul__', 'K.C', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_d01_h_d01(self):
        np.testing.assert_allclose(
            self.B01K01.mixed_1el_energy_hessian(self.f),
            clmixhess(self.B01K01, '__mul__', 'K.C', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_d01_h_d10(self):
        np.testing.assert_allclose(
            self.B01K10.mixed_1el_energy_hessian(self.f),
            clmixhess(self.B01K10, '__mul__', 'K.C', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_d01_h_d11(self):
        np.testing.assert_allclose(
            self.B01K11.mixed_1el_energy_hessian(self.f),
            clmixhess(self.B01K11, '__mul__', 'K.C', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_d10_h_d00(self):
        np.testing.assert_allclose(
            self.B10K00.mixed_1el_energy_hessian(self.f),
            clmixhess(self.B10K00, '__mul__', 'K.C', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_d10_h_d01(self):
        np.testing.assert_allclose(
            self.B10K01.mixed_1el_energy_hessian(self.f),
            clmixhess(self.B10K01, '__mul__', 'K.C', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_d10_h_d10(self):
        np.testing.assert_allclose(
            self.B10K10.mixed_1el_energy_hessian(self.f),
            clmixhess(self.B10K10, '__mul__', 'K.C', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_d10_h_d11(self):
        np.testing.assert_allclose(
            self.B10K11.mixed_1el_energy_hessian(self.f),
            clmixhess(self.B10K11, '__mul__', 'K.C', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_d11_h_d00(self):
        np.testing.assert_allclose(
            self.B11K00.mixed_1el_energy_hessian(self.f),
            clmixhess(self.B11K00, '__mul__', 'K.C', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_d11_h_d01(self):
        np.testing.assert_allclose(
            self.B11K01.mixed_1el_energy_hessian(self.f),
            clmixhess(self.B11K01, '__mul__', 'K.C', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_d11_h_d10(self):
        np.testing.assert_allclose(
            self.B11K10.mixed_1el_energy_hessian(self.f),
            clmixhess(self.B11K10, '__mul__', 'K.C', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_d11_h_d11(self):
        np.testing.assert_allclose(
            self.B11K11.mixed_1el_energy_hessian(self.f),
            clmixhess(self.B11K11, '__mul__', 'K.C', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

# Mixed two-electron energy Hessian

    def test_d00_g_d00(self):
        np.testing.assert_allclose(
            self.B00K00.mixed_2el_energy_hessian(),
            clmixhess(self.B00K00, 'twoel_tme', 'K.C', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d00_g_d01(self):
        np.testing.assert_allclose(
            self.B00K01.mixed_2el_energy_hessian(),
            clmixhess(self.B00K01, 'twoel_tme', 'K.C', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d00_g_d10(self):
        np.testing.assert_allclose(
            self.B00K10.mixed_2el_energy_hessian(),
            clmixhess(self.B00K10, 'twoel_tme', 'K.C', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d00_g_d11(self):
        np.testing.assert_allclose(
            self.B00K11.mixed_2el_energy_hessian(),
            clmixhess(self.B00K11, 'twoel_tme', 'K.C', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d01_g_d00(self):
        np.testing.assert_allclose(
            self.B01K00.mixed_2el_energy_hessian(),
            clmixhess(self.B01K00, 'twoel_tme', 'K.C', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d01_g_d01(self):
        np.testing.assert_allclose(
            self.B01K01.mixed_2el_energy_hessian(),
            clmixhess(self.B01K01, 'twoel_tme', 'K.C', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d01_g_d10(self):
        np.testing.assert_allclose(
            self.B01K10.mixed_2el_energy_hessian(),
            clmixhess(self.B01K10, 'twoel_tme', 'K.C', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d01_g_d11(self):
        np.testing.assert_allclose(
            self.B01K11.mixed_2el_energy_hessian(),
            clmixhess(self.B01K11, 'twoel_tme', 'K.C', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d10_g_d00(self):
        np.testing.assert_allclose(
            self.B10K00.mixed_2el_energy_hessian(),
            clmixhess(self.B10K00, 'twoel_tme', 'K.C', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d10_g_d01(self):
        np.testing.assert_allclose(
            self.B10K01.mixed_2el_energy_hessian(),
            clmixhess(self.B10K01, 'twoel_tme', 'K.C', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d10_g_d10(self):
        np.testing.assert_allclose(
            self.B10K10.mixed_2el_energy_hessian(),
            clmixhess(self.B10K10, 'twoel_tme', 'K.C', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d10_g_d11(self):
        np.testing.assert_allclose(
            self.B10K11.mixed_2el_energy_hessian(),
            clmixhess(self.B10K11, 'twoel_tme', 'K.C', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d11_g_d00(self):
        np.testing.assert_allclose(
            self.B11K00.mixed_2el_energy_hessian(),
            clmixhess(self.B11K00, 'twoel_tme', 'K.C', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d11_g_d01(self):
        np.testing.assert_allclose(
            self.B11K01.mixed_2el_energy_hessian(),
            clmixhess(self.B11K01, 'twoel_tme', 'K.C', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d11_g_d10(self):
        np.testing.assert_allclose(
            self.B11K10.mixed_2el_energy_hessian(),
            clmixhess(self.B11K10, 'twoel_tme', 'K.C', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d11_g_d11(self):
        np.testing.assert_allclose(
            self.B11K11.mixed_2el_energy_hessian(),
            clmixhess(self.B11K11, 'twoel_tme', 'K.C', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

####
class BraKetTest3(unittest.TestCase):

    def setUp(self):
        BraKet.tmpdir = os.path.join(os.path.dirname(__file__), 'test_data')
        Nod.S = init([[1.0, 0.2, 0.1], [0.2, 1.0, 0.2], [0.1, 0.2, 1.0]])
        Nod.h = init([[-0.5, 0.2, 0.1], [0.2, -0.25, 0.2], [0.1, -0.1, 1.0]])
        Nod.C = init([[0.7, 0.6, 0.5], [0.4, 0.3, 0.2]])
        self.f = (Nod.h, Nod.h)

        self.B010K010 = BraKet(Nod([0, 1], [0]), Nod([0, 1], [0]))
        self.B010K011 = BraKet(Nod([0, 1], [0]), Nod([0, 1], [1]))
        self.B011K010 = BraKet(Nod([0, 1], [1]), Nod([0, 1], [0]))
        self.B011K011 = BraKet(Nod([0, 1], [1]), Nod([0, 1], [1]))

# right overlap hessian

    def test_010_dd010(self):
        np.testing.assert_allclose(
            self.B010K010.right_overlap_hessian(),
            clhess(self.B010K010, 'overlap', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_010_dd011(self):
        np.testing.assert_allclose(
            self.B010K011.right_overlap_hessian(),
            clhess(self.B010K011, 'overlap', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_011_dd010(self):
        np.testing.assert_allclose(
            self.B011K010.right_overlap_hessian(),
            clhess(self.B011K010, 'overlap', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_011_dd011(self):
        np.testing.assert_allclose(
            self.B011K011.right_overlap_hessian(),
            clhess(self.B011K011, 'overlap', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

# right 1el hessian

    def test_010_h_dd010(self):
        np.testing.assert_allclose(
            self.B010K010.right_1el_energy_hessian(self.f),
            clhess(self.B010K010, '__mul__', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_010_h_dd011(self):
        np.testing.assert_allclose(
            self.B010K011.right_1el_energy_hessian(self.f),
            clhess(self.B010K011, '__mul__', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_011_h_dd010(self):
        np.testing.assert_allclose(
            self.B011K010.right_1el_energy_hessian(self.f),
            clhess(self.B011K010, '__mul__', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_011_h_dd011(self):
        np.testing.assert_allclose(
            self.B011K011.right_1el_energy_hessian(self.f),
            clhess(self.B011K011, '__mul__', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

### right 2el hessian

    def test_010_g_dd010(self):
        np.testing.assert_allclose(
            self.B010K010.right_2el_energy_hessian(),
            clhess(self.B010K010, 'twoel_tme', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_010_g_dd011(self):
        np.testing.assert_allclose(
            self.B010K011.right_2el_energy_hessian(),
            clhess(self.B010K011, 'twoel_tme', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_011_g_dd010(self):
        np.testing.assert_allclose(
            self.B011K010.right_2el_energy_hessian(),
            clhess(self.B011K010, 'twoel_tme', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_011_g_dd011(self):
        np.testing.assert_allclose(
            self.B011K011.right_2el_energy_hessian(),
            clhess(self.B011K011, 'twoel_tme', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

### mixed overlap hessian

    def test_d010_d010(self):
        np.testing.assert_allclose(
            self.B010K010.mixed_overlap_hessian(),
            clmixhess(self.B010K010, 'overlap', 'K.C', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d010_d011(self):
        np.testing.assert_allclose(
            self.B010K011.mixed_overlap_hessian(),
            clmixhess(self.B010K011, 'overlap', 'K.C', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d011_d010(self):
        np.testing.assert_allclose(
            self.B011K010.mixed_overlap_hessian(),
            clmixhess(self.B011K010, 'overlap', 'K.C', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d011_d011(self):
        np.testing.assert_allclose(
            self.B011K011.mixed_overlap_hessian(),
            clmixhess(self.B011K011, 'overlap', 'K.C', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

### mixed 1el hessian

    def test_d010_h_d010(self):
        np.testing.assert_allclose(
            self.B010K010.mixed_1el_energy_hessian(self.f),
            clmixhess(self.B010K010, '__mul__', 'K.C', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_d010_h_d011(self):
        np.testing.assert_allclose(
            self.B010K011.mixed_1el_energy_hessian(self.f),
            clmixhess(self.B010K011, '__mul__', 'K.C', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_d011_h_d010(self):
        np.testing.assert_allclose(
            self.B011K010.mixed_1el_energy_hessian(self.f),
            clmixhess(self.B011K010, '__mul__', 'K.C', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

    def test_d011_h_d011(self):
        np.testing.assert_allclose(
            self.B011K011.mixed_1el_energy_hessian(self.f),
            clmixhess(self.B011K011, '__mul__', 'K.C', 'L.C')(self.f),
            rtol=DELTA, atol=DELTA
            )

### mixed 2el hessian

    def test_d010_h_d010(self):
        np.testing.assert_allclose(
            self.B010K010.mixed_2el_energy_hessian(),
            clmixhess(self.B010K010, 'twoel_tme', 'K.C', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d010_h_d011(self):
        np.testing.assert_allclose(
            self.B010K011.mixed_2el_energy_hessian(),
            clmixhess(self.B010K011, 'twoel_tme', 'K.C', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d011_h_d010(self):
        np.testing.assert_allclose(
            self.B011K010.mixed_2el_energy_hessian(),
            clmixhess(self.B011K010, 'twoel_tme', 'K.C', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )

    def test_d011_h_d011(self):
        np.testing.assert_allclose(
            self.B011K011.mixed_2el_energy_hessian(),
            clmixhess(self.B011K011, 'twoel_tme', 'K.C', 'L.C')(),
            rtol=DELTA, atol=DELTA
            )
            
if __name__ == "__main__":
    unittest.main()
