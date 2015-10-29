import unittest
import numpy as np
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
        Nod.S = init([[1.0, 0.1], [0.1, 1.0]])
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
        KL00 = self.a0a0.right_orbital_gradient()
        np.testing.assert_allclose(KL00, [[0.76, 0.0], [0.67, 0]])

    def test_b0b0_right_numerical_differential_overlap(self):
        KdL = clgrad(self.b0b0, 'overlap', 'L.C')()
        np.testing.assert_allclose(KdL, [[0.76, 0.0], [0.67, 0.0]])

    def test_b0b0_right_analytical_differential_overlap(self):
        KL00 = self.b0b0.right_orbital_gradient()
        np.testing.assert_allclose(KL00, [[0.76, 0.0], [0.67, 0]])


###

    def test_a0a1_right_numerical_differential_overlap(self):
        KdL = clgrad(self.a0a1, 'overlap', 'L.C')()
        np.testing.assert_allclose(KdL, [[0.0, 0.76], [0.0, 0.67]])

    def test_b0b1_right_numerical_differential_overlap(self):
        KdL = clgrad(self.b0b1, 'overlap', 'L.C')()
        np.testing.assert_allclose(KdL, [[0.0, 0.76], [0.0, 0.67]])

    def test_01_right_analytical_differential_overlap(self):
        KL01 = self.a0a1.right_orbital_gradient()
        np.testing.assert_allclose(KL01, [[0, 0.76], [0.0, 0.67]])

    def test_01_right_analytical_differential_overlap(self):
        KL01 = self.b0b1.right_orbital_gradient()
        np.testing.assert_allclose(KL01, [[0, 0.76], [0.0, 0.67]])

###

    def test_a1a0_right_numerical_differential_overlap(self):
        KdL = clgrad(self.a1a0, 'overlap', 'L.C')()
        np.testing.assert_allclose(KdL, [[0.53, 0.0], [-0.64, 0.0]])

    def test_b1b0_right_numerical_differential_overlap(self):
        KdL = clgrad(self.b1b0, 'overlap', 'L.C')()
        np.testing.assert_allclose(KdL, [[0.53, 0.0], [-0.64, 0.0]])

    def test_a1a0_right_analytical_differential_overlap(self):
        KL10 = self.a1a0.right_orbital_gradient()
        np.testing.assert_allclose(KL10, [[0.53, 0.0], [-0.64, 0]])

    def test_b1b0_right_analytical_differential_overlap(self):
        KL10 = self.b1b0.right_orbital_gradient()
        np.testing.assert_allclose(KL10, [[0.53, 0.0], [-0.64, 0]])

###

    def test_a1a1_right_numerical_differential_overlap(self):
        KdL = clgrad(self.a1a1, 'overlap', 'L.C')()
        np.testing.assert_allclose(KdL, [[0.0, 0.53], [0.0, -0.64]])

    def test_b1b1_right_numerical_differential_overlap(self):
        KdL = clgrad(self.a1a1, 'overlap', 'L.C')()
        np.testing.assert_allclose(KdL, [[0.0, 0.53], [0.0, -0.64]])

    def test_a1a1_right_analytical_differential_overlap(self):
        KL11 = self.a1a1.right_orbital_gradient()
        np.testing.assert_allclose(KL11, [[0., 0.53], [0.0, -0.64]])

    def test_b1b1_right_analytical_differential_overlap(self):
        KL11 = self.b1b1.right_orbital_gradient()
        np.testing.assert_allclose(KL11, [[0., 0.53], [0.0, -0.64]])

###

    def test_a0a0_left_numerical_differential_overlap(self):
        dKL = clgrad(self.a0a0, 'overlap', 'K.C')().T
        np.testing.assert_allclose(dKL, [[0.76, 0.67], [0.0, 0.0]])

    def test_b0b0_left_numerical_differential_overlap(self):
        dKL = clgrad(self.b0b0, 'overlap', 'K.C')().T
        np.testing.assert_allclose(dKL, [[0.76, 0.67], [0.0, 0.0]])

    def test_a0a0_left_analytical_differential_overlap(self):
        KL00 = self.a0a0.left_orbital_gradient()
        np.testing.assert_allclose(KL00, [[0.76, 0.67], [0, 0]])

    def test_b0b0_left_analytical_differential_overlap(self):
        KL00 = self.b0b0.left_orbital_gradient()
        np.testing.assert_allclose(KL00, [[0.76, 0.67], [0, 0]])

###

    def test_a0a1_left_numerical_differential_overlap_00(self):
        dKL = clgrad(self.a0a1, 'overlap', 'K.C')().T
        np.testing.assert_allclose(dKL, [[0.53, -0.64], [0.0, 0.0]])

    def test_b0b1_left_numerical_differential_overlap_00(self):
        dKL = clgrad(self.b0b1, 'overlap', 'K.C')().T
        np.testing.assert_allclose(dKL, [[0.53, -0.64], [0.0, 0.0]])

    def test_a0a1_left_analytical_differential_overlap(self):
        KL01 = self.a0a1.left_orbital_gradient()
        np.testing.assert_allclose(KL01, [[0.53, -0.64], [0, 0]])

    def test_b0b1_left_analytical_differential_overlap(self):
        KL01 = self.b0b1.left_orbital_gradient()
        np.testing.assert_allclose(KL01, [[0.53, -0.64], [0, 0]])

###

    def test_a1a0_left_numerical_differential_overlap(self):
        dKL = clgrad(self.a1a0, 'overlap', 'K.C')().T
        np.testing.assert_allclose(dKL, [[0, .0], [0.76, 0.67]])

    def test_b1b0_left_numerical_differential_overlap(self):
        dKL = clgrad(self.b1b0, 'overlap', 'K.C')().T
        np.testing.assert_allclose(dKL, [[0, .0], [0.76, 0.67]])

    def test_a1a0_left_analytical_differential_overlap(self):
        KL10 = self.a1a0.left_orbital_gradient()
        np.testing.assert_allclose(KL10, [[0, .0], [0.76, 0.67]])

    def test_b1b0_left_analytical_differential_overlap(self):
        KL10 = self.b1b0.left_orbital_gradient()
        np.testing.assert_allclose(KL10, [[0, .0], [0.76, 0.67]])

###

    def test_a1a1_left_numerical_differential_overlap(self):
        dKL = clgrad(self.a1a1, 'overlap', 'K.C')().T
        np.testing.assert_allclose(dKL, [[0, 0], [0.53, -.64]])

    def test_b1b1_left_numerical_differential_overlap(self):
        dKL = clgrad(self.b1b1, 'overlap', 'K.C')().T
        np.testing.assert_allclose(dKL, [[0, 0], [0.53, -.64]])

    def test_a1a1_left_analytical_differential_overlap(self):
        KL11 = self.a1a1.left_orbital_gradient()
        np.testing.assert_allclose(KL11, [[0, 0], [0.53, -.64]])

    def test_b1b1_left_analytical_differential_overlap(self):
        KL11 = self.b1b1.left_orbital_gradient()
        np.testing.assert_allclose(KL11, [[0, 0], [0.53, -.64]])

### (0|0)

    def test_K00_L00_norm(self):
        NKL = self.K00_L00.overlap()
        self.assertAlmostEqual(NKL, 0.872356)


    def notest_K00_L00_overlap_numerical_right_gradient(self):
        KdL = clgrad(self.K00_L00, 'overlap', 'L.C')()
        np.testing.assert_allclose(KdL, [[1.41968, 0.0], [1.25156, 0.0]])

    def notest_K00_L00_overlap_analytial_right_gradient(self):
        KdL = self.K00_L00.right_orbital_gradient()
        np.testing.assert_allclose(KdL, [[1.41968, 0.0], [1.25156, 0.0]])

    def test_K00_L00_right_numerical_overlap_hessian(self):
        Kd2L = clhess(self.K00_L00, 'overlap', 'L.C')()
        self.assertAlmostEqual(Kd2L[0, 0, 0, 0], 1.15520, delta=DELTA)

    def test_K00_L00_right_analytical_overlap_hessian(self):
        Kd2L = self.K00_L00.right_orbital_hessian()
        self.assertAlmostEqual(Kd2L[0, 0, 0, 0], 1.15520, delta=DELTA)

    def test_K00_L00_right_analytical_vs_numerical_overlap_hessian(self):
        Kd2L_num = clhess(self.K00_L00, 'overlap', 'L.C')()
        Kd2L_ana = self.K00_L00.right_orbital_hessian()
        np.testing.assert_allclose(Kd2L_num, Kd2L_ana, rtol=DELTA, atol=DELTA)

    def test_K00_L00_left_right_numerical_overlap_hessian(self):
        Kd2L = clmixhess(self.K00_L00, 'overlap', 'K.C', 'L.C')()
        self.assertAlmostEqual(Kd2L[0, 0, 0, 0], 3.0232, delta=DELTA)

    def test_K00_L00_left_right_analytical_vs_numerical_overlap_hessian(self):
        dKdL_num = clmixhess(self.K00_L00, 'overlap', 'K.C', 'L.C')()
        dKdL_ana = self.K00_L00.mixed_orbital_hessian()
        print dKdL_num.view(matrix)
        print dKdL_ana
        np.testing.assert_allclose(dKdL_num, dKdL_ana, rtol=DELTA, atol=DELTA)


if __name__ == "__main__":
    unittest.main()
