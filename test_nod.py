import unittest
import numpy as np
from vb import Nod, DKL, NodPair
from vb import structure, StructError
from daltools.util.full import init


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

    def test_num_diff_right(self):
        K = Nod([0], [])
        L = Nod([0], [])
        L.C = Nod.C.copy()
        self.assertIsNot(K.C, L.C)
        delta = 1e-3
        L.C[0, 0] += delta/2
        KLp = K*L
        L.C[0, 0] -= delta
        KLm = K*L
        L.C[0, 0] += delta/2
        KL00 = (KLp - KLm)/ delta
        self.assertAlmostEqual(KL00, 0.77)
        
class NodPairTest(unittest.TestCase):

    def setUp(self):
        Nod.S = init([[1.0, 0.1], [0.1, 1.0]])
        Nod.C = init([[0.7, 0.6], [0.6, -0.7]])
        self.a0a0 = NodPair(Nod([0], []), Nod([0], []))
        self.a0a1 = NodPair(Nod([0], []), Nod([1], []))
        self.a1a0 = NodPair(Nod([1], []), Nod([0], []))
        self.a1a1 = NodPair(Nod([1], []), Nod([1], []))
        self.a0b0_a0b0 = NodPair(Nod([0], [0]), Nod([0], [0]))

    def tearDown(self):
        pass

    def test_nod_pair_setup(self):
        self.assertAlmostEqual(self.a0a0.K*self.a0a0.L, self.a0a0.overlap())

    def test_00_right_numerical_differential_overlap_00(self):
        K0L0_00 = self.a0a0.right_numerical_gradient(0, 0)
        #( 1 .1)(.7 .6)
        self.assertAlmostEqual(K0L0_00, 0.76)

    def test_00_right_numerical_differential_overlap_01(self):
        K0L0_01 = self.a0a0.right_numerical_gradient(0, 1)
        self.assertAlmostEqual(K0L0_01, 0.0)

    def test_00_right_numerical_differential_overlap_10(self):
        K0L0_10 = self.a0a0.right_numerical_gradient(1, 0)
        # (.1 1)(.7 .6) = .67
        self.assertAlmostEqual(K0L0_10, 0.67)

    def test_00_right_numerical_differential_overlap_11(self):
        K0L0_11 = self.a0a0.right_numerical_gradient(1, 1)
        self.assertAlmostEqual(K0L0_11, 0.0)

    def test_00_right_analytical_differential_overlap(self):
        KL00 = self.a0a0.right_orbital_gradient()
        np.testing.assert_allclose(KL00, [[0.76, 0.67], [0., 0]])

###

    def test_01_right_numerical_differential_overlap_00(self):
        K0L1_00 = self.a0a1.right_numerical_gradient(0, 0)
        self.assertAlmostEqual(K0L1_00, 0.0)

    def test_01_right_numerical_differential_overlap_01(self):
        K0L1_01 = self.a0a1.right_numerical_gradient(0, 1)
        # (1 .1)(.7 .6) = .76
        self.assertAlmostEqual(K0L1_01, 0.76)

    def test_01_right_numerical_differential_overlap_10(self):
        K0L1_10 = self.a0a1.right_numerical_gradient(1, 0)
        self.assertAlmostEqual(K0L1_10, 0.0)

    def test_01_right_numerical_differential_overlap_11(self):
        K0L1_11 = self.a0a1.right_numerical_gradient(1, 1)
        # (.1 1)(.7 .6) = 0.67
        self.assertAlmostEqual(K0L1_11, 0.67)

    def test_01_right_analytical_differential_overlap(self):
        KL01 = self.a0a1.right_orbital_gradient()
        np.testing.assert_allclose(KL01, [[0, .0], [0.76, 0.67]])

###

    def test_10_right_numerical_differential_overlap_00(self):
        K1L0_00 = self.a1a0.right_numerical_gradient(0, 0)
        self.assertAlmostEqual(K1L0_00, 0.53)

    def test_10_right_numerical_differential_overlap_01(self):
        K1L0_01 = self.a1a0.right_numerical_gradient(0, 1)
        self.assertAlmostEqual(K1L0_01, 0.0)

    def test_10_right_numerical_differential_overlap_10(self):
        K1L0_10 = self.a1a0.right_numerical_gradient(1, 0)
        self.assertAlmostEqual(K1L0_10, -0.64)

    def test_10_right_numerical_differential_overlap_11(self):
        K1L0_11 = self.a1a0.right_numerical_gradient(1, 1)
        self.assertAlmostEqual(K1L0_11, 0.0)

    def test_10_right_analytical_differential_overlap(self):
        KL10 = self.a1a0.right_orbital_gradient()
        np.testing.assert_allclose(KL10, [[0.53, -.64], [0, 0]])

###

    def test_11_right_numerical_differential_overlap_00(self):
        K1L1_00 = self.a1a1.right_numerical_gradient(0, 0)
        self.assertAlmostEqual(K1L1_00, 0.0)

    def test_11_right_numerical_differential_overlap_01(self):
        K1L1_01 = self.a1a1.right_numerical_gradient(0, 1)
        #(1 .1) (.6 -.7) = .53
        self.assertAlmostEqual(K1L1_01, 0.53)

    def test_11_right_numerical_differential_overlap_10(self):
        K1L1_10 = self.a1a1.right_numerical_gradient(1, 0)
        # (.1 1) (.6 -.7) = -.64
        self.assertAlmostEqual(K1L1_10, 0.0)

    def test_11_right_numerical_differential_overlap_11(self):
        K1L1_11 = self.a1a1.right_numerical_gradient(1, 1)
        # (.1 1) (.6 -.7) = -.64
        self.assertAlmostEqual(K1L1_11, -0.64)

    def test_11_right_analytical_differential_overlap(self):
        KL11 = self.a1a1.right_orbital_gradient()
        np.testing.assert_allclose(KL11, [[0., 0.], [0.53, -.64]])

###

    def test_00_left_numerical_differential_overlap_00(self):
        K0L0_00 = self.a0a0.left_numerical_gradient(0, 0)
        #( 1 .1)(.7 .6)
        self.assertAlmostEqual(K0L0_00, 0.76)

    def test_00_left_numerical_differential_overlap_01(self):
        K0L0_01 = self.a0a0.left_numerical_gradient(0, 1)
        self.assertAlmostEqual(K0L0_01, 0.0)

    def test_00_left_numerical_differential_overlap_10(self):
        K0L0_10 = self.a0a0.left_numerical_gradient(1, 0)
        # (.1 1)(.7 .6) = .67
        self.assertAlmostEqual(K0L0_10, 0.67)

    def test_00_left_numerical_differential_overlap_11(self):
        K0L0_11 = self.a0a0.left_numerical_gradient(1, 1)
        self.assertAlmostEqual(K0L0_11, 0.0)

    def test_00_left_analytical_differential_overlap(self):
        KL00 = self.a0a0.left_orbital_gradient()
        np.testing.assert_allclose(KL00, [[0.76, 0.67], [0, 0]])

###

    def test_01_left_numerical_differential_overlap_00(self):
        K0L1_00 = self.a0a1.left_numerical_gradient(0, 0)
        self.assertAlmostEqual(K0L1_00, 0.53)

    def test_01_left_numerical_differential_overlap_01(self):
        K0L1_01 = self.a0a1.left_numerical_gradient(0, 1)
        # (1 .1)(.7 .6) = .76
        self.assertAlmostEqual(K0L1_01, 0.0)

    def test_01_left_numerical_differential_overlap_10(self):
        K0L1_10 = self.a0a1.left_numerical_gradient(1, 0)
        self.assertAlmostEqual(K0L1_10, -0.64)

    def test_01_left_numerical_differential_overlap_11(self):
        K0L1_11 = self.a0a1.left_numerical_gradient(1, 1)
        # (.1 1)(.7 .6) = 0.67
        self.assertAlmostEqual(K0L1_11, 0.)

    def test_01_left_analytical_differential_overlap(self):
        KL01 = self.a0a1.left_orbital_gradient()
        np.testing.assert_allclose(KL01, [[0.53, -0.64], [0, 0]])

###

    def test_10_left_numerical_differential_overlap_00(self):
        K1L0_00 = self.a1a0.left_numerical_gradient(0, 0)
        self.assertAlmostEqual(K1L0_00, 0.)

    def test_10_left_numerical_differential_overlap_01(self):
        K1L0_01 = self.a1a0.left_numerical_gradient(0, 1)
        self.assertAlmostEqual(K1L0_01, 0.76)

    def test_10_left_numerical_differential_overlap_10(self):
        K1L0_10 = self.a1a0.left_numerical_gradient(1, 0)
        self.assertAlmostEqual(K1L0_10, 0.)

    def test_10_left_numerical_differential_overlap_11(self):
        K1L0_11 = self.a1a0.left_numerical_gradient(1, 1)
        self.assertAlmostEqual(K1L0_11, 0.67)

    def test_10_left_analytical_differential_overlap(self):
        KL10 = self.a1a0.left_orbital_gradient()
        np.testing.assert_allclose(KL10, [[0, .0], [0.76, 0.67]])

###

    def test_11_left_numerical_differential_overlap_00(self):
        K1L1_00 = self.a1a1.left_numerical_gradient(0, 0)
        self.assertAlmostEqual(K1L1_00, 0.0)

    def test_11_left_numerical_differential_overlap_01(self):
        K1L1_01 = self.a1a1.left_numerical_gradient(0, 1)
        #(1 .1) (.6 -.7) = .53
        self.assertAlmostEqual(K1L1_01, 0.53)

    def test_11_left_numerical_differential_overlap_10(self):
        K1L1_10 = self.a1a1.left_numerical_gradient(1, 0)
        # (.1 1) (.6 -.7) = -.64
        self.assertAlmostEqual(K1L1_10, 0.0)

    def test_11_left_numerical_differential_overlap_11(self):
        K1L1_11 = self.a1a1.left_numerical_gradient(1, 1)
        # (.1 1) (.6 -.7) = -.64
        self.assertAlmostEqual(K1L1_11, -0.64)

    def test_11_left_analytical_differential_overlap(self):
        KL11 = self.a1a1.left_orbital_gradient()
        np.testing.assert_allclose(KL11, [[0, 0], [0.53, -.64]])


class StructTest(unittest.TestCase):

    def setUp(self):
        Nod.S = init([[1.0, 0.1], [0.1, 1.0]])
        Nod.C = init([[0.7, 0.7], [0.7, -0.7]])

    def tearDown(self):
        pass

    def test_structure_coefficients_consistent(self):
        with self.assertRaises(StructError):
            struct = structure([Nod([0], [0])], [])

    def test_structure_output(self):
        alpha = Nod([0], [])
        struct_a = structure([alpha], [1])
        self.assertEqual(str(struct_a), "1.000000    (0|)")

    def test_structure_ms(self):
        alpha = Nod([0], [])
        beta = Nod([], [0])
        with self.assertRaises(StructError):
            struct = structure([alpha, beta], [1, 1])

if __name__ == "__main__":
    unittest.main()
