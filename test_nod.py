import unittest
import numpy as np
from vb import nod, DKL, NodPair
from vb import structure, StructError
from daltools.util.full import init


class NodTest(unittest.TestCase):

    def setUp(self):
        nod.S = init([[1.0, 0.1], [0.1, 1.0]])
        nod.C = init([[0.7, 0.7], [0.7, -0.7]])

    def tearDown(self):
        pass

    def test_vac_empty(self):
        vac = nod([], [])
        self.assertEqual(vac.electrons(), 0)

    def test_overlap_class_variable(self):
        one = nod([0], [])
        np.testing.assert_allclose(one.S, nod.S)

    def test_retrive_alpha_orbitals(self):
        det = nod([1, 2], [2, 3])
        self.assertListEqual(det(0), [1, 2])

    def test_retrive_beta_orbitals(self):
        det = nod([1, 2], [2, 3])
        self.assertListEqual(det(1), [2, 3])

    def test_repr(self):
        det = nod([1, 2], [2, 3])
        self.assertEqual(str(det), '(1 2|2 3)')

    def test_empty_determinant_returns_none(self):
        det = nod([0], [])
        self.assertIsNone(det.orbitals()[1])

    def test_alpha_orbitals(self):
        det = nod([0], [1])
        alpha_orbitals, _ = det.orbitals()
        np.testing.assert_allclose(alpha_orbitals, [[.7], [.7]])

    def test_beta_orbitals(self):
        det = nod([0], [1])
        _, beta_orbitals = det.orbitals()
        np.testing.assert_allclose(beta_orbitals, [[.7], [-.7]])

    def test_vac_normalized(self):
        det = nod([], [])
        self.assertEqual(det*det, 1)

    def test_alpha_beta_orthogonal(self):
        alpha = nod([0], [])
        beta = nod([], [0])
        self.assertEqual(alpha*beta, 0)

    def test_single_norm(self):
        alpha = nod([0], [])
        self.assertEqual(alpha*alpha, 2*0.7*0.77)

    def test_closed_shell_norm(self):
        sigmag = nod([0], [0])
        self.assertEqual(sigmag*sigmag, (2*0.7*0.77)**2)

    def test_high_spin_norm(self):
        sigmagu = nod([0, 1], [])
        self.assertAlmostEqual(sigmagu*sigmagu, 2*.7*(1+.1)*.7*2*.7*(1-.1)*.7)

    def test_vac_ao_density(self):
        vac = nod([], [])
        np.testing.assert_allclose(DKL(vac, vac), [[[0, 0], [0, 0]],  [[0, 0], [0, 0]]])

    def test_vac_mo_density(self):
        vac = nod([], [])
        self.assertEqual(DKL(vac, vac, mo=1), [None, None])

    def test_alpha_mo_density(self):
        alpha = nod([0], [])
        DKL_a, _ = DKL(alpha, alpha, mo=1)
        np.testing.assert_allclose(DKL_a, [[1./1.078]])

    def test_alpha_ao_density(self):
        alpha = nod([0], [])
        DKL_a, _ = DKL(alpha, alpha, mo=0)
        d_a = 0.49/1.078
        np.testing.assert_allclose(DKL_a, [[d_a, d_a], [d_a, d_a]])

    def test_num_diff_right(self):
        K = nod([0], [])
        L = nod([0], [])
        L.C = nod.C.copy()
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
        nod.S = init([[1.0, 0.1], [0.1, 1.0]])
        nod.C = init([[0.7, 0.7], [0.7, -0.7]])

    def tearDown(self):
        pass

    def test_not_pair_setup(self):
        K = nod([0], [])
        L = nod([0], [])
        KL = NodPair(K, L)
        self.assertAlmostEqual(K*L, KL.overlap())



class StructTest(unittest.TestCase):

    def setUp(self):
        nod.S = init([[1.0, 0.1], [0.1, 1.0]])
        nod.C = init([[0.7, 0.7], [0.7, -0.7]])

    def tearDown(self):
        pass

    def test_structure_coefficients_consistent(self):
        with self.assertRaises(StructError):
            struct = structure([nod([0], [0])], [])

    def test_structure_output(self):
        alpha = nod([0], [])
        struct_a = structure([alpha], [1])
        self.assertEqual(str(struct_a), "1.000000    (0|)")

    def test_structure_ms(self):
        alpha = nod([0], [])
        beta = nod([], [0])
        with self.assertRaises(StructError):
            struct = structure([alpha, beta], [1, 1])

if __name__ == "__main__":
    unittest.main()
