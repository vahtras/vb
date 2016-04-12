import unittest
from ..daltools.util.full import init
from ..vb import *
from ..num_diff.findif import clgrad, DELTA

class StructTest(unittest.TestCase):

    def setUp(self):
        Nod.S = init([[1.0, 0.1], [0.1, 1.0]])
        Nod.C = init([[0.7, 0.7], [0.7, -0.7]])
        self.alpha0 = Nod([0], [])
        self.alpha1 = Nod([1], [])
        self.beta0 = Nod([], [0])
        self.beta1 = Nod([], [1])
        self.ab00 = Nod([0], [0])

    def tearDown(self):
        pass

    def test_structure_coefficients_consistent(self):
        with self.assertRaises(StructError):
            struct = Structure([Nod([0], [0])], [])

    def test_structure_output(self):
        struct_a = Structure([self.alpha0], [1.0])
        self.assertEqual(str(struct_a), "0.963143    (0|)")

    def test_structure_ms(self):
        with self.assertRaises(StructError):
            struct = Structure([self.alpha0, self.beta0], [1, 1])

    def test_normalized(self):
        ab = Structure([self.ab00], [1.0])
        self.assertAlmostEqual(ab*ab, 1.0)

    def test_keep_unnormalized(self):
        ab = Structure([self.ab00], [1.0], normalize=False)
        self.assertAlmostEqual(ab*ab, 1.162084)

    def test_norm_gradient(self):
        ab = Structure([self.ab00], [1.0])
        #assert False
        num_diff = clgrad(ab, 'overlap', 'C')()
        ana_diff = ab.overlap_gradient()
        np.testing.assert_allclose(ana_diff, num_diff, rtol=DELTA, atol=DELTA)

    def test_struct_mo_propagated(self):
        ab = Structure([self.ab00], [1.0])
        ab.C = init([1,2,3,4])
        self.assertEqual(id(ab.C), id(ab.nods[0].C))


if __name__ == "__main__":
    unittest.main()
