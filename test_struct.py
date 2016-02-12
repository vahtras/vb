import unittest
from daltools.util.full import init
from vb import *

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
        struct_a = Structure([self.alpha0], [1])
        self.assertEqual(str(struct_a), "1.000000    (0|)")

    def test_structure_ms(self):
        with self.assertRaises(StructError):
            struct = Structure([self.alpha0, self.beta0], [1, 1])

    def test_normalize(self):
        ab = Structure([self.ab00], [1.0])
        ab.normalize()
        self.assertAlmostEqual(ab*ab, 1.0)

if __name__ == "__main__":
    unittest.main()
