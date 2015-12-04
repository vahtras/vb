import unittest
from daltools.util.full import init
from vb import *

class StructTest(unittest.TestCase):

    def setUp(self):
        Nod.S = init([[1.0, 0.1], [0.1, 1.0]])
        Nod.C = init([[0.7, 0.7], [0.7, -0.7]])

    def tearDown(self):
        pass

    def test_structure_coefficients_consistent(self):
        with self.assertRaises(StructError):
            struct = Structure([Nod([0], [0])], [])

    def test_structure_output(self):
        alpha = Nod([0], [])
        struct_a = Structure([alpha], [1])
        self.assertEqual(str(struct_a), "1.000000    (0|)")

    def test_structure_ms(self):
        alpha = Nod([0], [])
        beta = Nod([], [0])
        with self.assertRaises(StructError):
            struct = Structure([alpha, beta], [1, 1])

if __name__ == "__main__":
    unittest.main()
