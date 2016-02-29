import unittest
import numpy
from scipyifc import Minimizer

class MinTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_parabola(self):
        x0 = numpy.array((2.0, 0.0))
        f = lambda x: (x[0] - 1)**2 + (x[1] - 2)**2
        g = lambda x: numpy.array([2*(x[0] - 1), 2*(x[1] - 2)])
        xfg = Minimizer(x0, f, g, method='L-BFGS-B')
        xfg.minimize()
        numpy.testing.assert_allclose(xfg.x, (1.0, 2.0))

    def test_constraint(self):
        f = lambda x: (x[0] - 1)**2 + (x[1] - 2.5)**2
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

if __name__ == "__main__":
    unittest.main()
