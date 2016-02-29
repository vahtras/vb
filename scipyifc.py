import numpy
import scipy.optimize
import vb

class Minimizer(object):

    def __init__(self, x, f, g, method, args=(), constraints=None, bounds=None):
        self.x = x
        self.f = f
        self.g = g
        self.method = method
        self.args = args
        self.c = constraints
        self.b = bounds
        self.value = None

    def minimize(self):
        result = scipy.optimize.minimize(
            self.f, self.x,  method=self.method, jac=self.g,
            args=self.args, constraints=self.c, bounds=self.b
            )
        self.x = result.x
        self.value = result.fun

class VBMinimizer(Minimizer):

    def __init__(self, wf):
        Minimizer.__init__(self, wf.coef, self.f, self.g, 'SLSQP', args=(wf,))
        self.wf = wf
        self.c = (
            {'type': 'eq',
             'fun': self.constraint_norm,
             'jac': self.constraint_norm_grad,
             'args': (self.wf,)
            },
            )
        self.b = None
        

    @property
    def x(self):
        return self.coef

    @x.setter
    def x(self, coef):
        self.coef = coef

    @staticmethod
    def f(x, wf):
        wf.coef = x
        return wf.energy() + wf.Z

    @staticmethod
    def g(x, wf):
        return wf.energygrad()[0]

    @staticmethod
    def constraint_norm(x, wf):
        return wf.norm() - 1.0

    @staticmethod
    def constraint_norm_grad(x, wf):
        wf.coef = x
        return wf.normgrad()[0]

    def __getattr__(self, attr):
        return getattr(self.wf, attr)
