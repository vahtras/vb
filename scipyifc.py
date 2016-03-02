import numpy
import scipy.optimize
import vb
from daltools.util import full, blocked


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

class VBStructureCoefficientMinimizer(Minimizer):

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

class VBMinimizer(Minimizer):

    def __init__(self, wf):
        self.wf = wf
        x0 = self.x
        Minimizer.__init__(self, x0, self.f, self.g, 'SLSQP', args=(self,))
        self.c = (
            {'type': 'eq',
             'fun': self.constraint_norm,
             'jac': self.constraint_norm_grad,
             'args': (self,)
            },
            )
        self.b = None


    @property
    def x(self):
        return self.so2x(self.coef, self.C)

    def so2x(self, s, o):
        Cblockedsize = sum(i*j for i, j in zip(*self.blockdims))
        _x = full.matrix(s.size + Cblockedsize)
        _x[:s.size] = s
        _x[s.size:] = o.block(*self.blockdims).ravel(order='F')
        return _x

    @x.setter
    def x(self, x_in):
        nstructs = len(self.coef)
        self.wf.coef = x_in[:nstructs]
        C = blocked.BlockDiagonalMatrix.init_from_array(x_in[nstructs:], *self.wf.blockdims)
        self.wf.C[:, :] = C.unblock()

    @staticmethod
    def f(x, self):
        self.x = x
        return self.energy() + self.Z

    @staticmethod
    def g(x, self):
        self.x = x
        return self.so2x(*self.wf.energygrad())

    @staticmethod
    def constraint_norm(x, self):
        self.x = x
        return self.norm() - 1.0

    @staticmethod
    def constraint_norm_grad(x, self):
        self.x = x
        return self.so2x(*self.wf.normgrad())

    @staticmethod
    def constraint_orbital_norm(i):
        def constraint(x, self):
            self.x = x
            mo = self.C[:, i]
            return (mo.T & (vb.Nod.S*mo)) - 1.0 
        return constraint

    @staticmethod
    def constraint_structure_norm(i):
        def constraint(x, self):
            self.x = x
            return self.structs[i].overlap() - 1.0
        return constraint

    def __getattr__(self, attr):
        return getattr(self.wf, attr)
