import numpy
import scipy.optimize
from . import core as vb
from util import full, blocked


class Minimizer(object):

    def __init__(self, x, f, g=None, method=None, constraints=None, bounds=None):
        self.x = x
        self.f = f
        self.g = g
        self.method = method
        self.c = constraints
        self.b = bounds
        self.value = None
        def callback(xk):
            print "xk", xk, "f(xk)", self.f(xk, self), self.g(xk, self)
        self.callback = callback
        self.options = {}
        self.gtol = 1e-5
        self.maxit = 10

    def set_method(self, method):
        self.method = method

    def minimize(self):
        result = scipy.optimize.minimize(
            self.f, self.x,  method=self.method, jac=self.g,
            args=(self,), constraints=self.c, bounds=self.b,
            callback=None, options=self.options
            )
        self.x = result.x
        self.value = result.fun
        if not result.success:
            for key in result:
                if key == 'allvecs':
                    print "%s:"%key, len(result[key]), full.init(result[key])
                else:
                    print "%s:"%key, result[key]
            print self.f(result.x, self)
                
        

class LagrangianMinimizer(Minimizer):

    def __init__(self, p, f, g, *args, **kwargs):
        self.p = full.init(p)
        self.l = numpy.ones(len(kwargs['constraints']))
        Minimizer.__init__(
            self, 
            self.x, self.lagrangian(), g=self.lagrangian_derivative(),
            method=kwargs.get('method', 'BFGS'), constraints=())
        self.fp = f
        self.gp = g
        self.hp = kwargs.get('hessian', None)
        self.cp = kwargs['constraints']

    @property
    def x(self):
        _x = full.init(tuple(self.p) + tuple(self.l))
        return _x

    @x.setter
    def x(self, x_in):
        assert len(x_in) == len(self.p) + len(self.l)
        self.p[:] = x_in[:len(self.p)]
        self.l[:] = x_in[len(self.p):]

    @staticmethod
    def lagrangian():
        def fun(x, self):
            _x = self.x
            self.x = x
            L_x = self.fp(self.p) - sum(l*c['fun'](self.p) for l, c in zip(self.l, self.cp))
            self.x = _x
            return L_x
        return fun

    @staticmethod
    def lagrangian_derivative():
        def grad(x, self):
            _x = self.x
            self.x = x
            p = self.p
            dL = full.matrix(len(x))
            dL[:len(p)] = self.gp(p) - sum(l*c['jac'](p) for l, c in zip(self.l, self.cp))
            dL[len(p):] = [-c['fun'](p) for c in self.cp]
            self.x = _x
            return dL
        return grad

    @staticmethod
    def lagrangian_hessian():
        def hess(x, self):
            _x = self.x
            self.x = x
            p = self.p
            d2L = full.matrix((len(x), len(x)))
            d2L[:len(p), :len(p)] = self.hp(p) - sum(l*c['hes'](p) for l, c in zip(self.l, self.cp))
            for i, c in enumerate(self.cp):
                d2L[:len(p), len(p) + i] = -c['jac'](p)
            d2L[len(p):, :len(p)] = d2L[:len(p), len(p):].T
            d2L[len(p):, len(p):] = 0.0
            self.x = _x
            return d2L
        return hess

    def minimize(self):
        if self.method == 'MyNewton':
            dL = self.lagrangian_derivative()
            d2L = self.lagrangian_hessian()
            for i in range(self.maxit):
                if dL(self.x, self).norm2() < self.gtol:
                    break
                self.x = self.x - dL(self.x, self)/d2L(self.x, self)
        else:
            super(self.__class__, self).minimize()
        


class VBStructureCoefficientMinimizer(Minimizer):

    def __init__(self, wf):
        self.wf = wf
        x0 = self.x
        c = (
            {'type': 'eq',
             'fun': self.constraint_norm,
             'jac': self.constraint_norm_grad,
             'args': (self,)
            },
            )
        self.b = None
        Minimizer.__init__(self, x0, self.f, self.g, 'SLSQP', constraints=c)
        

    @property
    def x(self):
        return self.coef

    @x.setter
    def x(self, x_in):
        self.coef[:] = x_in

    @staticmethod
    def f(x, self):
        self.coef[:] = x
        return self.energy() + self.Z

    @staticmethod
    def g(x, self):
        return self.energygrad()[0]

    @staticmethod
    def constraint_norm(x, self):
        return self.norm() - 1.0

    @staticmethod
    def constraint_norm_grad(x, self):
        self.coef[:] = x
        return self.normgrad()[0]

    def __getattr__(self, attr):
        return getattr(self.wf, attr)

class VBMinimizer(Minimizer):

    def __init__(self, wf):
        self.wf = wf
        x0 = self.x
        self.c = (
            {'type': 'eq',
             'fun': self.constraint_norm,
             'jac': self.constraint_norm_grad,
             'args': (self,)
            },
            ) + tuple(
                {'type': 'eq',
                 'fun': self.constraint_orbital_norm(i),
                 'jac': self.constraint_orbital_norm_grad(i),
                 'args': (self,)
                } for i in range(self.C.shape[1])
            ) + tuple(
                {'type': 'eq',
                 'fun': self.constraint_structure_norm(i),
                 'jac': self.constraint_structure_norm_grad(i),
                 'args': (self,)
                } for i in range(len(self.coef))
            )
        self.b = None
        Minimizer.__init__(self, x0, self.f, self.g, 'SLSQP', constraints=self.c)


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
    def constraint_orbital_norm_grad(i):
        def constraint(x, self):
            self.x = x
            mo = self.C[:, i]
            sg = full.matrix(self.coef.shape)
            og = full.matrix(self.C.shape)
            og[:, i] = 2*vb.Nod.S*mo
            return self.so2x(sg, og)
        return constraint

    @staticmethod
    def constraint_structure_norm(i):
        def constraint(x, self):
            self.x = x
            return self.structs[i].overlap() - 1.0
        return constraint

    @staticmethod
    def constraint_structure_norm_grad(i):
        def constraint(x, self):
            self.x = x
            sg = full.matrix(self.coef.shape)
            og = self.structs[i].overlap_gradient()
            return self.so2x(sg, og)
        return constraint

    def __getattr__(self, attr):
        return getattr(self.wf, attr)
