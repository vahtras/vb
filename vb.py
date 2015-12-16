#!/usr/bin/env python
"""Valence Bond energies, gradients, Hessian"""
import os
import sys
import math
from .daltools import one
from .two_electron import two
from .two_electron.two import fockab as Fao
from .daltools.util import full

DELTA = 1e-4

class BraKet(object):
    """Non-orthogonal determinant pairs"""

    tmpdir = '/tmp'

    def __init__(self, K, L, tmpdir=None):
        self.K = K
        self.L = L
        self._td = None
        self._ftd = None
        self._aotd = None
        if tmpdir is not None:
            self.tmpdir = tmpdir

    def __str__(self):
        return "<%s|...|%s>" % (self.K, self.L)

    def tmp(self, filename):
        """Return full path name to file in tmpdir"""
        return os.path.join(self.tmpdir, filename)

    def __mul__(self, h):
        if is_one_electron(h):
            return self.oneel_energy(h)*self.overlap()
        elif is_two_electron(h):
            raise Exception("Not implemented")
        else:
            raise Exception("Unknown multiplicator")

    def overlap(self):
        """Returns determinant overlap <K|L>"""
        return self.K*self.L

### Densities

    @property
    def transition_density(self):
        """Return mo transition density based on inversion formula"""
        if self._td is None:
            self._td = Dmo(self.K, self.L)
        return self._td

    @property
    def contravariant_transition_density_ao_mo(self):
        """Return contravariant density matrix in mix ao,mo basis"""
        Dam = (full.matrix(Nod.C.shape), full.matrix(Nod.C.shape))
        CK = self.K.orbitals()
        for s in (0, 1):
            if self.K(s) and self.L(s):
                Dam[s][:, self.L(s)] = CK[s]*self.transition_density[s]
        return Dam

    @property
    def contravariant_transition_density_mo_ao(self):
        """Return contravariant density matrix in mix mo,ao basis"""
        Dma = (full.matrix(Nod.C.shape[::-1]), full.matrix(Nod.C.shape[::-1]))
        CL = self.L.orbitals()
        for s in (0, 1):
            if self.K(s) and self.L(s):
                Dma[s][self.K(s), :] = self.transition_density[s]*CL[s].T
        return Dma

    def co_contravariant_transition_density_ao_mo(self):
        """
        Return mixed contravariant-covariant density matrix in mix ao,mo basis
        """
        return tuple(
            Nod.S*d for d in self.contravariant_transition_density_ao_mo
            )

    def contra_covariant_transition_density_mo_ao(self):
        """
        Return mixed covariant-contravariant density matrix in mix mo,ao basis
        """
        return tuple(
            d*Nod.S for d in self.contravariant_transition_density_mo_ao
            )

    @property
    def full_mo_transition_density(self):
        """
        Return mo transition density matrix in full mo basis
        """
        if self._ftd is None:
            _, mo = Nod.C.shape
            D_KL = self.transition_density
            self._ftd = (full.matrix((mo, mo)), full.matrix((mo, mo)))
            for s in (0, 1):
                if self.K(s) and self.L(s):
                    D_KL[s].scatter(
                        self._ftd[s], rows=self.K(s), columns=self.L(s)
                        )
        return self._ftd

    @property
    def transition_ao_density(self):
        """
        Return ao transition density matrix
        """
        if self.overlap() == 0:
            raise Exception("Density not implemented for singular overlap")
        return Dao(self.K, self.L)

    def covariant_density_ao(self):
        """
        Return covariant ao transition density matrix
        """
        S = Nod.S
        return tuple(S*d*S for d in self.transition_ao_density)

    def covariant_transition_delta(self):
        """
        Return covariant ao delta matrix
        """
        S = Nod.S
        Delta_aa = (S - d for d in self.covariant_density_ao())
        return tuple(Delta_aa)

    def co_contravariant_transition_delta(self):
        """
        Return mixed covariant-contravariant ao delta matrix
        """
        S = Nod.S
        I = full.unit(S.shape[0])
        Delta_aa = (I - S*d for d in self.transition_ao_density)
        return tuple(Delta_aa)

    def contra_covariant_transition_delta(self):
        """
        Return mixed contravariant-covariant ao delta matrix
        """
        S = Nod.S
        S = Nod.S
        I = full.unit(S.shape[0])
        Delta_aa = (I - d*S for d in self.transition_ao_density)
        return tuple(Delta_aa)

### Fock

    @property
    def transition_ao_fock(self):
        """Return AO fock matrix F(DKL) for transition density"""
        FKL = tuple(
            f.view(full.matrix) for f in Fao(
                self.transition_ao_density, filename=self.tmp('AOTWOINT')
            )
        )
        return FKL

### Overlap differentiation

    def overlap_gradient(self):
        """Return total overlap gradient d/dC <K|L>"""
        return self.left_overlap_gradient() + self.right_overlap_gradient()

    def left_overlap_gradient(self):
        """
        Lhs derivative <dK/dC(^mu, _m)|L>

        D(^m,_mu)<K|L>
        """
        return sum(self.left_overlap_gradient_ab())

    def right_overlap_gradient(self):
        """
        Rhs derivative <K|dL/dC(^mu, _m)>

        D(_mu,^m)<K|L>
        """
        return sum(self.right_overlap_gradient_ab())

    def left_overlap_gradient_ab(self):
        """
        Lhs alpha,beta derivative
        <dKa/dC(mu, m) Kb|L> ,<Ka dKb/dC(mu,m)|La dLb>
        """

        ol = self.overlap()
        dKL = (d.T*ol for d in self.contra_covariant_transition_density_mo_ao())
        return tuple(dKL)

    def right_overlap_gradient_ab(self):
        """Rhs alpha,beta derivative <K|dLa/dC(mu, m)Lb> ,<K|La dLb/dC(mu,m)>"""

        ol = self.overlap()
        KdL = (d*ol for d in self.co_contravariant_transition_density_ao_mo())
        return tuple(KdL)

    def overlap_hessian(self):
        """Return total overlap Hessian d2/dC2 <K|L>"""
        return self.left_overlap_hessian() + self.right_overlap_hessian() +\
            2*self.mixed_overlap_hessian()

    def left_overlap_hessian(self):
        """Return left overlap Hessian <d2K|L>"""
        return BraKet(self.L, self.K).right_overlap_hessian()

    def right_overlap_hessian(self):
        """
        Rhs derivative <K|d2/dC(mu, m)dC(nu, n)|L>

        <K|a_mu+ a_nu+ a^n a^m|L>
        (D(_mu, ^n)D(_nu, ^n) - sum(s)D(s)(_mu, ^n)D(s)(_nu,^m))<K|L>
        """
        #
        # Orbital-orbital hessian
        #
        aD_am, bD_am = self.co_contravariant_transition_density_ao_mo()
        D_am = aD_am + bD_am

        Kd2L = (
            D_am.x(D_am)
            - (aD_am.x(aD_am) + bD_am.x(bD_am)).transpose((2, 1, 0, 3))
        )*self.overlap()

        return Kd2L

    def mixed_overlap_hessian(self):
        """
        L-R derivative <dK/dC(mu,m)|dL/dC(nu,n)>

        <K|a^m+ a_mu a_nu+ a^n |L>
        (D(^m, _mu)D(_nu, ^n) + sum(s)D(s)(^m, ^n)Delta(s)(_nu,_mu))<K|L>
        Delta = S - S*D*S
        """

        D_am = sum(self.co_contravariant_transition_density_ao_mo())
        Dm_a = sum(self.contra_covariant_transition_density_mo_ao())
        aDmm, bDmm = self.full_mo_transition_density

        aDelta, bDelta = self.covariant_transition_delta()

        dKdL = (
            Dm_a.T.x(D_am)
            + (aDelta.x(aDmm) + bDelta.x(bDmm)).transpose(1, 2, 0, 3)
            )*self.overlap()

        return dKdL

### Energy differentiation

    def energy(self, h):
        """Return total energy"""
        return self.oneel_energy(h) + self.twoel_energy()

    def oneel_energy(self, h):
        """<K|h|L>/<K|L>"""

        dao = self.transition_ao_density
        ha, hb = h
        return (ha&dao[0]) + (hb&dao[1])

    def twoel_energy(self, *args):
        """<K|g|L>/<K|L>"""

        DKL = self.transition_ao_density
        FKL = self.transition_ao_fock
        return .5*((FKL[0]&DKL[0]) + (FKL[1]&DKL[1]))

    def twoel_tme(self):
        """<K|g|L>"""

        return self.twoel_energy()*self.overlap()

    def energy_gradient(self, h1):
        """Return gradient of transition matrix element d<K|H|L>"""
        return self.right_energy_gradient(h1) + self.left_energy_gradient(h1)

    def right_energy_gradient(self, h1):
        """Return right gradient of transition matrix element <K|H|dL>"""
        return self.right_1el_energy_gradient(h1) + \
            self.right_2el_energy_gradient()

    def right_1el_energy_gradient(self, h1):
        """Return right gradient of 1el transition matrix element <K|H|dL>"""
        return sum(self.right_1el_energy_gradient_ab(h1))

    def right_2el_energy_gradient(self, *args):
        """Return right gradient of 2el transition matrix element <K|H|dL>"""
        return sum(self.right_2el_energy_gradient_ab(*args))

    def right_1el_energy_gradient_ab(self, h1):
        """Return alpha/beta right 1el gradient <K|H|dL>"""
        eg1_a, eg1_b = (
            self.oneel_energy(h1)*g for g in self.right_overlap_gradient_ab()
        )
        eg2_a, eg2_b = self.project_virtual_occupied(h1)
        eg_ab = (eg1_a + eg2_a, eg1_b + eg2_b)
        return eg_ab

    def right_2el_energy_gradient_ab(self, *args):
        """Return alpha/beta right 2el gradient <K|H|dL>"""
        try:
            fock, = args
        except ValueError:
            fock = self.transition_ao_fock

        eg1_a, eg1_b = (
            self.twoel_energy()*g for g in self.right_overlap_gradient_ab()
        )
        eg2_a, eg2_b = self.project_virtual_occupied(fock)
        eg_ab = (eg1_a + eg2_a, eg1_b + eg2_b)
        return eg_ab

    def right_2el_energy_gradient_ab2(self):
        """Rhs derivative <K|g|dL/dC(mu, m)>"""
        return self.project_virtual_occupied(self.transition_ao_fock)

    def project_virtual_occupied(self, h1):
        """Rhs derivative <K|h|dL/dC(mu, m)>"""

        D_mo = self.transition_density
        Delta = self.co_contravariant_transition_delta()

        K_h_dL = (full.matrix(Nod.C.shape), full.matrix(Nod.C.shape))

        CK = self.K.orbitals()
        for s in (0, 1):
            if self.K(s) and self.L(s):
                K_h_dL[s][:, self.L(s)] += \
                    Delta[s]*h1[s].T*CK[s]*D_mo[s]*self.overlap()
        return K_h_dL


    def left_energy_gradient(self, h1):
        """Return left gradient of transition matrix element <dK|H|L>"""
        return self.left_1el_energy_gradient(h1) + \
            self.left_2el_energy_gradient()

    def left_1el_energy_gradient(self, h1):
        """Return left gradient of 1el transition matrix element <dK|H|L>"""
        return sum(self.left_1el_energy_gradient_ab(h1))

    def left_2el_energy_gradient(self, *args):
        """Return left gradient of 2el transition matrix element <dK|H|L>"""
        return sum(self.left_2el_energy_gradient_ab(*args))

    def left_1el_energy_gradient_ab(self, h1):
        """Return alpha/beta left 1el gradient <dK|H|L>"""
        eg1_a, eg1_b = (
            self.oneel_energy(h1)*g for g in self.left_overlap_gradient_ab()
        )
        eg2_a, eg2_b = self.project_occupied_virtual(h1)
        eg_ab = (eg1_a + eg2_a, eg1_b + eg2_b)
        return eg_ab

    def left_2el_energy_gradient_ab(self, *args):
        """Return alpha/beta left 2el gradient <dK|H|L>"""
        try:
            fock, = args
        except ValueError:
            fock = self.transition_ao_fock
        eg1_a, eg1_b = (
            self.twoel_energy()*g for g in self.left_overlap_gradient_ab()
        )
        eg2_a, eg2_b = self.project_occupied_virtual(fock)
        eg_ab = (eg1_a + eg2_a, eg1_b + eg2_b)
        return eg_ab

    def project_occupied_virtual(self, h1):
        """Lhs derivative <dK/dC(mu,m)|h|L>"""

        D_mo = self.transition_density
        Delta = self.contra_covariant_transition_delta()

        dK_h_La = full.matrix(Nod.C.shape[::-1])
        dK_h_Lb = full.matrix(Nod.C.shape[::-1])

        CL = self.L.orbitals()
        if self.L(0):
            dK_h_La[self.K(0), :] += \
                D_mo[0]*CL[0].T*h1[0].T*Delta[0]*self.overlap()
        if self.L(1):
            dK_h_Lb[self.K(1), :] += \
                D_mo[1]*CL[1].T*h1[1].T*Delta[1]*self.overlap()

        return dK_h_La.T, dK_h_Lb.T

    def right_1el_energy_hessian(self, h1):
        """Right 1el energy Hessian <K|h|d2L>"""
        K_h_d2L = self.right_gen_energy_hessian(
            self.oneel_energy,
            self.project_virtual_occupied,
            h1
            )
        return K_h_d2L

    def right_2el_energy_hessian(self):
        """Right 2el energy Hessian <K|g|d2L>"""
        K_h_d2L = self.right_gen_energy_hessian(
            self.twoel_energy,
            self.right_2el_energy_gradient_ab2
            )

        Dam = self.contravariant_transition_density_ao_mo
        Delta = self.co_contravariant_transition_delta()
        K_h_d2L += two.vb_transform(
            Dam, Delta,
            filename=os.path.join(self.tmpdir, 'AOTWOINT')
            )*self.overlap()

        return K_h_d2L

    def right_gen_energy_hessian(self, energy, right_gradient, *args):
        """Rhs derivative <K|h|d2L/dC(mu, m)dC(nu, n)>"""

        KL = self.overlap()
        KdL = self.right_overlap_gradient()
        Kd2L = self.right_overlap_hessian()
        KhdL = sum(right_gradient(*args))

        e1 = energy(*args)

        K_h_d2L = e1*Kd2L + (KdL.x(KhdL) + KhdL.x(KdL))/KL

        na, nb = self.right_overlap_gradient_ab()
        ha, hb = right_gradient(*args)
        nh = na.x(ha) + nb.x(hb)
        K_h_d2L -= (nh.transpose(0, 3, 2, 1) + nh.transpose(2, 1, 0, 3))/KL

        return K_h_d2L


    def mixed_1el_energy_hessian(self, h1):
        """Mixed 1el energy Hessian <dK|h|dL>"""
        dK_h_dL = self.mixed_gen_hessian(
            self.oneel_energy,
            self.project_occupied_virtual,
            self.project_virtual_occupied,
            h1
            )
        return dK_h_dL

    def mixed_2el_energy_hessian(self):
        """Mixed 2el energy Hessian <dK|g|dL>"""
        dK_g_dL = self.mixed_gen_hessian(
            self.twoel_energy,
            self.project_occupied_virtual,
            self.project_virtual_occupied,
            self.transition_ao_fock
            )


        dK_g_dL += two.vb_transform2(
            self.contravariant_transition_density_mo_ao,
            self.contravariant_transition_density_ao_mo,
            self.contra_covariant_transition_delta(),
            self.co_contravariant_transition_delta(),
            filename=os.path.join(self.tmpdir, 'AOTWOINT')
            )*self.overlap()
        return dK_g_dL

    def mixed_gen_hessian(self, energy, left_gradient, right_gradient, *args):
        r"""
        L-R derivative <dK/dC(mu,m)|h|dL/dC(nu,n)>

        <K|a^m+ a_\mu h a\nu+ a^n|L>
        """
        # <h><K|a^m+ a_\mu h a_nu+ a^n|L>
        dK_h_dL = energy(*args)*self.mixed_overlap_hessian()

        # D^m_mu<K|H|dL/dC^nu_n> + <dK/dC^mu_m|h|L>D_nu^n
        KL = self.overlap()
        dK_L = self.left_overlap_gradient()
        K_dL = self.right_overlap_gradient()
        K_h_dL = sum(right_gradient(*args))
        dK_h_L = sum(left_gradient(*args))

        dK_h_dL += (dK_L.x(K_h_dL) + dK_h_L.x(K_dL))/KL

        # D^{mn}Delta^xi_mu h_{xi,rho}Delta_nu^rho
        Dmm = self.full_mo_transition_density
        DhD = self.project_virtual_virtual(*args)
        dK_h_dL += (
            Dmm[0].x(DhD[0])*KL +\
            Dmm[1].x(DhD[1])*KL
            ).transpose(3, 0, 2, 1)

        # Delta_{nu, mu} D^{m,rho}h_{xi, rho}D^{xi, n}
        Hmm = self.project_occupied_occupied(*args)
        Delta = self.covariant_transition_delta()
        dK_h_dL -= (
            Delta[0].x(Hmm[0])*KL + Delta[1].x(Hmm[1])*KL
            ).transpose(1, 2, 0, 3)

        return dK_h_dL

    def project_virtual_virtual(self, op):
        """Project virtual-virtual"""
        Delta1 = self.co_contravariant_transition_delta()
        Delta2 = self.contra_covariant_transition_delta()
        return tuple(d1*h.T*d2 for d1, h, d2 in zip(Delta1, op, Delta2))

    def project_occupied_occupied(self, op):
        """Project occupied-occupied"""
        Dma = self.contravariant_transition_density_mo_ao
        Dam = self.contravariant_transition_density_ao_mo
        h1mm = (Dma[0]*op[0].T*Dam[0], Dma[1]*op[1].T*Dam[1])
        return h1mm


class Nod(object):
    """
    # Class of non-orthogonal determinants
    """
    #
    # Class global variables
    #
    S = None   #Overlap ao basis
    C = None   #VB orbital coefficients
    tmpdir = '/tmp'

    def __init__(self, astring, bstring, C=None, tmpdir=None):
        super(Nod, self).__init__()
        #
        # Input: list of orbital indices for alpha and beta strings
        #
        self.a = astring
        self.b = bstring

        if C is None:
            self.C = Nod.C
        else:
            self.C = C

        if tmpdir is not None:
            self.tmpdir = tmpdir
        #
        # Read overlap at first invocation of constructor
        #
        aooneint = os.path.join(self.tmpdir, 'AOONEINT')
        if Nod.S is None:
            Nod.S = one.read('OVERLAP', aooneint).unpack().unblock()
            #
            # init unit for H2 test
            #

    def electrons(self):
        """Return number of electrons in determinant"""
        return len(self.a + self.b)

    def __call__(self, s):
        """
        # Returns alpha (s=0) or beta (s=1) string
        """
        if s == 0:
            return self.a
        elif s == 1:
            return self.b
        else:
            return None

    def __repr__(self):
        stra = " ".join(["%g" % alpha for alpha in self.a])
        strb = " ".join(["%g" % beta for beta in self.b])
        retstr = "(%s|%s)" % (stra, strb)
        return retstr

    def orbitals(self):
        """Extract mo coefficients for current determinant"""
        CUa = None
        CUb = None
        if self.a:
            CUa = self.C[:, self.a]
        if self.b:
            CUb = self.C[:, self.b]
        return (CUa, CUb)

    def __mul__(self, other):
        """
        # Return overlap of two Slater determinants <K|L>
        # calculated as matrix determinant of overlap
        # of orbitals in determinants
        # det(S) = det S(alpha)*det S(beta)
        """
        if len(self.a) != len(other.a) or len(self.b) != len(other.b):
            return 0

        (CKa, CKb) = self.orbitals()
        (CLa, CLb) = other.orbitals()
        #
        # alpha
        #
        Det = 1
        if CKa is not None:
            SKLa = CKa.T*Nod.S*CLa
            Deta = SKLa.det()
            Det *= Deta
        #
        # beta
        #
        if CKb is not None:
            SKLb = CKb.T*Nod.S*CLb
            Detb = SKLb.det()
            Det *= Detb
        #
        return Det

    def ao_density(self):
        """Return ao density"""
        return Dao(self, self)

    def mo_density(self):
        """Return reduced mo density"""
        return Dmo(self, self)


#
# Calculate transition density from unnormalized molecular orbitals
#
def Dao(K, L):
    """
    # Return intermediate normalized ao transition density matrix given
    # determinants K and L
    # as [Dalpha, Dbeta]
    """

    CK = K.orbitals()
    CL = L.orbitals()
    #
    D = []

    for s in range(2):
        if CK[s] is None or CL[s] is None:
            #
            # if None orbitals set atomic density to zero matrix
            #
            D.append(full.matrix(Nod.S.shape))
        else:
            SLK = CL[s].T*Nod.S*CK[s]
            D.append(CK[s]*(CL[s].T/SLK))

    return D

def Dmo(K, L):
    """
    # Return intermediate normalized mo transition density matrix given
    # determinants K and L
    # as [Dalpha, Dbeta]
    """

    CK = K.orbitals()
    CL = L.orbitals()
    #
    D = []

    for s in range(2):
        if CK[s] is None or CL[s] is None:
            D.append(None)
        else:
            SLK = CL[s].T*Nod.S*CK[s]
            D.append(SLK.inv())

    return D


def HKL(F, D):
    """Returns two-electron energy given 2el-Fock/Density matrix"""
    E = 0.5*((F[0]&D[0]) + (F[1]&D[1]))
    return E

class Structure(object):
    """
    VB Structure type

    nods: Non-orthogonal determinants
    coef: fix coupling coefficients
    """
    def __init__(self, nods, coef):
        if len(nods) != len(coef):
            raise StructError

        self.nods = nods
        self.assert_consistent_electron_number()

        self.coef = full.init(coef)
        #
        # Also have MOs as a structure member
        # In BOVB these are unique to a structure
        # reference the first determinant MO
        #
        self.C = nods[0].C

    def assert_consistent_electron_number(self):
        n0 = self.nods[0]
        for n in self.nods[1:]:
            if len(n.a) != len(n0.a) or len(n.b) != len(n0.b):
                raise StructError

    def __mul__(self, other):
        N = 0
        for bra, c_bra in zip(self.nods, self.coef):
            for ket, c_ket in zip(other.nods, other.coef):
                N += bra*ket*c_bra*c_ket
        return N

    def __str__(self):
        output = ["%f    %s" % (c, d) for c, d in zip(self.coef, self.nods)]
        return "\n".join(output)

class StructError(Exception):
    pass


class WaveFunction(object):
    """
    VB wave function

    structs: list of structures
    coef: structure coefficients
    """

    tmpdir = '/tmp'

    def __init__(self, structs, coef, VBSCF=True, tmpdir=None):
        self.structs = structs
        self.coef = full.init(coef)
        if tmpdir is not None:
            self.tmpdir = tmpdir
        self.Z = one.readhead(self.tmp("AOONEINT"))["potnuc"]
        self.h = one.read("ONEHAMIL", self.tmp("AOONEINT")).unpack().unblock()
        BraKet.tmpdir = self.tmpdir
        #
        # For VBSCF all structures share orbitals
        #
        if VBSCF:
            self.C = Nod.C
        else:
            raise Exception("not implemented")

    def tmp(self, filename):
        """Return full path of file in tmpdir"""
        return os.path.join(self.tmpdir, filename)

    def __str__(self):
        retstr = "\n"
        for i, coef in enumerate(self.coef):
            retstr += "%10.4f (%d)\n" % (coef, i+1)
        return retstr

    def nel(self):
        """Return number of electrons from ao density"""
        return self.ao_density() & Nod.S

    def ao_density(self):
        """Electron number check"""
        D_ao = full.matrix(Nod.S.shape)
        for S, CS in zip(self.structs, self.coef):
            for T, CT in zip(self.structs, self.coef):
                for K, CK in zip(S.nods, S.coef):
                    for L, CL in zip(T.nods, T.coef):
                        KL = BraKet(K, L)
                        DKL = KL.transition_ao_density
                        D_ao += sum(DKL)*CS*CT*CK*CL*KL.overlap()
        return D_ao/self.norm()

    def StructureHamiltonian(self):
        SH = []
        Eone = 0
        Etwo = 0
        N = 0
        # Structures left
        for S in self.structs:
            for T in self.structs:
                #Determinants in left structure
                H = 0
                for K, CKS in zip(S.nods, S.coef):
                    for L, CLT in zip(T.nods, T.coef):
                        KL = BraKet(K, L)
                        D12 = KL.transition_ao_density
                        FKL = KL.transition_ao_fock
                        hKL = self.h&(D12[0]+D12[1])
                        gKL = HKL(FKL, D12)
                        H += (hKL+gKL)*CKS*CLT*KL.overlap()
                SH.append(H)
        LS = len(self.structs)
        return full.init(SH).reshape((LS, LS))

    def StructureOverlap(self):
        """Calculate structure overlap matrix"""
        OV = []
        for S in self.structs:
            for T in self.structs:
                OV.append(S*T)
        LS = len(self.structs)
        return full.init(OV).reshape((LS, LS))

    def StructureWeights(self):
        SO = self.StructureOverlap()
        W = full.matrix(len(self.structs))
        C = full.init(self.coef)
        SOC = SO*C
        i = 0
        for c, sc in zip(C, SOC):
            W[i] = c*sc
            i += 1
        return W

    def Normalize(self):
        #
        # Orbitals
        #
        ao, mo = self.C.shape
        for i in range(mo):
            cmo = self.C[:, i]
            nmo = 1/math.sqrt(cmo.T*Nod.S*cmo)
            cmo *= nmo
        #
        # Structures
        #
        SO = self.StructureOverlap()
        for i in range(len(self.structs)):
            N = 1/math.sqrt(SO[i, i])
            self.structs[i].coef *= N
        #
        # Structure coefficients
        #
        self.coef *= 1/math.sqrt(self.norm())

    def norm(self):
        """
        Calculate norm square of VB wave function: N=<0|0>
        """
        N = 0
        for S, CS in zip(self.structs, self.coef):
            for T, CT in zip(self.structs, self.coef):
                N += S*T*CS*CT
        return N

    def normgrad(self):
        #
        #  N = <0|0>
        # dN = 2<d0|0>
        #
        NGS = []
        r, c = Nod.C.shape
        Norbgrad = full.matrix((r, c))
        #
        for S, CS in zip(self.structs, self.coef):
            GS = 0
            for T, CT in zip(self.structs, self.coef):
                GS += (S*T)*CT
                for K, CK in zip(S.nods, S.coef):
                    for L, CL in zip(T.nods, T.coef):
                        Norbgrad += (CS*CT*CK*CL) * \
                            BraKet(K, L).right_overlap_gradient()
            NGS.append(GS)

        Nstructgrad = full.init(NGS)
        Nstructgrad *= 2
        Norbgrad *= 2
        return (Nstructgrad, Norbgrad[:, :])

    def numgrad(self, func, delta=1e-3):
        #
        # Numerical gradient
        deltah = delta/2
        #
        # structure coefficients
        #
        structgrad = full.matrix(len(self.structs))
        #
        #
        for s in range(structgrad.shape[0]):
            e0 = func()
            self.coef[s] += deltah
            ep = func()
            self.coef[s] -= delta
            em = func()
            self.coef[s] += deltah
            structgrad[s] = (ep - em)/delta
        #
        # orbital gradient
        #
        r, c = Nod.C.shape
        orbgrad = full.matrix((r, c))
        for m in range(c):
            for t in range(r):
                Nod.C[t, m] += deltah
                ep = func()
                Nod.C[t, m] -= delta
                em = func()
                orbgrad[t, m] = (ep - em)/delta
                Nod.C[t, m] += deltah
        return (structgrad, orbgrad[:, :])

    def numnormgrad(self, delta=1e-3):
        return self.numgrad(self.norm, delta)

    def normhess(self):
        #
        #  N = <0|0>
        # d2<0|0> = <d20|0> + <0|d20> + 2<d0|d0> = 2<0|d20> + 2<d0|d0>
        #
        ls = len(self.structs)
        ao, mo = Nod.C.shape
        Nstructhess = full.matrix((ls, ls))
        Norbstructhess = full.matrix((ao, mo, ls))
        Norbhess = full.matrix((ao, mo, ao, mo))
        d12 = full.matrix((mo, mo))
        #
        for s1, (str1, cstr1) in enumerate(zip(self.structs, self.coef)):
            for s2, (str2, cstr2) in enumerate(zip(self.structs, self.coef)):
                for det1, cdet1 in zip(str1.nods, str1.coef):
                    for det2, cdet2 in zip(str2.nods, str2.coef):
                        #
                        bk12 = BraKet(det1, det2)
                        C1 = cstr1*cdet1
                        C2 = cstr2*cdet2
                        #
                        Nstructhess[s1, s2] += (cdet1*cdet2)*bk12.overlap()
                        Norbstructhess[:, :, s1] += \
                            cdet1*C2*bk12.overlap_gradient()
                        Norbhess += C1*C2*bk12.overlap_hessian()
        Nstructhess *= 2
        Norbstructhess *= 2
        Norbhess
        return Nstructhess, Norbstructhess, Norbhess

    def vb_metric(self):
        #
        # Based on part of norm hessian <dF|dF>
        #
        ls = len(self.structs)
        ao, mo = Nod.C.shape
        Nstructhess = full.matrix((ls, ls))
        Nstructorbhess = full.matrix((ls, mo, ao))
        Norbstructhess = full.matrix((mo, ao, ls))
        Norbhess = full.matrix((mo, ao, mo, ao))
        d12 = full.matrix((mo, mo))
        Dm_mu = [full.matrix((mo, ao)), full.matrix((mo, ao))]
        Dmu_m = [full.matrix((ao, mo)), full.matrix((ao, mo))]
        Dmn = [full.matrix((mo, mo)), full.matrix((mo, mo))]
        Delta = [None, None]
        #
        # Structures left
        #
        for s1 in range(len(self.structs)):
            S = self.structs[s1]
            CS = self.coef[s1]
            #
            #Structures right
            #
            for s2 in range(len(self.structs)):
                T = self.structs[s2]
                CT = self.coef[s2]
                #
                #Determinants in left structure
                #
                for K, CKS in zip(S.nods, S.coef):
                    #
                    #Determinants in right structure
                    #
                    for L, CLT in zip(T.nods, T.coef):
                        #
                        # Structure Hessian terms
                        #
                        KL = K*L
                        Nstructhess[s1, s2] += KL*CKS*CLT
                        #
                        # Orbital-structure Hessian terms
                        #
                        #
                        CK = K.orbitals()
                        CL = L.orbitals()
                        DmoKL = Dmo(K, L)
                        #
                        # <0m_mu|T>: D^m_mu = D^{ml}C(L)S
                        #
                        for s in range(2):
                            #
                            # D\mu_m =  Smu. C.k Dkm
                            # Dm_\mu = Dml C.l S.mu
                            #
                            Dm_mu[s][:, :] = 0
                            Dm_mu[s][K(s), :] = DmoKL[s]*CL[s].T*Nod.S
                            Dmu_m[s][:, :] = 0
                            Dmu_m[s][:, L(s)] = Nod.S*CK[s]*DmoKL[s]

                            Dmn[s][:, :] = 0
                            #
                            # Probably not working but this is what we want
                            #
                            # Dmn[s][K(s), L(s)] = DmoKL[s]
                            #
                            # instead two steps
                            #
                            tmp = full.matrix((mo, CL[s].shape[1]))
                            tmp[K(s), :] = DmoKL[s]
                            Dmn[s][:, L(s)] = tmp

                            Delta[s] = Nod.S-Nod.S*CK[s]*DmoKL[s]*CL[s].T*Nod.S
                            #
                            # Add to mixed hessian
                            #
                            Norbstructhess[:, :, s2] += CS*CKS*CLT*KL*Dm_mu[s]
                            Nstructorbhess[s1, :, :] += CT*CKS*CLT*KL*Dmu_m[s]
                            #
                            # Orbital-orbital hessian
                            #
                        for s in range(2):
                            for t in range(2):
                                Norbhess += \
                                CS*CKS*CT*CLT*KL*Dm_mu[s].x(Dmu_m[t].T)
                            Norbhess += \
                            CS*CKS*CT*CLT*KL*Dmn[s].x(Delta[s]).transpose(
                                (0, 3, 1, 2)
                                )
                        #
        tmp = Norbhess[:, :, :, :]
        Norbhess = tmp[:, :, :, :]
        return (
            Nstructhess,
            Norbstructhess[:, :, :],
            Nstructorbhess[:, :, :], Norbhess
            )

    def numnormhess(self, delta=1e-3):
        return self.numhess(self.norm, delta)

    def numhess(self, func, delta=1e-3):
        #
        # Numerical norm of Hessian for chosen elements
        #
        # Numerical gradient
        deltah = delta/2
        delta2 = delta*delta
        #
        # structure coefficients
        #
        ls = len(self.structs)
        ao, mo = Nod.C.shape
        strstrhess = full.matrix((ls, ls))
        strorbhess = full.matrix((ls, ao, mo))
        orbstrhess = full.matrix((ao, mo, ls))
        orborbhess = full.matrix((ao, mo, ao, mo))
        #
        # Structure-structure
        #
        for p in range(ls):
            for q in range(ls):
                self.coef[p] += deltah
                self.coef[q] += deltah
                epp = func()
                self.coef[q] -= delta
                epm = func()
                self.coef[p] -= delta
                emm = func()
                self.coef[q] += delta
                emp = func()
                strstrhess[p, q] = (epp + emm - epm - emp)/delta2
                #
                # Reset
                #
                self.coef[p] += deltah
                self.coef[q] -= deltah
        #
        # Orbital-structure
        #
        for p in range(ls):
            for mu in range(ao):
                for m in range(mo):
                    self.coef[p] += deltah
                    Nod.C[mu, m] += deltah
                    epp = func()
                    Nod.C[mu, m] -= delta
                    epm = func()
                    self.coef[p] -= delta
                    emm = func()
                    Nod.C[mu, m] += delta
                    emp = func()
                    orbstrhess[mu, m, p] = (epp + emm - epm - emp)/delta2
                    #
                    # Reset
                    #
                    self.coef[p] += deltah
                    Nod.C[mu, m] -= deltah

        #
        # Orbital-orbital
        #
        for mu in range(ao):
            for m in range(mo):
                for nu in range(ao):
                    for n in range(mo):
                        Nod.C[mu, m] += deltah
                        Nod.C[nu, n] += deltah
                        epp = func()
                        Nod.C[nu, n] -= delta
                        epm = func()
                        Nod.C[mu, m] -= delta
                        emm = func()
                        Nod.C[nu, n] += delta
                        emp = func()
                        orborbhess[mu, m, nu, n] = \
                            (epp + emm - epm - emp)/delta2
                        #
                        # Reset
                        #
                        Nod.C[mu, m] += deltah
                        Nod.C[nu, n] -= deltah

        return (
            strstrhess,
            orbstrhess,
            orborbhess
            )

    def energy(self):
        Eone = 0
        Etwo = 0
        N = 0
        H = 0
        # Structures left
        for str1, cstr1 in zip(self.structs, self.coef):
            #Structures right
            for str2, cstr2 in zip(self.structs, self.coef):
                #Determinants in left structure
                for det1, cdet1 in zip(str1.nods, str1.coef):
                    #Determinants in right structure
                    for det2, cdet2 in zip(str2.nods, str2.coef):
                        det12 = BraKet(det1, det2)
                        C12 = cstr1*cstr2*cdet1*cdet2*det12.overlap()
                        N += C12
                        H += C12*det12.energy((self.h, self.h))

        #H = Eone+Etwo
        E = H/N
        return E

    def energygrad(self):
        #
        #  N = <0|0>
        # dN = 2<d0|0>
        #  E = <0|H|0>/<0|0>
        # dE = 2<d0|H-E|0>/<0|0>
        #
        ao, mo = Nod.C.shape
        Nstructgrad = full.matrix(len(self.structs))
        Norbgrad = full.matrix((ao, mo))
        N = 0
        Hstructgrad = full.matrix(len(self.structs))
        Horbgrad = full.matrix((ao, mo))
        H = 0
        I = full.unit(ao)
        #
        # Structures left
        #
        for s1, (str1, cstr1) in enumerate(zip(self.structs, self.coef)):
            for str2, cstr2 in zip(self.structs, self.coef):
                for det1, cdet1 in zip(str1.nods, str1.coef):
                    for det2, cdet2 in zip(str2.nods, str2.coef):
                        det12 = BraKet(det1, det2)
                        S12 = det1*det2
                        C1 = cstr1*cdet1
                        C2 = cstr2*cdet2
                        #
                        # Structure gradient terms
                        #
                        N12 = (cdet1*C2)*S12
                        Nstructgrad[s1] += N12
                        #
                        H12 = det12.energy((self.h, self.h))
                        Hstructgrad[s1] += N12*H12
                        C12 = (C1*C2)*S12
                        #
                        # Orbital gradient terms
                        #
                        Norbgrad += C1*C2*det12.right_overlap_gradient()
                        Horbgrad += \
                            C1*C2*det12.right_energy_gradient((self.h, self.h))
                        #
                        # Energy and norm contributions
                        #
                        N += C12
                        H += C12*det12.energy((self.h, self.h))

        E = H/N # only electronic
        structgrad = (2/N)*(Hstructgrad-E*Nstructgrad)
        orbgrad = (2/N)*(Horbgrad-E*Norbgrad)
        return (structgrad, orbgrad[:, :])

    def numenergygrad(self, delta=1e-3):
        return self.numgrad(self.energy, delta)

    def energyhess(self):
        """Energy full Hessian

        Relations:
        E = H/N
        H = <0|H|0>
        N = <0|0>

        dE = (dH - E dN)/N
        dH = 2*<0|H|d0>
        dN = 2*<0|d0>

        d2E = (d2H - dE dN - E d2N)/N  - (dH - EdN)/N^2 dN
        d2H = 2*<d0|H|d0> + 2*<0|H|d20>
        d2N = 2*<d0|d0> + 2*<0|d20>
        """

        ls = len(self.structs)
        ao, mo = Nod.C.shape
        Nstructgrad = full.matrix(ls)
        Norbgrad = full.matrix((ao, mo))
        Nstructhess = full.matrix((ls, ls))
        Nstructorbhess = full.matrix((ls, ao, mo))
        Norbstructhess = full.matrix((ao, mo, ls))
        Norbhess = full.matrix((ao, mo, ao, mo))
        Hstructgrad = full.matrix(ls)
        Horbgrad = full.matrix((ao, mo))
        Hstructhess = full.matrix((ls, ls))
        Horbstructhess = full.matrix((ao, mo, ls))
        Horbhess = full.matrix((ao, mo, ao, mo))
        N = 0
        H = 0
        hab = (self.h, self.h)
        #
        for s1, (str1, Cs1) in enumerate(zip(self.structs, self.coef)):
            for s2, (str2, Cs2) in enumerate(zip(self.structs, self.coef)):
                for det1, Cd1 in zip(str1.nods, str1.coef):
                    for det2, Cd2 in zip(str2.nods, str2.coef):
                        #
                        C1 = Cs1*Cd1
                        C2 = Cs2*Cd2
                        #
                        det12 = BraKet(det1, det2)
                        S12 = det12.overlap()
                        #
                        # Structure gradient terms
                        #
                        N12 = (Cd1*Cd2)*S12
                        H12 = det12.energy((self.h, self.h))
                        Hstructhess[s1, s2] += N12*H12
                        #
                        # Orbital-structure hessian terms
                        #
                        Horbstructhess[:, :, s1] += \
                            Cd1*C2*det12.energy_gradient((self.h, self.h))
                        ##
                        # Orbital-orbital hessian
                        #
                        C12 = C1*C2*S12
                        #
                        Horbhess += C1*C2*(
                            det12.right_1el_energy_hessian((self.h, self.h)) +
                            det12.mixed_1el_energy_hessian((self.h, self.h)) +
                            det12.right_2el_energy_hessian() +
                            det12.mixed_2el_energy_hessian()
                            )
                        #

        N = self.norm()
        E = self.energy()

        Nstructgrad, Norbgrad = self.normgrad()
        Estructgrad, Eorbgrad = self.energygrad()

        Nstructhess, Norbstructhess, Norbhess = self.normhess()
        Hstructhess *= 2
        Horbstructhess *= 2
        Horbhess *= 2

        Estructhess = (
            Hstructhess - E*Nstructhess -
            Estructgrad.x(Nstructgrad) - Nstructgrad.x(Estructgrad)
            )/N

        Eorbstructhess = (
            Horbstructhess - E*Norbstructhess - \
            Eorbgrad.x(Nstructgrad) - Norbgrad.x(Estructgrad)
            )/N

        Eorbhess = (
            Horbhess - E*Norbhess - Eorbgrad.x(Norbgrad)-Norbgrad.x(Eorbgrad))/N

        return (
            Estructhess,
            Eorbstructhess,
            Eorbhess,
            )

    def numenergyhess(self, delta=1e-3):
        return self.numhess(self.energy, delta)

    def gradientvector(self):
        sg, og = self.energygrad()
        mo, ao = og.shape
        lc = mo*ao
        ls, = sg.shape
        og1 = og.flatten('F')
        gdim = ls+lc
        g = full.matrix(gdim)
        g[:ls] = sg
        g[ls:] = og1
        return g

    def gradient_norm(self):
        gv = self.gradientvector()
        G = self.vb_metric_matrix()
        return gv*G.I*gv

    def hessianmatrix(self):
        sh, mh, oh = self.energyhess()
        mo, ao, ls = mh.shape
        lc = mo*ao
        mh1 = mh.reshape((lc, ls), order='Fortran')
        oh1 = oh.reshape((lc, lc), order='Fortran')
        hdim = lc+ls
        h = full.matrix((hdim, hdim))
        h[:ls, :ls] = sh
        h[ls:, :ls] = mh1
        h[:ls, ls:] = mh1.T
        h[ls:, ls:] = oh1
        return h

    def vb_metric_matrix(self):
        sh, mh, mhT, oh = self.vb_metric()
        mo, ao, ls = mh.shape
        lc = mo*ao
        mh1 = mh.reshape((lc, ls), order='Fortran')
        mh1T = mhT.reshape((ls, lc), order='Fortran')
        oh1 = oh.reshape((lc, lc), order='Fortran')
        hdim = lc+ls
        h = full.matrix((hdim, hdim))
        h[:ls, :ls] = sh
        h[ls:, :ls] = mh1
        h[:ls, ls:] = mh1T
        h[ls:, ls:] = oh1
        return h

def is_one_electron(h):
    return is_one_tuple(h) or is_one_array(h)

def is_one_tuple(h):
    return isinstance(h, tuple) and \
        len(h) == 2 and \
        is_one_array(h[0]) and \
        is_one_array(h[1])

def is_one_array(h):
    import numpy
    return isinstance(h, numpy.ndarray) and \
           len(h.shape) == 2 and \
           h.shape[0] == h.shape[1]

def is_two_electron(h):
    pass

if __name__ == "__main__":
    pass
