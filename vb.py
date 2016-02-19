#!/usr/bin/env python
"""Valence Bond energies, gradients, Hessian"""
import os
import math
import numpy as np
from .daltools import one
from .two_electron import two
from .two_electron.two import fockab as Fao
from .daltools.util import full
from .nod import Nod, Dao, Dmo


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
            raise NotImplementedError
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
        d_am = (full.matrix(Nod.C.shape), full.matrix(Nod.C.shape))
        CK = self.K.orbitals()
        for s in (0, 1):
            if self.K(s) and self.L(s):
                d_am[s][:, self.L(s)] = CK[s]*self.transition_density[s]
        return d_am

    @property
    def contravariant_transition_density_mo_ao(self):
        """Return contravariant density matrix in mix mo,ao basis"""
        d_ma = (full.matrix(Nod.C.shape[::-1]), full.matrix(Nod.C.shape[::-1]))
        CL = self.L.orbitals()
        for s in (0, 1):
            if self.K(s) and self.L(s):
                d_ma[s][self.K(s), :] = self.transition_density[s]*CL[s].T
        return d_ma

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
        delta_aa = (S - d for d in self.covariant_density_ao())
        return tuple(delta_aa)

    def co_contravariant_transition_delta(self):
        """
        Return mixed covariant-contravariant ao delta matrix
        """
        S = Nod.S
        I = full.unit(S.shape[0])
        delta_aa = (I - S*d for d in self.transition_ao_density)
        return tuple(delta_aa)

    def contra_covariant_transition_delta(self):
        """
        Return mixed contravariant-covariant ao delta matrix
        """
        S = Nod.S
        S = Nod.S
        I = full.unit(S.shape[0])
        delta_aa = (I - d*S for d in self.transition_ao_density)
        return tuple(delta_aa)

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
        (D(^m, _mu)D(_nu, ^n) + sum(s)D(s)(^m, ^n)delta(s)(_nu,_mu))<K|L>
        delta = S - S*D*S
        """

        D_am = sum(self.co_contravariant_transition_density_ao_mo())
        Dm_a = sum(self.contra_covariant_transition_density_mo_ao())
        ad_mm, bd_mm = self.full_mo_transition_density

        adelta, bdelta = self.covariant_transition_delta()

        dKdL = (
            Dm_a.T.x(D_am)
            + (adelta.x(ad_mm) + bdelta.x(bd_mm)).transpose(1, 2, 0, 3)
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
        delta = self.co_contravariant_transition_delta()

        K_h_dL = (full.matrix(Nod.C.shape), full.matrix(Nod.C.shape))

        CK = self.K.orbitals()
        for s in (0, 1):
            if self.K(s) and self.L(s):
                K_h_dL[s][:, self.L(s)] += \
                    delta[s]*h1[s].T*CK[s]*D_mo[s]*self.overlap()
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
        delta = self.contra_covariant_transition_delta()

        dK_h_La = full.matrix(Nod.C.shape[::-1])
        dK_h_Lb = full.matrix(Nod.C.shape[::-1])

        CL = self.L.orbitals()
        if self.L(0):
            dK_h_La[self.K(0), :] += \
                D_mo[0]*CL[0].T*h1[0].T*delta[0]*self.overlap()
        if self.L(1):
            dK_h_Lb[self.K(1), :] += \
                D_mo[1]*CL[1].T*h1[1].T*delta[1]*self.overlap()

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

        d_am = self.contravariant_transition_density_ao_mo
        delta = self.co_contravariant_transition_delta()
        K_h_d2L += two.vb_transform(
            d_am, delta,
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

        # D^{mn}delta^xi_mu h_{xi,rho}delta_nu^rho
        d_mm = self.full_mo_transition_density
        DhD = self.project_virtual_virtual(*args)
        dK_h_dL += (
            d_mm[0].x(DhD[0])*KL +\
            d_mm[1].x(DhD[1])*KL
            ).transpose(3, 0, 2, 1)

        # delta_{nu, mu} D^{m,rho}h_{xi, rho}D^{xi, n}
        Hmm = self.project_occupied_occupied(*args)
        delta = self.covariant_transition_delta()
        dK_h_dL -= (
            delta[0].x(Hmm[0])*KL + delta[1].x(Hmm[1])*KL
            ).transpose(1, 2, 0, 3)

        return dK_h_dL

    def project_virtual_virtual(self, op):
        """Project virtual-virtual"""
        delta1 = self.co_contravariant_transition_delta()
        delta2 = self.contra_covariant_transition_delta()
        return tuple(d1*h.T*d2 for d1, h, d2 in zip(delta1, op, delta2))

    def project_occupied_occupied(self, op):
        """Project occupied-occupied"""
        d_ma = self.contravariant_transition_density_mo_ao
        d_am = self.contravariant_transition_density_ao_mo
        h1mm = (d_ma[0]*op[0].T*d_am[0], d_ma[1]*op[1].T*d_am[1])
        return h1mm



class Structure(object):
    """
    VB Structure type

    nods: Non-orthogonal determinants
    coef: fix coupling coefficients
    """
    def __init__(self, nods, coef, normalize=True):
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
        self._C = nods[0].C

        if normalize:
            self.normalize()

    @property
    def C(self):
        return self._C

    @C.setter
    def C(self, cmo):
        """Propagate changes in C to determinants"""
        self._C = cmo
        for nod in self.nods:
            nod.C = cmo

    def assert_consistent_electron_number(self):
        """Verify that determinants in structure are consistent"""
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

    def normalize(self):
        self.coef *= 1.0/math.sqrt(self*self)

    def __str__(self):
        output = ["%f    %s" % (c, d) for c, d in zip(self.coef, self.nods)]
        return "\n".join(output)

    def overlap(self):
        "Differentiate structure norm square wrt vb orbitals"
        return self*self

    def overlap_gradient(self):
        "Differentiate structure norm square wrt vb orbitals"
        ds = sum(
            (ck*cl)*BraKet(k, l).overlap_gradient() 
            for k, ck in zip(self.nods, self.coef)
            for l, cl in zip(self.nods, self.coef)
            )
        return ds

class StructError(Exception):
    """General structure exception"""
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

    def structure_hamiltonian(self):
        """Returns Hamiltonian matrix in basis of structures"""
        SH = []
        # Structures left
        for S in self.structs:
            for T in self.structs:
                #Determinants in left structure
                H = 0
                for K, CKS in zip(S.nods, S.coef):
                    for L, CLT in zip(T.nods, T.coef):
                        KL = BraKet(K, L)
                        H += CKS*CLT*(KL.energy((self.h, self.h)) + self.Z)*KL.overlap()
                SH.append(H)
        LS = len(self.structs)
        return full.init(SH).reshape((LS, LS))

    def eigenvalues_vectors(self):
        self.normalize()
        S = self.structure_overlap()
        H = self.structure_hamiltonian()
        e, V = (H/S).eigvec()
        for v in V.T:
            v.normalize(S)
        return e, V

    def structure_overlap(self):
        """Calculate structure overlap matrix"""
        return full.init([[s*t for t in self.structs] for s in self.structs])

    def structure_weights(self):
        """
        Returns structure weights
        w_S = C(S) sum(T) <S|T>C(T)
        """
        SO = self.structure_overlap()
        C = full.init(self.coef)
        SOC = SO*C
        W = [c*sc for c, sc in zip(C, SOC)]
        return W

    def normalize(self):
        """
        Normalize state
        """
        self.normalize_mo()
        self.normalize_structures()
        self.coef *= 1/math.sqrt(self.norm())

    def normalize_mo(self):
        """
        Normalize mo
        """
        _, mo = self.C.shape
        for i in range(mo):
            cmo = self.C[:, i]
            nmo = 1/math.sqrt(cmo.T*Nod.S*cmo)
            cmo *= nmo

    def normalize_structures(self):
        """
        Normalize structures and adjust coefficient so the
        overall state is preserved
        """

        for i, s in enumerate(self.structs):
            ns = math.sqrt(s*s)
            s.coef /= ns
            self.coef[i] *= ns

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
        """
        #  N = <0|0>
        # dN = 2<d0|0>
        """
        NGS = []
        r, c = Nod.C.shape
        n_orb_grad = full.matrix((r, c))
        #
        for S, CS in zip(self.structs, self.coef):
            GS = 0
            for T, CT in zip(self.structs, self.coef):
                GS += (S*T)*CT
                for K, CK in zip(S.nods, S.coef):
                    for L, CL in zip(T.nods, T.coef):
                        n_orb_grad += (CS*CT*CK*CL) * \
                            BraKet(K, L).right_overlap_gradient()
            NGS.append(GS)

        n_struct_grad = full.init(NGS)
        n_struct_grad *= 2
        n_orb_grad *= 2
        return (n_struct_grad, n_orb_grad[:, :])


    def normhess(self):
        """
        Numerical norm Hessian
        #  N = <0|0>
        # d2<0|0> = <d20|0> + <0|d20> + 2<d0|d0> = 2<0|d20> + 2<d0|d0>
        """
        n_struct_hess = full.matrix(self.coef.shape*2)
        n_orb_structhess = full.matrix(Nod.C.shape + self.coef.shape)
        n_orb_hess = full.matrix(Nod.C.shape*2)
        #
        for s1, (str1, cstr1) in enumerate(zip(self.structs, self.coef)):
            for s2, (str2, cstr2) in enumerate(zip(self.structs, self.coef)):
                for det1, cdet1 in zip(str1.nods, str1.coef):
                    for det2, cdet2 in zip(str2.nods, str2.coef):
                        #
                        bk12 = BraKet(det1, det2)
                        #
                        n_struct_hess[s1, s2] += (cdet1*cdet2)*bk12.overlap()
                        n_orb_structhess[:, :, s1] += \
                            cdet1*cstr2*cdet2*bk12.overlap_gradient()
                        n_orb_hess += \
                            cstr1*cdet1*cstr2*cdet2*bk12.overlap_hessian()
        n_struct_hess *= 2
        n_orb_structhess *= 2
        #n_orb_hess
        return n_struct_hess, n_orb_structhess, n_orb_hess


    def energy(self):
        """Returns total electronic energy"""
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

        E = H/N
        return E

    def energygrad(self):
        """
        #  N = <0|0>
        # dN = 2<d0|0>
        #  E = <0|H|0>/<0|0>
        # dE = 2<d0|H-E|0>/<0|0>
        """
        h_struct_grad = full.matrix(len(self.structs))
        h_orb_grad = full.matrix(Nod.C.shape)

        for s1, (str1, cstr1) in enumerate(zip(self.structs, self.coef)):
            for str2, cstr2 in zip(self.structs, self.coef):
                for det1, cdet1 in zip(str1.nods, str1.coef):
                    for det2, cdet2 in zip(str2.nods, str2.coef):
                        det12 = BraKet(det1, det2)

                        h_struct_grad[s1] += \
                            2*(cdet1*cstr2*cdet2)*\
                            det12.overlap()*\
                            det12.energy((self.h, self.h))

                        h_orb_grad += \
                            cstr1*cdet1*cstr2*cdet2*\
                            det12.energy_gradient((self.h, self.h))

        N = self.norm()
        E = self.energy()

        n_struct_grad, n_orb_grad = self.normgrad()

        structgrad = (1/N)*(h_struct_grad - E*n_struct_grad)
        orbgrad = (1/N)*(h_orb_grad - E*n_orb_grad)
        return (structgrad, orbgrad[:, :])


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

        h_struct_hess = full.matrix(self.coef.shape*2)
        h_orb_structhess = full.matrix(Nod.C.shape + self.coef.shape)
        h_orb_hess = full.matrix(Nod.C.shape*2)
        #
        for s1, (str1, Cs1) in enumerate(zip(self.structs, self.coef)):
            for s2, (str2, Cs2) in enumerate(zip(self.structs, self.coef)):
                for det1, Cd1 in zip(str1.nods, str1.coef):
                    for det2, Cd2 in zip(str2.nods, str2.coef):
                        #
                        det12 = BraKet(det1, det2)
                        #
                        # Structure gradient terms
                        #
                        h_struct_hess[s1, s2] += \
                            (Cd1*Cd2)*det12.overlap()*\
                            det12.energy((self.h, self.h))
                        #
                        # Orbital-structure hessian terms
                        #
                        h_orb_structhess[:, :, s1] += \
                            Cd1*Cs2*Cd2*det12.energy_gradient((self.h, self.h))
                        ##
                        # Orbital-orbital hessian
                        #
                        h_orb_hess += Cs1*Cd1*Cs2*Cd2*(
                            det12.right_1el_energy_hessian((self.h, self.h)) +
                            det12.mixed_1el_energy_hessian((self.h, self.h)) +
                            det12.right_2el_energy_hessian() +
                            det12.mixed_2el_energy_hessian()
                            )
                        #

        N = self.norm()
        E = self.energy()

        n_struct_grad, n_orb_grad = self.normgrad()
        e_struct_grad, e_orb_grad = self.energygrad()

        n_struct_hess, n_orb_structhess, n_orb_hess = self.normhess()
        h_struct_hess *= 2
        h_orb_structhess *= 2
        h_orb_hess *= 2

        e_struct_hess = (
            h_struct_hess - E*n_struct_hess -
            e_struct_grad.x(n_struct_grad) - n_struct_grad.x(e_struct_grad)
            )/N

        e_orb_structhess = (
            h_orb_structhess - E*n_orb_structhess - \
            e_orb_grad.x(n_struct_grad) - n_orb_grad.x(e_struct_grad)
            )/N

        e_orb_hess = (
            h_orb_hess - E*n_orb_hess - e_orb_grad.x(n_orb_grad)-n_orb_grad.x(e_orb_grad))/N

        return (
            e_struct_hess,
            e_orb_structhess,
            e_orb_hess,
            )

    def gradientvector(self):
        """Full energy gradient as vector"""
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
        """Gradient norm"""
        gv = self.gradientvector()
        G = self.vb_metric_matrix()
        return gv*G.I*gv

    def hessianmatrix(self):
        """Energy Hessian as matrix"""
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
        """VB metric?"""
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
    """True if arg one-electron operator type"""
    return is_one_tuple(h) or is_one_array(h)

def is_one_tuple(h):
    """True if arg one-electron operator tuple"""
    return isinstance(h, tuple) and \
        len(h) == 2 and \
        is_one_array(h[0]) and \
        is_one_array(h[1])

def is_one_array(h):
    """True if arg one-electron operator array"""
    import numpy
    return isinstance(h, numpy.ndarray) and \
           len(h.shape) == 2 and \
           h.shape[0] == h.shape[1]

def is_two_electron(*args):
    """To be implmeneted"""
    raise NotImplementedError

if __name__ == "__main__":
    pass
