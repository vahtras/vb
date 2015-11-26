#!/usr/bin/env python
"""Valence Bond energies, gradients, Hessian"""
import os
import sys
import math
from daltools import one
from two_electron import two
from two_electron.two import fockab as Fao
from daltools.util import full

DELTA = 1e-4

class BraKet(object):
    """Non-orthogonal determinant pairs"""

    def __init__(self, K, L):
        self.K = K
        self.L = L
        self._td = None
        self._ftd = None
        self._aotd = None
        self.tmpdir = '/tmp'

    def __str__(self):
        return "<%s|...|%s>" % (self.K, self.L)

    def tmp(self, filename):
        return os.path.join(self.tmpdir, filename)

    def __mul__(self, h):
        if is_one_electron(h):
            return self.energy(h)*self.overlap()
        elif is_two_electron(h):
            raise Exception("Not implemented")
        else:
            raise Exception("Unknown multiplicator")

    def overlap(self):
        return self.K*self.L

    def energy(self, h):
        dao = self.transition_ao_density
        ha, hb = h
        return (ha&dao[0]) + (hb&dao[1])

### Densities

    @property
    def transition_density(self):
        if self._td is None:
            self._td = Dmo(self.K, self.L)
        return self._td

    @property
    def contravariant_transition_density_ao_mo(self):
        Dam = (full.matrix(Nod.C.shape), full.matrix(Nod.C.shape))
        CK = self.K.orbitals()
        for s in (0, 1):
            if self.K(s) and self.L(s):
                Dam[s][:, self.L(s)] = CK[s]*self.transition_density[s]
        return Dam

    @property
    def contravariant_transition_density_mo_ao(self):
        Dma = (full.matrix(Nod.C.shape[::-1]), full.matrix(Nod.C.shape[::-1]))
        CL = self.L.orbitals()
        for s in (0, 1):
            if self.K(s) and self.L(s):
                Dma[s][self.K(s), :] = self.transition_density[s]*CL[s].T
        return Dma

    def co_contravariant_transition_density_ao_mo(self):
        return tuple(Nod.S*d for d in self.contravariant_transition_density_ao_mo)

    def contra_covariant_transition_density_mo_ao(self):
        return tuple(d*Nod.S for d in self.contravariant_transition_density_mo_ao)


    @property
    def full_mo_transition_density(self):
        if self._ftd is None:
            ao, mo = Nod.C.shape
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
        return Dao(self.K, self.L)

    def covariant_density_ao(self):
        S = Nod.S
        return tuple(S*d*S for d in self.transition_ao_density)

    def covariant_transition_delta(self):
        S = Nod.S
        Delta_aa = (S - d for d in self.covariant_density_ao())
        return tuple(Delta_aa)

    def co_contravariant_transition_delta(self):
        S = Nod.S
        I = full.unit(S.shape[0])
        Delta_aa = (I - S*d for d in self.transition_ao_density)
        return tuple(Delta_aa)

    def contra_covariant_transition_delta(self):
        S = Nod.S
        I = full.unit(S.shape[0])
        Delta_aa = (I - d*S for d in self.transition_ao_density)
        return tuple(Delta_aa)

### Fock

    @property
    def transition_ao_fock(self):
        FKL = tuple(f.view(full.matrix) 
            for f in Fao(
                self.transition_ao_density, filename=self.tmp('AOTWOINT')
                )
            )
        return FKL

### Overlap differentiation

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

    def right_1el_energy_gradient(self, h1):
        return sum(self.right_1el_energy_gradient_ab(h1))

    def right_1el_energy_gradient_ab(self, h1):
        eg1_a, eg1_b = (self.energy(h1)*g for g in self.right_overlap_gradient_ab())
        eg2_a, eg2_b = self.project_virtual_occupied(h1)
        eg_ab = (eg1_a + eg2_a, eg1_b + eg2_b)
        return eg_ab

    def project_virtual_occupied(self, h1):
        """Rhs derivative <K|h|dL/dC(mu, m)>"""

        Dmo = self.transition_density
        Delta = self.co_contravariant_transition_delta()

        K_h_dL = (full.matrix(Nod.C.shape), full.matrix(Nod.C.shape))

        CK = self.K.orbitals()
        for s in (0, 1):
            if self.K(s) and self.L(s):
                K_h_dL[s][:, self.L(s)] += Delta[s]*h1[s].T*CK[s]*Dmo[s]*self.overlap()
        return K_h_dL


    def left_1el_energy_gradient(self, h1):
        return sum(self.left_1el_energy_gradient_ab(h1))

    def left_1el_energy_gradient_ab(self, h1):
        eg1_a, eg1_b = (self.energy(h1)*g for g in self.left_overlap_gradient_ab())
        eg2_a, eg2_b = self.project_occupied_virtual(h1)
        eg_ab = (eg1_a + eg2_a, eg1_b + eg2_b)
        return eg_ab

    def project_occupied_virtual(self, h1):
        """Lhs derivative <dK/dC(mu,m)|h|L>"""

        Dmo = self.transition_density
        Delta = self.contra_covariant_transition_delta()

        dK_h_La = full.matrix(Nod.C.shape[::-1])
        dK_h_Lb = full.matrix(Nod.C.shape[::-1])

        CL = self.L.orbitals()
        if self.L(0):
            dK_h_La[self.K(0), :] += Dmo[0]*CL[0].T*h1[0].T*Delta[0]*self.overlap()
        if self.L(1):
            dK_h_Lb[self.K(1), :] += Dmo[1]*CL[1].T*h1[1].T*Delta[1]*self.overlap()
        
        return dK_h_La.T, dK_h_Lb.T

    def twoel_energy(self):
        DKL = self.transition_ao_density
        FKL = self.transition_ao_fock
        return .5*((FKL[0]&DKL[0]) + (FKL[1]&DKL[1]))

    def twoel_tme(self):
        """<K|g|L>"""
        return self.twoel_energy()*self.overlap()

    def right_2el_energy_gradient(self):
        return sum(self.right_2el_energy_gradient_ab())

    def right_2el_energy_gradient_ab(self):
        eg1_a, eg1_b = (self.twoel_energy()*g for g in self.right_overlap_gradient_ab())
        eg2_a, eg2_b = self.project_virtual_occupied(self.transition_ao_fock)
        eg_ab = (eg1_a + eg2_a, eg1_b + eg2_b)
        return eg_ab

    def right_2el_energy_gradient_ab2(self):
        """Rhs derivative <K|g|dL/dC(mu, m)>"""
        Fao = self.transition_ao_fock
        return self.project_virtual_occupied(Fao)

    def left_2el_energy_gradient(self):
        return sum(self.left_2el_energy_gradient_ab())

    def left_2el_energy_gradient_ab(self):
        eg1_a, eg1_b = self.left_2el_energy_gradient_ab1()
        eg2_a, eg2_b = self.left_2el_energy_gradient_ab2()
        eg_ab = (eg1_a + eg2_a, eg1_b + eg2_b)
        return eg_ab

    def left_2el_energy_gradient_ab1(self):
        fab = self.transition_ao_fock
        dK_h_La, dK_h_Lb = (
            0.5*self.energy(fab)*g 
            for g in self.left_overlap_gradient_ab()
            )
        return dK_h_La, dK_h_Lb 

    def left_2el_energy_gradient_ab2(self):
        """Rhs derivative <dK/dC(mu, m)|g|L>"""
        Fao = self.transition_ao_fock
        return self.project_occupied_virtual(Fao)

    

    def right_1el_energy_hessian(self, h1):
        K_h_d2L = self.right_energy_hessian(
            self.energy,
            self.project_virtual_occupied,
            h1
            )
        return K_h_d2L

    def right_2el_energy_hessian(self):
        K_h_d2L = self.right_energy_hessian(
            self.twoel_energy,
            self.right_2el_energy_gradient_ab2
            )

        Dam = self.contravariant_transition_density_ao_mo
        Delta = self.co_contravariant_transition_delta()
        K_h_d2L += two.vb_transform(Dam, Delta)*self.overlap()

        return K_h_d2L

    def right_energy_hessian(self, energy, right_gradient, *args):
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
        dK_h_dL = self.mixed_gen_hessian(
            h1, 
            self.energy(h1), 
            self.project_occupied_virtual,
            self.project_virtual_occupied
            )
        return dK_h_dL

    def mixed_2el_energy_hessian(self):
        dK_g_dL = self.mixed_gen_hessian(
            self.transition_ao_fock,
            self.twoel_energy(),
            self.left_2el_energy_gradient,
            self.right_2el_energy_gradient
            )


        dK_g_dL += two.vb_transform2(
            self.contravariant_transition_density_mo_ao,
            self.contravariant_transition_density_ao_mo,
            self.contra_covariant_transition_delta,
            self.co_contravariant_transition_delta
            )
        return dK_g_dL

    def mixed_gen_hessian(self, h1, e1, left_gradient, right_gradient):
        """
        L-R derivative <dK/dC(mu,m)|h|dL/dC(nu,n)>

        <K|a^m+ a_\mu h a\nu+ a^n|L>
        """

        S = Nod.S
        KL = self.overlap()
        D_KL = self.transition_density

        # <h><K|a^m+ a_\mu h a_nu+ a^n|L>
        dK_dL = self.mixed_overlap_hessian()

        dK_h_dL = e1*dK_dL

        # D^m_mu<K|H|dL/dC^nu_n> + <dK/dC^mu_m|h|L>D_nu^n
        dK_L = self.left_overlap_gradient()
        K_dL = self.right_overlap_gradient()
        K_h_dL = sum(right_gradient(h1))
        dK_h_L = sum(left_gradient(h1))
        
        dK_h_dL += (dK_L.x(K_h_dL) + dK_h_L.x(K_dL))/KL

        # D^{mn}Delta^xi_mu h_{xi,rho}Delta_nu^rho
        ao, mo = Nod.C.shape
        Dmm = self.full_mo_transition_density
        Delta1 = self.co_contravariant_transition_delta()
        Delta2 = self.contra_covariant_transition_delta()

        dK_h_dL += (
            Dmm[0].x(Delta1[0]*h1[0].T*Delta2[0])*KL + \
            Dmm[1].x(Delta1[1]*h1[1].T*Delta2[1])*KL
            ).transpose(3, 0, 2, 1)

        # Delta_{nu, mu} D^{m,rho}h_{xi, rho}D^{xi, n}
        CK = self.K.orbitals()
        CL = self.L.orbitals()

        h1mm = (full.matrix((mo, mo)), full.matrix((mo, mo)))
        if self.K(0) and self.L(0):
            (D_KL[0]*CL[0].T*h1[0].T*CK[0]*D_KL[0]).scatter(
            h1mm[0], rows=self.K(0), columns=self.L(0)
            )
        if self.K(1) and self.L(1):
            (D_KL[1]*CL[1].T*h1[1].T*CK[1]*D_KL[1]).scatter(
            h1mm[1], rows=self.K(1), columns=self.L(1)
            )
        Delta = self.covariant_transition_delta()
        dK_h_dL -= (
            Delta[0].x(h1mm[0])*KL + Delta[1].x(h1mm[1])*KL
            ).transpose(1, 2, 0, 3)

        return dK_h_dL

        
#
# Class of non-orthogonal determinants
#
class Nod(object):
    #
    #
    # Class global variables
    #
    S = None   #Overlap ao basis
    C = None   #VB orbital coefficients

    def __init__(self, astring, bstring, C=None, tmpdir='/tmp'):
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
        #
        # Read overlap at first invocation of constructor
        #
        aooneint = os.path.join(tmpdir, 'AOONEINT')
        if Nod.S is None:
            Nod.S = one.read('OVERLAP', aooneint).unpack().unblock()
            #
            # init unit for H2 test
            #

    def electrons(self):
        return len(self.a + self.b)

    def __call__(self, s):
        #
        # Returns alpha (s=0) or beta (s=1) string
        #
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
        CUa = None
        CUb = None
        if self.a: CUa = self.C[:, self.a]
        if self.b: CUb = self.C[:, self.b]
        return (CUa, CUb)

    def __mul__(self, other):
        #
        # Return overlap of two Slater determinants <K|L>
        # calculated as matrix determinant of overlap
        # of orbitals in determinants
        # det(S) = det S(alpha)*det S(beta)
        #
        #
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
        return DKL(self, self, mo=0)

    def mo_density(self):
        return DKL(self, self, mo=1)


#
# Calculate transition density from unnormalized molecular orbitals
#
def Dao(K, L):
    return DKL(K, L, mo=0)

def Dmo(K, L):
    return DKL(K, L, mo=1)

def DKL(K, L, mo=0):
    #
    # Return intermediate normalized transition density matrix given
    # determinants K and L
    # as [Dalpha, Dbeta]
    # default: ao basis
    # optional: mo non-zero, gives in terms of included orbitals
    #           i.e. compact form which has to be scattered to a
    #           general mo basis
    #

    CK = K.orbitals()
    CL = L.orbitals()
    #
    D = []

    for s in range(2):
        if CK[s] is None or CL[s] is None:
            #
            # if None orbitals set atomic density to zero matrix
            #
            if mo:
                D.append(None)
            else:
                D.append(full.matrix(Nod.S.shape))
        else:
            SLK = CL[s].T*Nod.S*CK[s]
            #
            # Density is inverse transpose
            #
            if mo:
                D.append(SLK.inv())
            else:
                D.append(CK[s]*(CL[s].T/SLK))

    return D


def HKL(F, D):
    E = 0.5*((F[0]&D[0]) + (F[1]&D[1]))
    return E

class Structure(object):

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

    def __init__(self, structs, coef, VBSCF=True, tmpdir='/tmp', frozen=[]):
        self.structs = structs
        self.coef = full.init(coef)
        self.tmpdir = tmpdir
        self.Z = one.readhead(tmpdir+"/AOONEINT")["potnuc"]
        self.h = one.read("ONEHAMIL", tmpdir+"/AOONEINT").unpack().unblock()
        self.frozen = frozen
        #
        # For VBSCF all structures share orbitals
        #
        if VBSCF:
            self.C = Nod.C
            nao, nmo = self.C.shape
        else:
            raise Exception("not implemented")

    def tmp(self, filename):
        return os.path.join(self.tmpdir, filename)

    def __str__(self):
        retstr = "\n"
        for i, coef in enumerate(self.structs):
            retstr += self.coef.fmt%coef + "   " + "(%d)\n"%i
        return retstr

    def nel(self):
        Nel = 0
        N = 0
        for S, CS in zip(self.structs, self.coef):
            for T, CT in zip(self.structs, self.coef):
                for K, CK in zip(S.nods, S.coef):
                    for L, CL in zip(T.nods, T.coef):
                        KL = BraKet(K, L)
                        D12 = KL.transition_ao_density
                        S12 = KL.overlap()
                        Na = (D12[0]&Nod.S)
                        Nb = (D12[1]&Nod.S)
                        C12 = CS*CT*CK*CL
                        Nel += (Na+Nb)*C12
                        N += C12
        return Nel/N

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
        """Calculate norm square of VB wave function"""
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
        # Structures left
        #
        for S, CS in zip(self.structs, self.coef):
            GS = 0
            #
            #Structures right
            #
            for T, CT in zip(self.structs, self.coef):
                #
                #Determinants in left structure
                #
                for K, CKS in zip(S.nods, S.coef):
                    #
                    #Determinants in right structure
                    #
                    for L, CLT in zip(T.nods, T.coef):
                        K_L = BraKet(K, L)
                        #
                        # Structure gradient terms
                        #
                        KL = K_L.overlap()
                        #
                        # Structure gradient terms
                        #
                        GS += CKS*CT*CLT*KL
                        #
                        # Orbital gradient terms
                        #
                        Norbgrad += CS*CT*CKS*CLT*K_L.right_overlap_gradient()
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
        for s1, (str1, Cs1) in enumerate(zip(self.structs, self.coef)):
            for s2, (str2, Cs2) in enumerate(zip(self.structs, self.coef)):
                for d1, (det1, Cd1) in enumerate(zip(str1.nods, str1.coef)):
                    for d2, (det2, Cd2) in enumerate(zip(str2.nods, str2.coef)):
                        #
                        bk12 = BraKet(det1, det2)
                        C1 = Cs1*Cd1
                        C2 = Cs2*Cd2
                        #
                        Nstructhess[s1, s2] += (Cd1*Cd2)*bk12.overlap()
                        Norbstructhess[:, :, s1] += Cd1*C2*bk12.overlap_gradient()
                        Norbhess += C1*C2*bk12.norm_overlap_hessian()
        Nstructhess *= 2
        Norbstructhess *= 2
        Norbhess *= 2
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
        # Structures left
        for str1, cs1 in zip(self.structs, self.coef):
            #Structures right
            for str2, cs2 in zip(self.structs, self.coef):
                #Determinants in left structure
                for det1, cd1 in zip(str1.nods, str1.coef):
                    #Determinants in right structure
                    for det2, cd2 in zip(str2.nods, str2.coef):
                        BK12 = BraKet(det1, det2)
                        D12 = BK12.transition_ao_density
                        S12 = BK12.overlap()
                        F12 = BK12.transition_ao_fock
                        h12 = self.h&(D12[0]+D12[1])
                        #h12 = Braket(det1, det2).trace(self.h1)
                        g12 = HKL(F12, D12)
                        C12 = cs1*cs2*cd1*cd2*S12
                        Eone += h12*C12
                        Etwo += g12*C12
                        N += C12

        H = Eone+Etwo
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
        #d12 = full.matrix((mo, mo))
        #
        # Structures left
        #
        for s1 in range(len(self.structs)):
            str1 = self.structs[s1]
            #
            #Structures right
            #
            for s2 in range(len(self.structs)):
                str2 = self.structs[s2]
                #
                #Determinants in left structure
                #
                for d1 in range(len(str1.nods)):
                    det1 = str1.nods[d1]
                    #
                    #Determinants in right structure
                    #
                    for d2 in range(len(str2.nods)):
                        det2 = str2.nods[d2]
                        #
                        # Structure gradient terms
                        #
                        S12 = det1*det2
                        Cs1 = self.coef[s1]
                        Cs2 = self.coef[s2]
                        Cd1 = str1.coef[d1]
                        Cd2 = str2.coef[d2]
                        C1 = Cs1*Cd1
                        C2 = Cs2*Cd2
                        #
                        # Structure gradient terms
                        #
                        N12 = (Cd1*C2)*S12
                        Nstructgrad[s1] += N12
                        D12 = Dao(det1, det2)
                        F12 = Fao(D12, filename=self.tmp('AOTWOINT'))
                        #
                        H12 = self.h&(D12[0]+D12[1])
                        H12 += 0.5*((F12[0]&D12[0]) + (F12[1]&D12[1]))
                        Hstructgrad[s1] += N12*H12
                        C12 = (C1*C2)*S12
                        N += C12
                        H += C12*H12

                        #
                        # Orbital gradient terms
                        #
                        CK = det1.orbitals()
                        CL = det2.orbitals()
                        Dmo12 = Dmo(det1, det2)
                        #
                        Sog12 = [None, None]
                        Sog21 = [None, None]
                        Hog12 = [None, None]
                        Hog21 = [None, None]
                        for s in range(2):
                            Sog12[s] = Dmo12[s]*CL[s].T*Nod.S
                            Hog12[s] = Dmo12[s]*CL[s].T*(
                                (self.h+F12[s]).T*(I-D12[s]*Nod.S)
                                ) + Sog12[s]*H12
                        #
                        # Scatter to orbitals
                        #
                            ((C1*C2*S12)*Sog12[s]).T.scatteradd(Norbgrad, columns=det1(s))
                            ((C1*C2*S12)*Hog12[s]).T.scatteradd(Horbgrad, columns=det1(s))
                        #
                        #   F12[s].T*(I-D12[s]*S/S12)+H12*S/S12
                        #
        E = H/N # only electronic
        #print "energygrad:E, H, N", E, H, N
        structgrad = (2/N)*(Hstructgrad-E*Nstructgrad)
        orbgrad = (2/N)*(Horbgrad-E*Norbgrad)
        return (structgrad, orbgrad[:, :])

    def numenergygrad(self, delta=1e-3):
        return self.numgrad(self.energy, delta)

    def energyhess(self):
        #
        #  N = <0|0>
        # dN = 2<d0|0>
        #
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
        dm_a = [None, None]
        d_am = [None, None]
        Hog12 = [None, None]
        Hog21 = [None, None]
        dma = [None, None]
        dam = [None, None]
        Dm_a = [full.matrix((mo, ao)), full.matrix((mo, ao))]
        D_am = [full.matrix((ao, mo)), full.matrix((ao, mo))]
        Dog21 = [full.matrix((mo, ao)), full.matrix((mo, ao))]
        I = full.unit(ao)
        h = self.h
        S = Nod.S
        #
        # Structures left
        #
        for s1 in range(len(self.structs)):
            str1 = self.structs[s1]
            #
            #Structures right
            #
            for s2 in range(len(self.structs)):
                str2 = self.structs[s2]
                #
                #Determinants in left structure
                #
                for d1 in range(len(str1.nods)):
                    det1 = str1.nods[d1]
                    #
                    #Determinants in right structure
                    #
                    for d2 in range(len(str2.nods)):
                        det2 = str2.nods[d2]
                        #
                        # Coefficients
                        #
                        Cs1 = self.coef[s1]
                        Cs2 = self.coef[s2]
                        Cd1 = str1.coef[d1]
                        Cd2 = str2.coef[d2]
                        C1 = Cs1*Cd1
                        C2 = Cs2*Cd2
                        #
                        # Determinant overlap
                        #
                        S12 = det1*det2
                        #
                        # Fock and density
                        #
                        D12 = Dao(det1, det2)
                        F12 = Fao(D12, filename=self.tmp('AOTWOINT'))
                        #
                        # Energy and norm
                        #
                        C12 = (C1*C2)*S12
                        hKL = self.h&(D12[0]+D12[1])
                        H12 = hKL+HKL(F12, D12)
                        N += C12
                        H += C12*H12
                        #
                        # Structure gradient terms
                        #
                        N12 = (Cd1*C2)*S12
                        Nstructgrad[s1] += N12
                        Hstructgrad[s1] += N12*H12
                        #
                        # Orbital gradient terms
                        #
                        CK = det1.orbitals()
                        CL = det2.orbitals()
                        dmm = Dmo(det1, det2)
                        for s in range(2):
                            dm_a[s] = dmm[s]*CL[s].T*S
                            d_am[s] = S*CK[s]*dmm[s]
                            Hog12[s] = dmm[s]*CL[s].T*(
                                (h+F12[s]).T*(I-D12[s]*S)
                                ) + dm_a[s]*H12
                            Hog21[s] = dmm[s].T*CK[s].T*(
                                (h+F12[s])*(I-D12[s].T*S)
                                ) + d_am[s].T*H12
                            Norbgrad[:, det1(s)] += (C1*C2*S12)*dm_a[s].T
                            Horbgrad[:, det1(s)] += (C1*C2*S12)*Hog12[s].T
                        #
                        # Structure hessian terms
                        #
                        N12 = (Cd1*Cd2)*S12
                        Nstructhess[s1, s2] += N12
                        Hstructhess[s1, s2] += N12*H12
                        #
                        # Orbital-structure hessian terms
                        #
                        Dmm = [full.matrix((mo, mo)), full.matrix((mo, mo))]
                        Dma = [full.matrix((mo, ao)), full.matrix((mo, ao))]
                        Dam = [full.matrix((ao, mo)), full.matrix((ao, mo))]
                        Delta = [full.matrix((ao, ao)), full.matrix((ao, ao))]
                        Delta1 = [full.matrix((ao, ao)), full.matrix((ao, ao))]
                        Delta2 = [full.matrix((ao, ao)), full.matrix((ao, ao))]
                        for s in range(2):
                            #D^m_\mu
                            Dm_a[s].clear()
                            D_am[s].clear()
                            Dm_a[s][det1(s), :] = dm_a[s]
                            D_am[s][:, det2(s)] = d_am[s]
                            #Dmm[s][det1(s), det2(s)]=dmm[s] numpy bug
                            dmm[s].scatter(
                                Dmm[s], rows=det1(s), columns=det2(s)
                                )
                            Delta1[s] = I-CK[s]*dmm[s]*CL[s].T*S
                            Delta2[s] = I-S*CK[s]*dmm[s]*CL[s].T
                            Delta[s] = S*Delta1[s]
                            dma[s] = dmm[s]*CL[s].T
                            dam[s] = CK[s]*dmm[s]
                            Dma[s][det1(s), :] = dma[s]
                            Dam[s][:, det2(s)] = dam[s]
                        #
                        # Scatter
                        #
                            Nd12 = Cd1*C2*S12
                            Norbstructhess[:, det1(s), s1] += Nd12*dm_a[s].T
                            Norbstructhess[:, det2(s), s1] += Nd12*d_am[s]
                            Horbstructhess[:, det1(s), s1] += Nd12*Hog12[s].T
                            Horbstructhess[:, det2(s), s1] += Nd12*Hog21[s].T

                        #print "Dmm", Dmm[0], Dmm[1]
                        #print "Dam", Dam[0], Dam[1]
                        #print "Dma", Dma[0], Dma[1]
                        ##
                        # Orbital-orbital hessian
                        #
                        C12 = C1*C2*S12
                        #
                        # Norm
                        #
                        for s in range(2):
                            #print "Dog12[%d]"%s, Dog12[s]
                            for t in range(2):
                                #print "Dog21[%d]"%t, Dog21[t]
                                Norbhess += C12*Dm_a[s].x(Dm_a[t]+D_am[t].T)
                            Norbhess -= C12*Dm_a[s].x(Dm_a[s]).transpose(
                                (0, 3, 2, 1)
                                )
                            Norbhess += C12*Dmm[s].x(Delta[s]).transpose(
                                (0, 3, 1, 2)
                                )
                        #
                        # Hamiltonian
                        #
                        for s in range(2):
                            for t in range(2):
                                #1
                                Horbhess += H12*C12*Dm_a[s].x(
                                    Dm_a[t]+D_am[t].T
                                    )
                                #3
                                Horbhess += C12*(
                                    Dma[s]*(h+F12[s].T)*Delta1[s]
                                    ).x(Dm_a[t]+D_am[t].T)
                                #5
                                Horbhess += C12*Dm_a[s].x(
                                    Dma[t]*(h+F12[t].T)*Delta1[t]
                                    )
                                Horbhess += C12*Dm_a[s].x(
                                    (Delta2[t]*(h+F12[t].T)*Dam[t]).T
                                    )
                                #
                                #non-Fock contributions
                                #
                                # <Kmn/pq|g|L>
                                #
                                Htmp = two.semitransform(
                                    Dma[s], Dma[t], same=False,
                                    file=self.tmp('AOTWOINT')
                                    ).view(full.matrix) # returns (m, m, a, a)
                                # transpose due to numpy feature
                                left = (1, 2, 0, 3)
                                Htmp = (
                                    Delta1[s].T*Htmp
                                    ).transpose(left)*Delta1[t]
                                if s == t:
                                    # add Exchange
                                    Htmp = Htmp - Htmp.transpose(
                                        (0, 1, 3, 2)
                                        )
                                #
                                # <Km/p|g|Lm/n>
                                #
                                Htmp += (
                                    Delta1[s].T*two.semitransform(
                                        Dma[s], Dam[t].T,
                                        same=False, file=self.tmp('AOTWOINT')
                                        ).view(full.matrix)
                                    ).transpose(left)*Delta2[t].T
                                if s == t:
                                    Htmp = Htmp - (
                                        Delta1[s].T*two.semitransform(
                                            Dma[s], Dam[s].T,
                                            same=True,
                                            file=self.tmp('AOTWOINT')
                                            ).view(full.matrix)).transpose(
                                                left
                                                )*Delta2[s].T
                                #
                                # semitransformed in form (m, m, a, a)
                                # add to hessian (m, a, m, a)
                                #
                                Horbhess += C12*Htmp.transpose((0, 2, 1, 3))

                            #2
                            Horbhess -= H12*C12*Dm_a[s].x(
                                Dm_a[s]
                                ).transpose((0, 3, 2, 1))
                            Horbhess += H12*C12*Dmm[s].x(
                                Delta[s]).transpose((0, 3, 1, 2))
                            #4
                            #m,nu x n mu
                            Horbhess -= C12*Dm_a[s].x(
                                Dma[s]*(h+F12[s].T)*Delta1[s]
                                ).transpose((0, 3, 2, 1))
                            Horbhess += C12*Dmm[s].x(
                                Delta2[s]*(
                                    h+F12[s].T)*Delta1[s]
                                ).transpose((0, 3, 1, 2))
                            #6
                            #n, mu x m nu
                            Horbhess -= C12*Dm_a[s].x(
                                Dma[s]*(h+F12[s].T)*Delta1[s]
                                ).transpose((2, 1, 0, 3))
                            Horbhess -= C12*(
                                Dma[s]*(h+F12[s].T)*Dam[s]
                                ).x(Delta[s]).transpose((0, 3, 1, 2))
                        #
        Hstructgrad *= 2
        Horbgrad *= 2
        Hstructhess *= 2
        Horbstructhess *= 2
        Horbhess *= 2
        Nstructgrad *= 2
        Norbgrad *= 2
        Nstructhess *= 2
        Nstructorbhess *= 2
        Norbstructhess *= 2
        Norbhess *= 2
        E = H/N
        Estructgrad = (Hstructgrad-E*Nstructgrad)/N
        Eorbgrad = (Horbgrad-E*Norbgrad)/N
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
