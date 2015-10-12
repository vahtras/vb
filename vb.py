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

class NodPair(object):
    """Non-orthogonal determinant pairs"""

    def __init__(self, K, L):
        self.K = K
        self.L = L

    def overlap(self):
        return self.K*self.L

    def right_numerical_differential(self, mu, m):
        """Rhs numerical derivative <K|dL/dC(mu, m)>"""
        self.L.C = self.L.C.copy()
        self.L.C[mu, m] += DELTA/2
        KLp = self.K*self.L
        self.L.C[mu, m] -= DELTA
        KLm = self.K*self.L
        self.L.C[mu, m] += DELTA/2
        return (KLp - KLm)/DELTA

    def left_numerical_differential(self, mu, m):
        """Lhs numerical derivative <dK/dC(mu, m)|L>"""
        self.K.C = self.K.C.copy()
        self.K.C[mu, m] += DELTA/2
        KLp = self.K*self.L
        self.K.C[mu, m] -= DELTA
        KLm = self.K*self.L
        self.K.C[mu, m] += DELTA/2
        return (KLp - KLm)/DELTA

    def right_orbital_gradient(self):
        """Rhs derivative <K|dL/dC(mu, m)>"""
        DmoKL = Dmo(self.K, self.L)
        CK = self.K.orbitals()
        KdL = nod.S*CK[0]*DmoKL[0]*self.overlap()
        
        ao, mo = nod.C.shape
        rog = full.matrix((mo, ao))
        KdL.scatteradd(rog, columns=self.L(0))
        
        return rog
#
# Class of non-orthogonal determinants
#
class nod(list):
    #
    #
    # Class global variables
    #
    S = None   #Overlap ao basis
    C = None   #VB orbital coefficients

    def __init__(self, astring, bstring, C=None, tmpdir='/tmp'):
        super(nod, self).__init__()
        #
        # Input: list of orbital indices for alpha and beta strings
        #
        self.a = astring
        self.b = bstring

        if C is None:
            self.C = nod.C
        else:
            self.C = C
        #
        # Read overlap at first invocation of constructor
        #
        aooneint = os.path.join(tmpdir, 'AOONEINT')
        if nod.S is None:
            nod.S = one.read('OVERLAP', aooneint).unpack().unblock()
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
            SKLa = CKa.T*nod.S*CLa
            Deta = SKLa.det()
            Det *= Deta
        #
        # beta
        #
        if CKb is not None:
            SKLb = CKb.T*nod.S*CLb
            Detb = SKLb.det()
            Det *= Detb
        #
        return Det


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
                D.append(full.matrix(nod.S.shape))
        else:
            SLK = CL[s].T*nod.S*CK[s]
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

class structure:

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

    def __str__(self):
        output = ["%f    %s" % (c, d) for c, d in zip(self.coef, self.nods)]
        return "\n".join(output)

class StructError(Exception):
    pass


class wavefunction:

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
        if VBSCF is True:
            self.C = structs[0].C
            self.opt = []
            nao, nmo = self.C.shape
            for i in range(nmo):
                try:
                    self.frozen.index(i)
                except ValueError:
                    self.opt.append(i)
        else:
            raise Exception("not implemented")

    def tmp(self, filename):
        return '/'.join([self.tmpdir, filename])

    def __str__(self):
        retstr = "\n"
        for i in range(len(self.structs)):
            retstr += self.coef.fmt%self.coef[i] + "   "
            retstr += "(%d)"%i
            retstr += "\n"
        return retstr

    def nel(self):
        Nel = 0
        N = 0
        for S, CS in zip(self.structs, self.coef):
            for T, CT in zip(self.structs, self.coef):
                for K, CKS in zip(S.nods, S.coef):
                    for L, CLT in zip(T.nods, T.coef):
                        D12 = DKL(K, L)
                        S12 = K*L
                        C1 = CS*CKS
                        C2 = CT*CLT
                        Na = (D12[0]&nod.S)
                        Nb = (D12[1]&nod.S)
                        Nel += (Na+Nb)*C1*C2*S12
                        N += S12*C1*C2
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
                        D12 = Dao(K, L)
                        KL = K*L
                        FKL = Fao(D12, filename=self.tmp('AOTWOINT'))
                        hKL = self.h&(D12[0]+D12[1])
                        gKL = HKL(FKL, D12)
                        H += (hKL+gKL)*KL*CKS*CLT
                SH.append(H)
        LS = len(self.structs)
        return full.init(SH).reshape((LS, LS))

    def StructureOverlap(self):
        """Calculate norm square of VB wave function"""
        OV = []
        for S in self.structs:
            for T in self.structs:
                N = 0
                for K, CKS in zip(S.nods, S.coef):
                    for L, CLT in zip(T.nods, T.coef):
                        KL = K*L
                        N += KL*CKS*CLT
                OV.append(N)
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
            nmo = 1/math.sqrt(cmo.T*nod.S*cmo)
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
                for K, CKS in zip(S.nods, S.coef):
                    for L, CLT in zip(T.nods, T.coef):
                        KL = K*L
                        N += KL*CKS*CS*CLT*CT
        return N

    def normgrad(self):
        #
        #  N = <0|0>
        # dN = 2<d0|0>
        #
        NGS = []
        r, c = nod.C.shape
        Norbgrad = full.matrix((c, r))
        #d12=full.matrix((c,c))
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
                        #
                        # Structure gradient terms
                        #
                        KL = K*L
                        #
                        # Structure gradient terms
                        #
                        GS += CKS*CT*CLT*KL
                        #
                        # Orbital gradient terms
                        #
                        CK = K.orbitals()
                        CL = L.orbitals()
                        Dmo12 = Dmo(K, L)
                        #
                        Sog12 = [None, None]
                        Sog21 = [None, None]
                        for s in range(2):
                            # D^m_\mu
                            Sog12[s] = Dmo12[s]*CL[s].T*nod.S
                        #
                        # Scatter to orbitals
                        #
                            ((CS*CT*CKS*CLT*KL)*Sog12[s]).scatteradd(
                                Norbgrad, K(s)
                                )
            NGS.append(GS)

        Nstructgrad = full.init(NGS)
        Nstructgrad *= 2
        Norbgrad *= 2
        return (Nstructgrad, Norbgrad[self.opt, :])

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
        r, c = nod.C.shape
        orbgrad = full.matrix((c, r))
        for m in range(c):
            for t in range(r):
                nod.C[t, m] += deltah
                ep = func()
                nod.C[t, m] -= delta
                em = func()
                orbgrad[m, t] = (ep - em)/delta
                nod.C[t, m] += deltah
        return (structgrad, orbgrad[self.opt])

    def numnormgrad(self, delta=1e-3):
        return self.numgrad(self.norm, delta)

    def normhess(self):
        #
        #  N = <0|0>
        # dN = 2<d0|0>
        #
        ls = len(self.structs)
        ao, mo = nod.C.shape
        Nstructhess = full.matrix((ls, ls))
        #Nstructorbhess=full.matrix((ls, mo, ao))
        Norbstructhess = full.matrix((mo, ao, ls))
        Norbhess = full.matrix((mo, ao, mo, ao))
        d12 = full.matrix((mo, mo))
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
                        # Structure hessian terms
                        #
                        S12 = det1*det2
                        Cs1 = self.coef[s1]
                        Cs2 = self.coef[s2]
                        Cd1 = str1.coef[d1]
                        Cd2 = str2.coef[d2]
                        C1 = Cs1*Cd1
                        C2 = Cs2*Cd2
                        #
                        # Structure hessian terms
                        #
                        N12 = (Cd1*Cd2)*S12
                        Nstructhess[s1, s2] += N12
                        #
                        # Orbital-structrue hessian terms
                        #
                        CK = det1.orbitals()
                        CL = det2.orbitals()
                        Dmo12 = Dmo(det1, det2)
                        Dmo21 = Dmo(det2, det1)
                        #
                        Sog12 = [None, None]
                        Sog21 = [None, None]
                        Dog12 = [full.matrix((mo, ao)), full.matrix((mo, ao))]
                        Dog21 = [full.matrix((mo, ao)), full.matrix((mo, ao))]
                        Dmm = [full.matrix((mo, mo)), full.matrix((mo, mo))]
                        Delta = [full.matrix((ao, ao)), full.matrix((ao, ao))]
                        for s in range(2):
                            #D^m_\mu
                            Sog12[s] = Dmo12[s]*CL[s].T*nod.S
                            Sog21[s] = Dmo21[s]*CK[s].T*nod.S
                            Dog12[s][det1(s), :] = Sog12[s]
                            Dog21[s][det2(s), :] = Sog21[s]
                            #Dmm[s][det1(s), det2(s)]=Dmo12[s] numpy bug
                            Dmo12[s].scatter(
                                Dmm[s], rows=det1(s), columns=det2(s)
                                )
                            Delta[s] = nod.S-nod.S*CK[s]*Dmo12[s]*CL[s].T*nod.S
                        #
                        # Scatter to orbitals
                        #
                            ((Cd1*C2*S12)*Sog12[s]).scatteradd(
                                Norbstructhess[:, :, s1], det1(s)
                                )
                            ((Cd1*C2*S12)*Sog21[s]).scatteradd(
                                Norbstructhess[:, :, s1], det2(s)
                                )
                        #
                        # Orbital-orbital hessian
                        #
                        C12 = C1*C2*S12
                        for s in range(2):
                            #print "Dog12[%d]"%s,Dog12[s]
                            for t in range(2):
                                #print "Dog21[%d]"%t,Dog21[t]
                                Norbhess += C12*Dog12[s].x(Dog12[t])
                                Norbhess += C12*Dog12[s].x(Dog21[t])
                            Norbhess -= C12*Dog12[s].x(Dog12[s]).transpose(
                                (0, 3, 2, 1)
                                )
                            Norbhess += C12*Dmm[s].x(Delta[s]).transpose(
                                (0, 3, 1, 2)
                                )
        Nstructhess *= 2
        Norbstructhess *= 2
        Norbhess *= 2
        return (
            Nstructhess,
            Norbstructhess[self.opt, :, :],
            Norbhess[self.opt, :, self.opt, :]
            )

    def vb_metric(self):
        #
        # Based on part of norm hessian <dF|dF>
        #
        ls = len(self.structs)
        ao, mo = nod.C.shape
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
                        # Structure hessian terms
                        #
                        KL = K*L
                        Nstructhess[s1, s2] += KL*CKS*CLT
                        #
                        # Orbital-structrue hessian terms
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
                            Dm_mu[s][K(s), :] = DmoKL[s]*CL[s].T*nod.S
                            Dmu_m[s][:, :] = 0
                            Dmu_m[s][:, L(s)] = nod.S*CK[s]*DmoKL[s]

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

                            Delta[s] = nod.S-nod.S*CK[s]*DmoKL[s]*CL[s].T*nod.S
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
        tmp = Norbhess[self.opt, :, :, :]
        Norbhess = tmp[:, :, self.opt, :]
        return (
            Nstructhess,
            Norbstructhess[self.opt, :, :],
            Nstructorbhess[:, self.opt, :], Norbhess
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
        ao, mo = nod.C.shape
        strstrhess = full.matrix((ls, ls))
        strorbhess = full.matrix((ls, mo, ao))
        orbstrhess = full.matrix((mo, ao, ls))
        orborbhess = full.matrix((mo, ao, mo, ao))
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
                    nod.C[mu, m] += deltah
                    epp = func()
                    nod.C[mu, m] -= delta
                    epm = func()
                    self.coef[p] -= delta
                    emm = func()
                    nod.C[mu, m] += delta
                    emp = func()
                    orbstrhess[m, mu, p] = (epp + emm - epm - emp)/delta2
                    #
                    # Reset
                    #
                    self.coef[p] += deltah
                    nod.C[mu, m] -= deltah

        #
        # Orbital-orbital
        #
        for mu in range(ao):
            for m in range(mo):
                for nu in range(ao):
                    for n in range(mo):
                        nod.C[mu, m] += deltah
                        nod.C[nu, n] += deltah
                        epp = func()
                        nod.C[nu, n] -= delta
                        epm = func()
                        nod.C[mu, m] -= delta
                        emm = func()
                        nod.C[nu, n] += delta
                        emp = func()
                        orborbhess[m, mu, n, nu] = \
                            (epp + emm - epm - emp)/delta2
                        #
                        # Reset
                        #
                        nod.C[mu, m] += deltah
                        nod.C[nu, n] -= deltah

        return (
            strstrhess,
            orbstrhess[self.opt, :, :],
            orborbhess[self.opt, :, self.opt, :]
            )

    def energy(self):
        Eone = 0
        Etwo = 0
        N = 0
        # Structures left
        for s1 in range(len(self.structs)):
            str1 = self.structs[s1]
            #Structures right
            for s2 in range(len(self.structs)):
                str2 = self.structs[s2]
                #Determinants in left structure
                for d1 in range(len(str1.nods)):
                    det1 = str1.nods[d1]
                    #Determinants in right structure
                    for d2 in range(len(str2.nods)):
                        det2 = str2.nods[d2]
                        D12 = Dao(det1, det2)
                        S12 = det1*det2
                        F12 = Fao(D12, filename=self.tmp('AOTWOINT'))
                        C1 = self.coef[s1]*str1.coef[d1]
                        C2 = self.coef[s2]*str2.coef[d2]
                        h12 = self.h&(D12[0]+D12[1])
                        g12 = HKL(F12, D12)
                        C12 = C1*C2*S12
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
        Nstructgrad = full.matrix(len(self.structs))
        c, r = nod.C.shape
        Norbgrad = full.matrix((r, c))
        N = 0
        Hstructgrad = full.matrix(len(self.structs))
        Horbgrad = full.matrix((r, c))
        H = 0
        I = full.unit(c)
        d12 = full.matrix((r, r))
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
                            Sog12[s] = Dmo12[s]*CL[s].T*nod.S
                            Hog12[s] = Dmo12[s]*CL[s].T*(
                                (self.h+F12[s]).T*(I-D12[s]*nod.S)
                                ) + Sog12[s]*H12
                        #
                        # Scatter to orbitals
                        #
                            ((C1*C2*S12)*Sog12[s]).scatteradd(Norbgrad, det1(s))
                            ((C1*C2*S12)*Hog12[s]).scatteradd(Horbgrad, det1(s))
                        #
                        #   F12[s].T*(I-D12[s]*S/S12)+H12*S/S12
                        #
        E = H/N # only electronic
        #print "energygrad:E, H, N", E, H, N
        structgrad = (2/N)*(Hstructgrad-E*Nstructgrad)
        orbgrad = (2/N)*(Horbgrad-E*Norbgrad)
        return (structgrad, orbgrad[self.opt, :])

    def numenergygrad(self, delta=1e-3):
        return self.numgrad(self.energy, delta)

    def energyhess(self):
        #
        #  N = <0|0>
        # dN = 2<d0|0>
        #
        ls = len(self.structs)
        ao, mo = nod.C.shape
        Nstructgrad = full.matrix(ls)
        Norbgrad = full.matrix((mo, ao))
        Nstructhess = full.matrix((ls, ls))
        Nstructorbhess = full.matrix((ls, mo, ao))
        Norbstructhess = full.matrix((mo, ao, ls))
        Norbhess = full.matrix((mo, ao, mo, ao))
        Hstructgrad = full.matrix(ls)
        Horbgrad = full.matrix((mo, ao))
        Hstructhess = full.matrix((ls, ls))
        Horbstructhess = full.matrix((mo, ao, ls))
        Horbhess = full.matrix((mo, ao, mo, ao))
        tmpdim = (mo, mo, ao, ao)
        d12 = full.matrix((mo, mo))
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
        S = nod.S
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
                            Norbgrad[det1(s), :] += (C1*C2*S12)*dm_a[s]
                            Horbgrad[det1(s), :] += (C1*C2*S12)*Hog12[s]
                        #
                        # Structure hessian terms
                        #
                        N12 = (Cd1*Cd2)*S12
                        Nstructhess[s1, s2] += N12
                        Hstructhess[s1, s2] += N12*H12
                        #
                        # Orbital-structrue hessian terms
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
                            Norbstructhess[det1(s), :, s1] += Nd12*dm_a[s]
                            Norbstructhess[det2(s), :, s1] += Nd12*d_am[s].T
                            Horbstructhess[det1(s), :, s1] += Nd12*Hog12[s]
                            Horbstructhess[det2(s), :, s1] += Nd12*Hog21[s]

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
            Eorbstructhess[self.opt, :, :],
            Eorbhess[self.opt, :, self.opt, :]
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

if __name__ == "__main__":
    pass
