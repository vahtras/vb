import os
from util import full
from numpy.linalg import LinAlgError


class Nod(object):
    """
    # Class of non-orthogonal determinants
    """
    #
    # Class global variables
    #
    S = None   #Overlap ao basis
    C = None   #VB orbital coefficients

    def __init__(self, astring, bstring, C=None, S=None):
        #
        # Input: list of orbital indices for alpha and beta strings
        #
        self.a = astring
        self.b = bstring

        if C is not None:
            self.C = C

        if S is not None:
            Nod.S = S


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
            raise TypeError("Non-binary input to Nod object")

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
            try:
                D.append(CK[s]*(CL[s].T/SLK))
            except LinAlgError:
                raise SingularOverlapError

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

class SingularOverlapError(Exception):
    pass
