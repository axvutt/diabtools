from copy import deepcopy
import pickle
import numpy as np
from .sympolymat import SymPolyMat
from .dampingfunction import DampingFunction

class DampedSymPolyMat(SymPolyMat):
    """ Symmetric Matrix of Polynomials, with damping functions """
    def __init__(self, Ns, Nd):
        super().__init__(Ns, Nd)
        self._damp = [One() for i in range(1,Ns) for j in range(i)]

    @classmethod
    def from_SymPolyMat(cls, other: SymPolyMat):
        """
        Construct DampedSymPolyMat from preexisting SymPolyMat.
        Copy the SymPolyMat members and set no damping of off-diagonal
        elements.
        
        NB: Since no true copy constructor exists in Python, we'll do the dirty
        trick of copying all private attributes, manually.
        """
        DW = DampedSymPolyMat(other.Ns, other.Nd)
        DW._polys = deepcopy(other._polys)
        return DW

    def _check_indices(self, i, j):
        if i == j:
            raise IndexError("Diagonal elements not allowed")

    def set_damping(self, pos, dfun: DampingFunction):
        self._check_indices(*pos)
        i, j = pos
        if j > i :
            i, j = j, i
        self._damp[i*(i-1)//2 + j] = dfun

    def get_damping(self, pos) -> DampingFunction:
        self._check_indices(*pos)
        i, j = pos
        if j > i :
            i, j = j, i
        return self._damp[i*(i-1)//2 + j]

    def __call__(self, x):
        x = np.atleast_2d(x)
        if self._Nd == 1 and x.shape[0] == 1:   # If x is a row, make it a column
            x = x.T
        Wx = super().__call__(x)
        Dx = np.ones(Wx.shape)
        for i in range(1, self._Ns):   # Compute off-diag lower triangular part
            for j in range(i):
                Dx[:,i,j] = self.get_damping((i,j))(x)
                Dx[:,j,i] = Dx[:,i,j]
        return Wx * Dx

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if super().__eq__(other):
                if self._damp == other._damp:
                    return True
        return False

    @staticmethod
    def read_from_file(fin):
        W = pickle.load(fin)
        return W

    @staticmethod
    def zero(Ns, Nd):
        """ Create zero matrix.
        Each matrix term is a constant monomial with zero coefficient.
        No damping (matrix of ones).
        """
        M = DampedSymPolyMat(Ns,Nd)
        for i in range(Ns):
            for j in range(Ns):
                M[i,j] = NdPoly.zero(Nd)
        return M

    @staticmethod
    def zero_like(other: DampedSymPolyMat):
        """ Create copy with zero polynomial coefficients.
        Each matrix element is a polynomial with all powers as in other,
        but whose coefficients are all zero.
        No damping (matrix of ones).
        """
        newmat = deepcopy(other)
        for wij in newmat:
            for powers in wij:
                wij[powers] = 0
        return newmat

    @staticmethod
    def eye(Ns, Nd):
        """ Create identity matrix
        Each matrix term is a constant monomial with coefficient 1 along
        the diagonal and 0 otherwise.
        No damping (matrix of ones).
        """
        I = DampedSymPolyMat.zero(Ns, Nd)
        for i in range(Ns):
            I[i,i] = NdPoly({tuple([0 for _ in range(Nd)]): 1})
        return I

    @staticmethod
    def eye_like(other: DampedSymPolyMat):
        """ Create identity matrix, with polynomial coefficients of other.
        Each matrix element is a polynomial with all powers as in other,
        but whose coefficients are all zero except along the diagonal where
        the constant term .
        Warning: If there is no constant term in a diagonal element of other,
        this will be added (with a coefficient 1).
        No damping (matrix of ones).
        """
        newmat = DampedSymPolyMat.zero_like(other)
        for i in range(newmat.Ns):
            for j in range(i+1):
                if i == j :
                    newmat[i,j][newmat[i,j].zeroPower] = 1
        return newmat

    def to_JSON_dict(self) -> dict:
        dct = super().to_JSON_dict()
        dct.update({
                "__DampedSymPolyMat__" : True,
                "damping" : {f"({i}, {j})": self.get_damping((i,j)).to_JSON_dict() \
                        for i in range(1, self._Ns) for j in range(i)}
                })
        return dct
        
    @staticmethod
    def from_JSON_dict(dct) -> cls:
        if "__DampedSymPolyMat__" not in dct:
            raise KeyError("The JSON object is not a DampedSymPolyMat.")

        M = SymPolyMat.from_JSON_dict(dct)
        M = DampedSymPolyMat.from_SymPolyMat(M)
        for ij, dfdct in dct["damping"].items():
            if "__One__" in dfdct:
                df = One.from_JSON_dict(dfdct)
            if "__Gaussian__" in dfdct:
                df = Gaussian.from_JSON_dict(dfdct)
            elif "__Lorentzian__" in dfdct:
                df = Lorentzian.from_JSON_dict(dfdct)
            else:
                df = None
                Warning("Unknown damping function, setting to None")
            M.set_damping(_str2tuple(ij), df)

        return M
