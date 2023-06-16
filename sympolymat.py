from __future__ import annotations
from copy import deepcopy
import pickle
import json
import numpy as np
from .ndpoly import NdPoly
from .jsonutils import _str2tuple

class SymPolyMat():
    """ Real symmetric matrix of multivariate polynomial functions """
    def __init__(self, Ns, Nd):
        self._Ns = Ns
        self._Nd = Nd
        self._polys = [ NdPoly.empty(Nd) for i in range(Ns) for j in range(i+1) ]

    @classmethod
    def copy(cls, other):
        """ Copy constructor """
        if not isinstance(other, cls):
            raise TypeError(f"Copy constructor expecting type {cls}.")
        return deepcopy(other)

    @property
    def Ns(self):
        return self._Ns

    @property
    def Nd(self):
        return self._Nd

    def get_all_x0(self):
        all_x0 = {(i,j): self[i,j].x0 for i in range(self._Ns) for j in range(i+1)}
        return all_x0

    def set_common_x0(self, shift):
        assert len(np.array(shift).flatten()) == self._Nd, "Wrong x0 size."
        for p in self._polys:
            p.x0 = shift

    def set_x0_by_ij(self, x0_by_ij):
        """
        For each specified polynomial in (i,j), set its origin.
        Parameters:
        * x0_by_ij : dict (i,j) -> x0, with
            - (i,j): tuple of integers, the matrix element indices.
            - x0: 1D list or np.array of length Nd.
        """
        for ij, x0 in x0_by_ij.items():
            self[ij].x0 = x0

    def coeffs_and_keys(self):
        """
        Return a 1D array of all the matrix's coefficients.
        Additionally, return a list of tuples of the corresponding matrix element indices
        and monomial powers. The latter is of the same length as that of the array of
        coefficients and can be used for reconstruction of the matrix.
        Return:
        * coeffs: 1d array of the coefficients of all the monomials in the matrix.
        * list of tuples (i,j,(p1, p2, ...)) where:
            - i and j are the indices of a matrix element
            - (p1, p2, ...) is a tuple of powers of a monomial in matrix element (i,j)
        """
        keys = []
        coeffs = []
        for i in range(self._Ns):
            for j in range(i+1):
                coeffs.append(self[i,j].coeffs_to_array())
                keys += [(i,j,powers) for powers in self[i,j].powers_to_list()]
        return np.hstack(tuple(coeffs)), keys

    @classmethod
    def construct(cls, Ns, Nd, keys, coeffs, dict_x0={}):
        """ Reconstruct matrix from flat list of coefficients
        Parameters:
        * Ns : integer, number of states
        * Nd : integer, number of dimensions/coordinates
        * keys: list of (i,j,powers) with
            - i,j integer matrix element indices
            - powers = (p1, p2, ...) tuple of integer partial powers of monomial
        * coeffs: 1D np.array of coefficients [c1,c2,...] of same length of keys
        * x0 = dict of (i,j) -> x0 with
            - i,j matrix element indices and
            - x0 the corresponding origin (list/array of length Nd)
        Return:
        * Diabatic matrix W such that
          W[i,j][powers] = c_{powers}^(i,j)
          W[i,j].x0 = x0^(i,j)
        """
        W = cls(Ns, Nd)

        # Coefficients
        for n, key in enumerate(keys):
            i, j, powers = key
            W[i,j][powers] = coeffs[n]

        # Shifts
        W.set_x0_by_ij(dict_x0)

        return W
    
    def __iter__(self):
        return self._polys.__iter__()

    def __getitem__(self, pos) -> NdPoly:
        i, j = pos
        if j > i :
            i, j = j, i
        return self._polys[i*(i+1)//2 + j]

    def __setitem__(self, pos, value: NdPoly):
        i, j = pos
        if j > i :
            i, j = j, i
        self._polys[i*(i+1)//2 + j] = value

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """ Returns a len(x)*Ns*Ns ndarray of values at x. """
        x = np.atleast_2d(x)
        if self._Nd == 1 and x.shape[0] == 1:   # If x is a row, make it a column
            x = x.T
        W = np.zeros((x.shape[0], self._Ns, self._Ns))
        for i in range(self._Ns):   # Compute lower triangular part
            for j in range(i+1):
                W[:,i,j] = self[i,j](x)
                W[:,j,i] = W[:,i,j]
        return W

    def __repr__(self):
        s = object.__repr__(self) + "(Ns={self._Ns},Nd={self._Nd})\n"
        for i in range(self._Ns):
            for j in range(i+1):
                s += f"({i},{j}): {self[i,j].__repr__()}" + "\n"
        return s
   
    def __str__(self): 
        s = ""
        for i in range(self._Ns):
            for j in range(i+1):
                s += f"Matrix element ({i},{j})" + "\n"
                s += f"ORIGIN: {self[i,j].x0}" + "\n"
                for powers, coeff in self[i,j].items():
                    s += f"{powers}: {coeff}" + "\n"
                s += "\n"
        return s

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if (self.Nd == other.Nd
                    and self.Ns == other.Ns
                    and all([a == b for a,b in zip(self, other)])):
                return True
        return False

    def write_to_txt(self, filename):
        with open(filename, "w") as fout:
            fout.write(self.__str__())

    @staticmethod
    def read_from_txt(filename):
        # with open(filename, "r") as fin:
        # return W
        NotImplementedError(
            "Reading from text file is not possible yet. " \
            + "Please load from binary file using read_from_file().")

    def write_to_file(self, filename):
        with open(filename, "wb") as fout:
            pickle.dump(self, fout)

    @classmethod
    def read_from_file(cls, filename):
        with open(filename, "rb") as fin:
            W = pickle.load(fin)
        if not isinstance(W, cls):
            raise TypeError(f"File contains object of type {W.type}, expected {cls}.")
        return W

    @classmethod
    def zero(cls, Ns, Nd):
        """ Create zero matrix.
        Each matrix term is a constant monomial with zero coefficient.
        """
        M = cls(Ns,Nd)
        for i in range(Ns):
            for j in range(Ns):
                M[i,j] = NdPoly.zero(Nd)
        return M

    @classmethod
    def zero_like(cls, other):
        """ Create copy with zero polynomial coefficients.
        Each matrix element is a polynomial with all powers as in other,
        but whose coefficients are all zero.
        """
        newmat = cls.copy(other)
        for wij in newmat:
            for powers in wij:
                wij[powers] = 0
        return newmat

    @classmethod
    def eye(cls, Ns, Nd):
        """ Create identity matrix
        Each matrix term is a constant monomial with coefficient 1 along
        the diagonal and 0 otherwise.
        """
        I = cls.zero(Ns, Nd)
        for i in range(Ns):
            I[i,i] = NdPoly({tuple([0 for _ in range(Nd)]): 1})
        return I

    @classmethod
    def eye_like(cls, other):
        """ Create identity matrix, with polynomial coefficients of other.
        Each matrix element is a polynomial with all powers as in other,
        but whose coefficients are all zero except along the diagonal where
        the constant term .
        Warning: If there is no constant term in a diagonal element of other,
        this will be added (with a coefficient 1).
        """
        newmat = cls.zero_like(other)
        for i in range(newmat.Ns):
            for j in range(i+1):
                if i == j :
                    newmat[i,j][newmat[i,j].zeroPower] = 1
        return newmat

    def save_to_JSON(self, fname) -> None:
        with open(fname, "w") as f:
            json.dump(f, self.to_JSON_dict())

    def to_JSON_dict(self) -> dict:
        dct = {
                "__SymPolyMat__" : True,
                "Nd" : self.Nd,
                "Ns" : self.Ns,
                "elements" : {f"({i}, {j})": self[i,j].to_JSON_dict() \
                        for i in range(self.Ns) for j in range(i+1)}
                }
        return dct
        
    @classmethod
    def load_from_JSON(cls, fname) -> cls:
        with open(fname, "r") as f:
            M = cls.from_JSON_dict(json.load(f))
        return M

    @staticmethod
    def from_JSON_dict(dct):
        if "__SymPolyMat__" not in dct:
            raise KeyError("The JSON object is not a SymPolyMat.")

        M = SymPolyMat(dct["Ns"],dct["Nd"])
        for ij, poly in dct["elements"].items():
            M[_str2tuple(ij)] = NdPoly.from_JSON_dict(poly)

        return M
