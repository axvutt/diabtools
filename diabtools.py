from __future__ import annotations
import sys
from copy import *
from typing import List, Tuple, Dict, Union
from collections import UserDict
from dataclasses import dataclass
from abc import ABC, abstractmethod 
import math
import pickle
import json
import numpy as np
import scipy
import pytest

class NdPoly(UserDict):
    """ Multivariate polynomial in Nd dimensions of arbitrary order.
    
    Dictionnary data structure whose:
        - keys are tuples of length Nd with non-negative integer entries.
            Each tuple is associated to a monomial. The k-th partial order is the k-th
            entry in the tuple (so the total monomial order is the sum of the entries).
            e.g.: (2,0,1) represents x**2 * z (in 3D)
                  (1,1,2,5) represents x * y * z**2 * t**5
        - values are the coefficients multiplied by the associated monomial.

    """

    __slots__ = ('_Nd', '_degree', '_x0', '_zeroPower')

    def __init__(self, data):
        if len(data) == 0:
            raise(ValueError("Cannot make empty polynomial with unknown " \
                    + "dimensionality. Use NdPoly.empty() instead."))

        # Aliases for keys and values
        self.powers = self.keys
        self.coeffs = self.values

        self._Nd = None
        self._x0 = None
        super().__init__(data)  # Sets Nd and degree
        self._zeroPower = tuple([0 for _ in range(self._Nd)])
        self._x0 = np.array([0 for _ in range(self._Nd)])

    @property
    def Nd(self):
        """ Number of variables of the polynomial """
        return self._Nd

    @property
    def degree(self):
        """ Return the degree of the polynomial. """
        max_degree = -1
        for p in self:
            if self[p] != 0:
                max_degree = max(max_degree, sum(p))
        return max_degree 

    @property
    def def_degree(self):
        """ Return the degree of the polynomial including all defined terms, even zero. """
        max_degree = -1
        for p in self:
            max_degree = max(max_degree, sum(p))
        return max_degree 

    @property
    def zeroPower(self):
        """ Return the tuple of powers of constant term, i.e. (0, ..., 0) """
        return self._zeroPower

    @property
    def x0(self):
        return self._x0

    @x0.setter
    def x0(self, x0):
        """ Transform the polynomial P(x) into Q(x) = P(x-x0)
        by setting the origin point without changing coefficients. """
        if self._Nd is None:
            raise(RuntimeError("Cannot set origin in a polynomial of unknown dimensionality."))
        x0 = np.array(x0).flatten()
        if x0.size != self._Nd:
            raise(ValueError(f"{x0} should be a point in {self._Nd} dimensions."))
        self._x0 = x0

    def max_partial_degrees(self):
        return [max([powers[i] for powers in self.powers()]) \
                for i in range(self._Nd)]

    def expanded(self):
        """ Return polynomial with (x-x0)^n terms expanded

        This has the effect of adding more monomials to the new NdPoly object.
        More specifically, if (n_0,...,n_{Nd-1}) are the maximum partial degrees
        in the undisplaced polynomial, the displaced one will contain all
        monomials with partial degrees n'_i <= n_i for i in {0, Nd-1}.

        x0
        list, tuple or 1d np.ndarray of length Nd of the coordinates of the
        shift vector

        return
        the shifted polynomial
        """
        if all(self._x0 == 0):
            return self

        max_degs = self.max_partial_degrees()
        n_monomials = math.prod([n+1 for n in max_degs])

        P_expanded = NdPoly.empty(self._Nd)
        for i in range(n_monomials):
            # Convert from flat index to multi-index (= order of diff)
            order = []
            i_left = i
            for j in range(self._Nd):
                if max_degs[j] == 0:
                    order.append(0)
                    continue
                i_rem = i_left % (max_degs[j]+1)    
                i_left -= i_left // (max_degs[j]+1)
                order.append(i_rem) 
            order = tuple(order)

            # Coefficient form multi-dimensional Taylor shift
            coeff = self.derivative(order)(-self._x0) \
                    / math.prod([math.factorial(order[d]) for d in range(self._Nd)])
            
            # Set monomial
            P_expanded[order] = coeff
        return P_expanded

    def setZeroConst(self):
        """ Set constant term to zero.

        This has the effect of adding self._zeroPower to the dictionnary
        keys if it was not previously there. Equivalent to self += 0
        """
        self[self._zeroPower] = 0

    def grow_degree(self, degree: int, fill = 0, max_pdeg = None):
        """ Add all the terms to the polynomial up to a given degree,
        without changing already defined terms.
        max_pdeg:
        tuple or list of the maximum partial degrees that can be reached by
        the corresponding coordinate. Elements in this tuple can be None and
        thus the partial degree may go up to degree.
        fill:
        float value of the coefficients to be added.
        """
        for n in range(degree + 1):
            powers = NdPoly._get_tuples(self._Nd, n)
            for p in powers:
                if p in self:   # If term already set, do not touch it
                    continue
                max_pdeg_reached = False
                if max_pdeg is not None:
                    for i in range(self._Nd):
                        if max_pdeg[i] is not None:
                            if p[i] > max_pdeg[i]:
                                max_pdeg_reached = True
                                break
                if not max_pdeg_reached:
                    self[p] = fill

    def derivative(self, orders: tuple):
        """ Partial derivative of the polynomial
        
        order
        tuple whose items are integers indicating the order
        of differentiation in the corresponding coordinate.
        e.g. for Nd = 3, order = (1,0,2) yields
        d^3 P / dx dz^2

        returns
        NdPoly corresponding to the partial derivative.
        For each power 'p' in the original polynomial, the result
        polynomial contains the powers 'p - order'.
        If for any 0 <= i < Nd, p[i]-order[i] < 0, the key power is
        removed. If no monomial survives to differentiation,
        the returned polynomial is constant with zero coefficient
        (NdPoly.zero(Nd)).
        """
        if len(orders) != self._Nd:
            raise(ValueError("Invalid differentiation order of length "\
                    + "{len(order)}, expecting {self._Nd}"))

        D = NdPoly.empty(self._Nd)
        for powers, coeff in self.items():  # Differentiate each monomial
            new_powers = list(powers)
            for dof in range(self._Nd):
                new_powers[dof] -= orders[dof]
            # If monomial is annihilated, go to next
            if any([p < 0 for p in new_powers]): 
                continue

            # The monomial survived, compute corresponding coefficient
            new_coeff = coeff
            for dof in range(self._Nd):
                for k in range(new_powers[dof]+1, new_powers[dof] + orders[dof] + 1):
                    new_coeff *= k
            D[tuple(new_powers)] = new_coeff

        if len(D.powers()) == 0:
            return NdPoly.zero(self._Nd)
        else:
            return D

    def remove(self, key):
        del self[key]

    def __setitem__(self, powers, coeff):
        if self._Nd is None and len(powers) != 0 :
            self._Nd = len(powers)
        else:
            assert len(powers) == self._Nd, f"Inappropriate powers {powers}. Expected {self._Nd} integers."
        super().__setitem__(powers, coeff)
        self.data = dict(sorted(self.data.items())) # Rough sorting, may need improvement

    def __call__(self, x: np.ndarray):
        """ Calculate value of the polynomial at x.
        
        Parameters:
        x
        N * Nd ndarray containing the values of the coordinates where to evaluate the polynomial.

        Returns:
        P
        np.ndarray containing the N values of the polynomial over the given x meshgrids.
        """
        x = np.atleast_2d(x)
        if self._Nd == 1 and x.shape[0] == 1:   # If x is a row, make it a column
            x = x.T
        assert x.shape[1] == self._Nd, "Wrong point dimensionality "\
                + f"{x.shape[1]}, should be {self._Nd}."
        P = np.zeros((x.shape[0],))
        for powers, coeff in self.items():
            monomial = np.ones((x.shape[0],))
            for k in range(len(powers)):
                monomial *= (x[:,k]-self._x0[k])**powers[k]
            P += coeff * monomial
        if P.size == 1:
            P = P.item()
        return P 
    
    def __repr__(self):
        s = object.__repr__(self) \
                + f"(Nd={self._Nd},x0={self._x0},deg={self.degree},ddeg={self.def_degree})\n" \
                + super().__repr__()
        return s

    def __str__(self):
        """ Explicit representation of the polynomial in terms of its variables """

        if self._Nd is None:
            return f"<Empty {self.__class__} with no dimension at {hex(id(self))}.>"

        if self == {}:
            return f"<Empty {self.__class__} of dimension {self._Nd} at {hex(id(self))}.>"

        s = ""

        first = True
        for powers, coeff in self.items():
            if coeff == 0:  # Ignore terms with 0 coeff
                continue

            cstr = ""
            # Write only non "1" coefficients except constant term
            if coeff != 1 or powers == self.zeroPower :  
                cstr = f"{coeff} "

            # + sign and coefficients
            if first :      # No + sign for the first term
                s += cstr
                first = False
            else:
                s += f" + " + cstr

            for d, p in enumerate(powers):
                if p == 0 : # Do not show variables to the 0th power
                    continue
                if p == 1 : # Do not show ^1
                    s += f"x{d}"
                else :
                    s += f"x{d}^{p}"

        # If zero polynomial, show 0 instead of empty string
        if len(s) == 0 :
            s = "0"

        # Show coordinate shift n polynomial if non zero
        for d in range(self._Nd):
            if self._x0[d] != 0:
                s = s.replace(f"x{d}", f"(x{d}-X{d})")

        # Show value of origin shift if not zero
        if any(self._x0 != 0):
            s += f"   |   [X0"
            for d in range(1,self._Nd):
                s += f" X{d}"
            s += f"] = {self._x0}"

        if self._Nd < 8:    # Use x,y,z,... instead of x1, x2, x3 if Nd < 8
            xyz = "xyztuvw"
            for d in range(self._Nd):
                s = s.replace(f"x{d}", xyz[d])
                s = s.replace(f"X{d}", xyz.upper()[d])
        
        return s

    def __add__(self, other: Union[int,float, self.__class__]):
        result = deepcopy(self)
        if isinstance(other, self.__class__):
            assert np.all(self._x0 == other.x0), "Origins of added polynomials do not match."
            for powers, coeff in other.items():
                if powers in result:
                    result[powers] += coeff
                else:
                    result[powers] = coeff
            return result
        elif isinstance(other, (float, int)):
            # Convert number to polynomial, then call __add__ recursively
            other = NdPoly({self.zeroPower: other})
            return self + other
        else:
            raise(TypeError)

    def __mul__(self, other: Union[int, float, self.__class__]):
        result = deepcopy(self)
        if isinstance(other, self.__class__):
            # assert np.all(self._x0 == other.x0), "Origins of added polynomials do not match."
            raise(NotImplementedError("Product between NdPoly's not yet implemented"))
        elif isinstance(other, (float, int)):
            for powers in self.powers():
                self[powers] *= other
            return self
        else:
            raise(TypeError)

    def coeffs_to_array(self):
        """ Return monomial coefficients as a 1D np.ndarray."""
        return np.array(list(self.coeffs()))

    def powers_to_list(self):
        """ Return list of tuples of powers in each monomial."""
        return list(self.powers())

    @classmethod
    def empty(cls, Nd):
        """ Return an empty Nd-dimensional polynomial """
        # Dirty way: create a contant zero poly and remove the dict item
        powers = tuple([0 for _ in range(Nd)])
        P = cls.zero(Nd) 
        del P[powers]
        return P

    @classmethod
    def zero(cls, Nd):
        """ Return a Nd-dimensional polynomial with zero constant term only """
        powers = tuple([0 for _ in range(Nd)])
        return cls({powers: 0})

    @classmethod
    def one(cls, Nd):
        """ Return a Nd-dimensional polynomial with unit constant term only """
        powers = tuple([0 for _ in range(Nd)])
        return cls({powers: 1})

    @classmethod
    def zero_like(cls, P):
        """ Return a polynomial with all powers in P but with zero coefficients """
        return cls({ p : 0 for p in list(P.powers()) })

    @classmethod
    def one_like(cls, P):
        """ Return a polynomial with all powers in P, with unit constant term
        and all the others with zero coefficient """
        return cls({ p : 0 for p in list(P.powers()) }) + 1

    @staticmethod
    def _get_tuples(length, total):
        """ Generator which essentially constructs all the
        tuples of a given length and sum of its integer entries.

        Taken from https://stackoverflow.com/questions/29170953
        
        Idea:
        Make 1D tuples from total to zero
        Prepend index such that the sum of the 2D tuple is total
        Make 1D tuples from (total-1) to zero
        Prepend index such that the sum of the 2D tuple is (total-1)
        ................... (total-2) to zero
        .................................................. (total-2)
        For each generated 2D tuple:
        Prepend index such that the sum of the 3D tuple is total
        [... continue if 4D ...]
        Prepend index such that the sum of the 3D tuple is (total-1)
        etc...
        """
        if length == 1:
            yield (total,)
            return

        for i in range(total + 1):
            for t in NdPoly._get_tuples(length - 1, total - i):
                yield (i,) + t

    @classmethod
    def fill_maxdegree(cls, Nd, degree, fill, max_pdeg = None):
        """ Return a polynomial with all powers up to a given degree """
        P = cls.empty(Nd)
        P.grow_degree(degree, fill, max_pdeg)
        return P

    @classmethod
    def zero_maxdegree(cls, Nd, degree, max_pdeg = None):
        """ Return a polynomial with all powers up to a given degree """
        return cls.fill_maxdegree(Nd, degree, 0, max_pdeg)

    @classmethod
    def one_maxdegree(cls, Nd, degree):
        return cls.zero_maxdegree(Nd, degree) + 1

    def to_JSON_dict(self) -> dict:
        jsondct = {
                "__NdPoly__" : True,
                "Nd" : self.Nd,
                "x0" : self.x0.tolist(),
                "coeffs_by_powers" : {str(p) : c for p,c in self.items()},
                }
        return jsondct
        
    @staticmethod
    def from_JSON_dict(dct) -> NdPoly:
        if "__NdPoly__" not in dct:
            raise KeyError("The JSON object is not a NdPoly.")

        P = NdPoly.empty(dct["Nd"])
        P.x0 = dct["x0"]
        
        # Get polynomial coefficients
        for raw_power, coeff in dct["coeffs_by_powers"].items():
            P[_str2tuple(raw_power)] = coeff

        return P

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


class DampingFunction(ABC):
    """ Abstract base class for damping functions
    Subclasses should implement __call__"""
    def __init__(self, x0):
        self._x0 = x0

    @property
    def x0(self):
        return self._x0

    @x0.setter
    def x0(self, x0):
        self._x0 = x0

    @abstractmethod
    def __call__(self, x):
        pass

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if self._x0 == other._x0:
                return True
        return False

    def to_JSON_dict(self):
        return {"__DampingFunction__": True, "x0": self._x0}

    @staticmethod
    def from_JSON_dict(dct):
        return None


class One(DampingFunction):
    def __init__(self):
        super().__init__(0)

    def __call__(self, x):
        return 1

    def to_JSON_dict(self):
        dct = super().to_JSON_dict()
        dct.update({"__One__": True})
        return dct

    @staticmethod
    def from_JSON_dict(dct):
        if "__One__" not in dct:
            raise KeyError("The JSON object is not a One.")

        return One()


class Gaussian(DampingFunction):
    def __init__(self, x0, sigma):
        super().__init__(x0)
        if math.isclose(sigma,0):
            raise ValueError("Zero std deviation.")
        self._sigma = sigma

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, sigma):
        self._sigma = sigma

    def __call__(self,x):
        return np.exp(-0.5*((x-self.x0)/self.sigma)**2)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if super().__eq__(other):
                if self.sigma == other.sigma:
                    return True
        return False

    def to_JSON_dict(self):
        dct = super().to_JSON_dict()
        dct.update({"__Gaussian__": True, "sigma": self._sigma})
        return dct

    @staticmethod
    def from_JSON_dict(dct):
        if "__Gaussian__" not in dct:
            raise KeyError("The JSON object is not a Gaussian.")

        return Gaussian(dct["x0"], dct["sigma"])


class Lorentzian(DampingFunction):
    def __init__(self, x0, gamma):
        super().__init__(x0)
        if math.isclose(gamma,0):
            raise ValueError("Zero gamma width parameter.")
        self._gamma = gamma

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        self._gamma = gamma

    def __call__(self,x):
        return 1/(1 + ((x-self.x0)/self.gamma)**2)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if super().__eq__(other):
                if self.gamma == other.gamma:
                    return True
        return False

    def to_JSON_dict(self):
        dct = super().to_JSON_dict()
        dct.update({"__Lorentzian__": True, "gamma": self._gamma})
        return dct

    @staticmethod
    def from_JSON_dict(dct):
        if "__Lorentzian__" not in dct:
            raise KeyError("The JSON object is not a Lorentzian.")

        return Lorentzian(dct["x0"], dct["gamma"])

@dataclass
class Results:
    success : bool = False
    coeffs : np.ndarray = np.array([])
    n_it : int = 0
    n_fev : int = 0
    n_jev : int = 0
    n_hev : int = 0
    rmse : float = 0.0
    wrmse : float = 0.0
    mae : float = 0.0
    wmae : float = 0.0
    delta_rmse : float = 0.0
    delta_wrmse : float = 0.0
    delta_mae : float = 0.0
    delta_wmae : float = 0.0
    residuals : np.ndarray = np.array([])
    cost : float = 0.0
    delta_cost : float = 0.0

    def reset(self):
        self.success = False
        self.coeffs = np.array([])
        self.n_it = 0
        self.rmse = 0.0
        self.wrmse = 0.0
        self.mae = 0.0
        self.wmae = 0.0
        self.delta_rmse = 0.0
        self.delta_wrmse = 0.0
        self.delta_mae = 0.0
        self.delta_wmae = 0.0
        self.residual = np.array([])
        self.cost = 0.0
        self.delta_cost = 0.0

    def from_OptimizeResult(self, optres : scipy.optimize.OptimizeResult):
        self.success = optres.success
        self.coeffs = optres.x
        self.cost = optres.fun
        self.n_it = optres.nit
        self.n_fev = optres.nfev
        self.n_jev = optres.njev
        self.success = optres.success
    
    def to_JSON_dict(self):
        out = {
                "__Results__" : True,
                "success"     : self.success,
                "coeffs"      : self.coeffs.tolist(),
                "n_it"        : self.n_it,
                "n_fev"       : self.n_fev,
                "n_jev"       : self.n_jev,
                "n_hev"       : self.n_hev,
                "rmse"        : self.rmse,
                "wrmse"       : self.wrmse,
                "mae"         : self.mae,
                "wmae"        : self.wmae,
                "delta_rmse"  : self.delta_rmse,
                "delta_wrmse" : self.delta_wrmse,
                "delta_mae"   : self.delta_mae,
                "delta_wmae"  : self.delta_wmae,
                "residuals"   : self.residuals.tolist(),
                "cost"        : self.cost,
                "delta_cost"  : self.delta_cost,
                }
        return out

    @staticmethod
    def from_JSON_dict(dct):
        if "__Results__" not in dct:
            raise KeyError("The JSON object is not a Results object.")

        return Results(
                success     = dct["success"    ],
                coeffs      = np.array(dct["coeffs"     ]),
                n_it        = dct["n_it"       ],
                n_fev       = dct["n_fev"      ],
                n_jev       = dct["n_jev"      ],
                n_hev       = dct["n_hev"      ],
                rmse        = dct["rmse"       ],
                wrmse       = dct["wrmse"      ],
                mae         = dct["mae"        ],
                wmae        = dct["wmae"       ],
                delta_rmse  = dct["delta_rmse" ],
                delta_wrmse = dct["delta_wrmse"],
                delta_mae   = dct["delta_mae"  ],
                delta_wmae  = dct["delta_wmae" ],
                residuals   = np.array(dct["residuals"  ]),
                cost        = dct["cost"       ],
                delta_cost  = dct["delta_cost" ],
                )

class Diabatizer:
    def __init__(self, Ns, Nd, Nm, diab_guess: List[SymPolyMat] = None):
        self._Nd = Nd
        self._Ns = Ns
        self._Nm = Nm
        if diab_guess is not None:
            if len(diab_guess) != Nm:
                raise ValueError("Inconsistent number of matrices {} vs number of given guesses {}.".format(Nm, len(diab_guess)))
            self._Wguess = diab_guess
        else :
            self._Wguess = [SymPolyMat.eye(Ns, Nd) for _ in range(Nm)]
        self._Wout = self._Wguess
        self._x = dict()
        self._energies = dict()
        self._states_by_domain = [dict() for _ in range(Nm)]
        self._domain_IDs = set()
        self._Ndomains = 0
        self._last_domain_ID = 0
        self._auto_fit = True
        self._wfun_coord = None
        self._wfun_energy = None
        self._manually_weighted_domains = set()
        self._weights = dict()
        self._weights_coord = dict()
        self._weights_energy = dict()
        self._print_every = 50
        self._n_cost_calls = 0
        self._last_residuals = np.array([])
        self._results = [Results() for _ in range(Nm)]

    @property
    def x(self):
        """
        Map from domain id to array of coordinates of the domain.
        """
        return self._x

    @property
    def energies(self):
        """
        Map from domain id to n*m array of energies,
        n being the number of points in the domain and
        m being the number of states considered in that domain.
        """
        return self._energies

    @property
    def states_by_domain(self):
        """
        Returns list of dicts domain_id -> (s1, s2, ...) with
        * domain_id : integer, a domain identification number
        * (s1, s2, ...) : tuple of integers, indicating state numbers
        Each entry in the list referes to a diabatic matrix included in the model.
        """
        return self._states_by_domain

    @property
    def Nd(self):
        """ Get number of dimensions """
        return self._Nd

    @property
    def Ns(self):
        """ Get number of states """
        return self._Ns

    def n_fitted_points(self, i_matrix):
        Npts = 0
        for dom_id, states in self._states_by_domain[i_matrix].items():
            Npts += len(self._x[dom_id]) * len(states)
        return Npts

    @property
    def Wguess(self):
        return self._Wguess

    @Wguess.setter
    def Wguess(self, guess):
        self._Wguess = guess

    @property
    def Wout(self):
        return self._Wout

    @property
    def results(self):
        """ Return diabatization results dictionnary """
        return self._results

    def add_domain(self, x: np.ndarray, en: np.ndarray):
        """ Add N points to the database of energies to be fitted
        
        Parameters
        x
        np.ndarray of size Nd*N containing coordinates of the points to add

        en
        np.ndarray of Ns*N energy values at the corresponding coordinates

        Return
        n_chunk
        integer id of the data chunk added to the database
        """
        x = np.atleast_2d(x)
        en = np.atleast_2d(en)
        if self._Nd == 1 and x.shape[0] == 1:
            x = x.T
        if self._Ns == 1 and en.shape[0] == 1:
            en = en.T
        assert x.shape[1] == self._Nd, "Wrong coord. array dimensions, " \
                + "{x.shape[1]} vars, expected {self._Nd}."
        assert en.shape[1] == self._Ns, "Wrong energy array dimensions " \
                + f"{en.shape[1]} states, expected {self._Ns}."
        assert x.shape[0] == en.shape[0], "Coordinates vs energies "\
                + "dimensions mismatch."
        id_domain = self._last_domain_ID
        self._domain_IDs.add(id_domain)
        self._x[id_domain] = x
        self._weights_coord[id_domain] = np.ones((x.shape[0],1))
        self._energies[id_domain] = en
        self._weights_energy[id_domain] = np.ones_like(en)
        self._Ndomains += 1
        self._last_domain_ID += 1
        return id_domain

    def remove_domain(id_domain):
        self._domains.remove(id_domain)
        self._x.pop(id_domain)
        self._energies.pop(id_domain)
        self._weights_coord.pop(id_domain)
        self._weights_energy.pop(id_domain)
        self._weights.pop(id_domain)
        for i_matrix in self._Nm:
            self._fitDomains[i_matrix].pop(id_domain)
        self._Ndomains -= 1

    def set_fit_domain(self, n_matrix: int, id_domain: int, states: Tuple[int, ...] = None):
        """ Specify the domain and states that a diabatic potential matrix
        should fit.
        """
        if states is None:
            states = tuple([s for s in range(self._Ns)])

        assert n_matrix < self._Nm, "Matrix index should be less than " \
            + f"{self._Nm}, got {n_matrix}."
        assert all([0 <= s < self._Ns for s in states]), "One of specified " \
            + f"states {states} is out of range, " \
            + f"should be 0 <= s < {self._Ns}."

        self._states_by_domain[n_matrix][id_domain] = states
        self._auto_fit = False

    def set_fit_all_domain(self, n_matrix: int):
        for idd in self._domain_IDs:
            self.set_fit_domain(n_matrix, idd)

    def set_domain_weight(self, id_domain, weight):
        """ Assign a fixed weight to a coordinate domain. """
        self._manually_weighted_domains.add(id_domain)
        self._weights[id_domain] = weight

    def unset_domain_weight(self, id_domain):
        """ Unassign fixed weight to a coordinate domain. """
        self._manually_weighted_domains.remove(id_domain)
        domain_shape = self._energies[id_domain].shape
        self._weights_coord[id_domain] = np.ones((domain_shape[0],1))
        self._weights_energy[id_domain] = np.ones(domain_shape)
        self._weights[id_domain] = np.ones(domain_shape)

    def set_weight_function(self, wfun: callable, apply_to):
        """ Set user-defined energy-based weighting function for the residuals """
        if apply_to == "energy":
            self._wfun_energy = wfun
        elif apply_to == "coord":
            self._wfun_coord = wfun
        else:
            raise ValueError("Weighting function must be applied to either \'coord\' or \'energy\'.")

    def set_gauss_wfun(self, mu, sigma, apply_to):
        """ Set Gaussian weighting function for the residuals

        w = exp(- 1/2 ((y-mu)/sigma)**2 )
        """
        self.set_weight_function(lambda y: np.exp(-0.5*((y-mu)/sigma)**2), apply_to)
    
    def set_gaussband_wfun(self, mu, sigma, apply_to):
        """ Set band weighting function for the residuals, with Gaussian tails

        w = 1 if mu[0] < y < mu[1]
        otherwise:        
        w = exp(- 1/2 ((y-mu[i])/sigma[i])**2 ), i=0,1 if y<mu[0],mu[1] resp
        """
        def gaussband(y, mu_down, mu_up, sigma_down, sigma_up):
            if y < mu_down:
                return np.exp(-0.5*((y-mu_down)/sigma_down)**2)
            elif y > mu_up:
                return np.exp(-0.5*((y-mu_up)/sigma_up)**2)
            else:
                return 1
        self.set_weight_function(lambda y: gaussband(y, mu[0], mu[1], sigma[0], sigma[1]), apply_to)

    def set_exp_wfun(self, x0, beta, apply_to):
        """ Set exponential decaying weighting function for the residuals

        w = exp(-(x-x0)/beta) if x > x0
        w = 1   otherwise
        """
        self.set_weight_function(lambda y: np.exp(-(y-x0)/beta) if y>x0 else 1, apply_to)

    def set_expband_wfun(self, x0, beta, apply_to):
        """ Set band weighting function for the residuals, with exponential decaying tails

        w = 1 if x0[0] < x < x0[1]
        w = exp( (x-x0[0])/beta[0]) if y<x0[0]
        w = exp(-(x-x0[1])/beta[1]) if y<x0[1]
        """
        def expband(y, y_down, y_up, beta_down, beta_up):
            if y < y_down:
                return np.exp((y-y_down)/beta_down)
            elif y > y_up:
                return np.exp(-(y-mu_up)/beta_up)
            else:
                return 1
        self.set_weight_function(lambda y: expband(y, x0[0], x0[1], beta[0], beta[1]), apply_to)

    def compute_weights(self):
        """
        Precompute weights assigned to points in coordinate space or the corresponding energies
        if weighting functions have been specified.
        """
        # Leave out domains whose weights were specified manually
        for id_ in self._domain_IDs.difference(self._manually_weighted_domains):
            # Coordinate-based weights
            if self._wfun_coord:
                self._weights_coord[id_] = self._wfun_coord(self._x[id_])
            # Energy-based weights
            if self._wfun_energy:
                self._weights_energy[id_] = self._wfun_energy(self._energies[id_])
            # Combine
            self._weights[id_] = self._weights_coord[id_] * self._weights_energy[id_]


    def compute_errors(self):
        """
        Compute weighted and unweighted RMSE and MAE between
        reference adiabatic energies and those deduced from of all
        current diabatic matrices in Wout, within the associated domains
        """
        rmse_list = []
        wrmse_list = []
        mae_list = []
        wmae_list = []

        # Compute errors for each matrix
        for im in range(self._Nm):
            res = []
            w = []

            # Evaluate adiabatic matrices over each domain
            for id_, states in self.states_by_domain[im].items():
                x = self._x[id_]
                Wx = self._Wout[im](x)
                Vx, Sx = adiabatic(Wx)

                # Compute residual against selected states over the domain
                for s in states:
                    res.append(self._energies[id_][:,s] - Vx[:,s])
                    w.append(np.broadcast_to(self._weights[id_][:,s], res[-1].shape))

            # Compute errors and save
            res = np.hstack(res)
            w = np.hstack(w)

            resabs = np.abs(res)
            res2 = np.dot(res,res)
            sumw = np.sum(w)

            rmse = np.sqrt(np.sum(res2)/self.n_fitted_points(im))
            wrmse = np.sqrt(np.sum(w * res2)/sumw)
            mae = np.sum(resabs)/self.n_fitted_points(im)
            wmae = np.sum(w * resabs)/sumw

            self._results[im].rmse = rmse
            self._results[im].wrmse = wrmse
            self._results[im].mae = mae
            self._results[im].wmae = wmae


    def _compute_residuals(self, W: SymPolyMat, domains: dict):
        """
        Compute residual between reference energies and those deduced from a diabatic potential
        matrix.
        Parameters:
        * W : SymPolymat, diabatic matrix
        * domains : dict id_ -> (s1, s2, ...), where id_ is a domain id (int) and s1, s2, ...
          are electronic states.
        Return:
        * res : 1D np.ndarray of residuals
        """
        residuals = []
        for id_, states in domains.items():
            # Compute adiabatic potential from diabatic ansatz
            x = self._x[id_]
            Wx = W(x)
            Vx, Sx = adiabatic(Wx)

            # Retreive data energies and compute (weighted) residuals over the domain
            for s in states:
                Vdata = self._energies[id_][:,s]
                if np.any(np.isnan(Vdata)):
                    raise(ValueError(f"Found NaN energies in domain {id_}, state {s}. "
                        + "Please deselect from fitting dataset."))

                residuals.append(Vx[:,s]-Vdata)

        return np.hstack(tuple(residuals))

    def _cost(self, c, keys, x0s, domains, weights):
        """
        Compute cost function for finding optimal diabatic anzats coefficients.

        This method is passed to the optimizer in order to find the optimal coefficients c
        such that the adiabatic surfaces obtained from diagonalization of the
        diabatic ansatz fit the adiabatic data.

        Parameters:
        * c : 1d list/numpy array of coefficients
        * keys : list of keys (i,j,(p1,p2,...)) mapping each of the coefficients
          to a specific matrix element and monomial key
        * x0s : dict of (i,j) -> x0 mapping matrix indices to origin point
        * domains : dict of id_ -> (s1, s2, ...) mapping domain index to tuple of state indices
        """

        # Construct diabatic matrix by reassigning coefficients to
        # powers of each of the matrix elements
        W = SymPolyMat.construct(self._Ns, self._Nd, keys, c, x0s)

        res = self._compute_residuals(W, domains)
        # Store for verbose output
        self._last_residuals = res
        wrmse = wRMSE(res, weights)
        return wrmse
    
    def _verbose_cost(self, c, keys, x0s, domains, weights):
        """ Wrapper of cost function which also prints out optimization progress. """
        wrmse = self._cost(c, keys, x0s, domains, weights)
        n = self._increment_cost_calls()
        if n % self._print_every == 0:
            rmse = RMSE(self._last_residuals)
            mae = MAE(self._last_residuals)
            wmae = wMAE(self._last_residuals, weights) 
            print("{:<10d} {:12.8e} {:12.8e} {:12.8e} {:12.8e}".format(n,wrmse,rmse,wmae,mae))
        return wrmse

    def _increment_cost_calls(self):
        self._n_cost_calls += 1
        return self._n_cost_calls

    def optimize(self, verbose=0, maxiter=1000):
        """ Run optimization

        Find best coefficients for polynomial diabatics and couplings fitting
        given adiabatic data.
        """
        # By default, if no specific domain setting is given, use all the data
        # in the database for the fit
        # NB: autoFit is false if Nm > 1
        if self._auto_fit:
            self.set_fit_all_domain(0)

        # Compute weights associated to points
        self.compute_weights()
        
        # Run a separate optimization for each diabatic matrix
        for i_matrix in range(self._Nm):
            self._results[i_matrix].reset()
            self._n_cost_calls = 0

            # Here each key in 'keys' refers to a coefficient in 'coeffs' and is
            # used for reconstructing the diabatic ansatzes during the optimization
            # and at the end
            coeffs, keys = self._Wguess[i_matrix].coeffs_and_keys()
            origins = self._Wguess[i_matrix].get_all_x0()
            this_matrix_domains = self._states_by_domain[i_matrix]
            weights = []
            for id_, states in this_matrix_domains.items():
                for s in states:
                    weights.append(self._weights[id_][:,s])
            weights = np.hstack(tuple(weights))

            if verbose == 1:
                cost_fun = self._verbose_cost
                print("I    " + "COST")
            else:
                cost_fun = self._cost
            
            optres = scipy.optimize.minimize(
                    cost_fun,    # Objective function to minimize
                    coeffs,                 # Initial guess
                    args=(keys, origins, this_matrix_domains, weights),   # other arguments passed to objective function
                    method="l-bfgs-b",
                    # method="trust-constr",
                    options={
                        "gtol": 1e-08,      # Termination conditions (quality)
                        # "xtol": 1e-08,
                        "maxiter": maxiter, # Termination condition (# iterations) 
                        # "verbose": verbose, # Printing option
                        }
                    )
            
            self._Wout[i_matrix] = SymPolyMat.construct(self._Ns, self._Nd, keys, optres.x, origins)
            self._results[i_matrix].from_OptimizeResult(optres)

        self.compute_errors()
        
        return self._Wout

    def to_JSON_dict(self):
        dct = {
                "__Diabatizer__" : True,
                "Nd"                        : self._Nd,
                "Ns"                        : self._Ns,
                "Nm"                        : self._Nm,
                "Wguess"                    : [Wg.to_JSON_dict() for Wg in self._Wguess],
                "Wout"                      : [Wo.to_JSON_dict() for Wo in self._Wout],
                "x"                         : {id_ : x.tolist() for id_,x in self._x.items()},
                "energies"                  : {id_ : e.tolist() for id_,e in self._energies.items()},
                "states_by_domain"          : [{id_ : str(states) for id_,states in dct.items()}
                                                    for dct in self._states_by_domain],
                "domain_IDs"                : list(self._domain_IDs),
                "Ndomains"                  : self._Ndomains,
                "last_domain_ID"            : self._last_domain_ID,
                "auto_fit"                  : self._auto_fit,
                "wfun_coord"                : None, # self._wfun_coord,
                "wfun_energy"               : None, # self._wfun_energy,
                "manually_weighted_domains" : list(self._manually_weighted_domains),
                "weights"                   : {id_: w.tolist() for id_, w in self._weights.items()},
                "weights_coord"             : {id_: w.tolist() for id_, w in self._weights_coord.items()},
                "weights_energy"            : {id_: w.tolist() for id_, w in self._weights_energy.items()},
                "print_every"               : self._print_every,
                "n_cost_calls"              : self._n_cost_calls,
                "last_residuals"            : self._last_residuals.tolist(),
                "results"                   : [r.to_JSON_dict() for r in self.results],
                }
        return dct

    @staticmethod
    def from_JSON_dict(dct):
        """
        Deserialize Diabatizer in an intrusive way, i.e. setting some private members manually.
        ! This may become dangerous if the structure of the Diabatizer class change.
        """
        Wguess = []
        for Wg_dict in dct["Wguess"]:
            if "__DampedSymPolyMat__" in Wg_dict:
                Wguess.append(DampedSymPolyMat.from_JSON_dict(Wg_dict))
                continue
            Wguess.append(SymPolyMat.from_JSON_dict(Wg_dict))

        Wout = []
        for Wo_dict in dct["Wout"]:
            if "__DampedSymPolyMat__" in Wo_dict:
                Wout.append(DampedSymPolyMat.from_JSON_dict(Wo_dict))
                continue
            Wout.append(SymPolyMat.from_JSON_dict(Wo_dict))

        diab = Diabatizer(dct["Nd"], dct["Ns"], dct["Nm"], Wguess)
        diab._Wguess = Wguess
        diab._Wout = Wout
        diab._x = {id_: np.array(xlist) for id_, xlist in dct["x"].items()}
        diab._energies = {
                id_: np.array(elist) \
                        for id_, elist in dct["energies"].items()
                        }
        diab._states_by_domain = [
                {
                    id_: _str2tuple(str_states) \
                        for id_, str_states in sbd_dct.items()
                } for sbd_dct in dct["states_by_domain"]
                ]
        diab._domain_IDs = set(dct["domain_IDs"])
        diab._Ndomains = dct["Ndomains"]
        diab._last_domain_ID = dct["last_domain_ID"]
        diab._auto_fit = dct["auto_fit"]
        diab._wfun_coord = None
        diab._wfun_energy = None
        diab._manually_weighted_domains = set(dct["manually_weighted_domains"])
        diab._weights = {id_: np.array(w) for id_, w in dct["weights"].items()}
        diab._weights_coord = {id_: np.array(w) for id_, w in dct["weights_coord"].items()}
        diab._weights_energy = {id_: np.array(w) for id_, w in dct["weights_energy"].items()}
        diab._print_every = dct["print_every"]
        diab._n_cost_calls = dct["n_cost_calls"]
        diab._last_residuals = np.array(dct["last_residuals"])
        diab._results = [Results.from_JSON_dict(rdct) for rdct in dct["results"]]


class SingleDiabatizer(Diabatizer):
    def __init__(self, Ns, Nd, diab_guess: SymPolyMat = None, **kwargs):
        super().__init__(Ns, Nd, 1, [diab_guess], **kwargs)

    def rmse(self):
        return super().results[0].rmse

    def mae(self):
        return super().results[0].mae

    @property
    def fit(self):
        return self._fit[0]

    @property
    def Wguess(self):
        return self._Wguess[0]

    @Wguess.setter
    def Wguess(self, guess):
        self._Wguess[0] = guess

    @property
    def Wout(self):
        return self._Wout[0]

### NON-CLASS FUNCTIONS ###

def _str2tuple(s):
    """
    Convert string to original tuple of integers.
    Assuming that the string was obtained with str(t),
    t being the tuple.
    """
    return tuple(map(int,s.strip('()').split(', ')))

# In C++, use template
def save_to_JSON(obj, fname):
    with open(fname, "w") as f:
        json.dump(f, obj.to_JSON_dict())

# In C++, maybe use variant?
def load_from_JSON(fname):
    with open(fname, "r") as f:
        dct = json.load(f)

    if "__NdPoly__" in dct:
        return NdPoly.from_JSON_dict(dct)

    if "__SymPolyMat__" in dct:
        if "__DampedSymPolyMat__" in dct:
            return DampedSymPolyMat.from_JSON_dict(dct)
        return SymPolyMat.from_JSON_dict(dct)

    if "__DampingFunction__" in dct:
        if "__One__" in dct:
            return One.from_JSON_dict(dct)
        if "__Gaussian__" in dct:
            return Gaussian.from_JSON_dict(dct)
        if "__Lorentzian__" in dct:
            return Lorenzian.from_JSON_dict(dct)
        raise Warning("Serialized abstract DampingFunction instance.")

    if "__Results__" in dct:
        return Results.from_JSON_dict(dct)
    
    if "__Diabatizer__" in dct:
        return Diabatizer.from_JSON_dict(dct)


def adiabatic2(W1, W2, W12, sign):
    """ Return analytical 2-states adiabatic potentials from diabatic ones and coupling. """
    m = 0.5*(W1+W2)
    p = W1*W2 - W12**2
    return m + sign * np.sqrt(m**2 - p)

def adiabatic(W):
    """ Return numerical N-states adiabatic potentials from diabatic ones and couplings. """
    return np.linalg.eigh(W)

def switch_poly(x, smin=0, smax=1):
    """ Polynomial switching function with zero first and second derivatives at 0 and 1.
    See: Zhang et al., J. Phys. Chem. A, 2004, 108, 41, 8980–8986.
    https://doi.org/10.1021/jp048339l
    Beware of the typo: should be -10*x^3 instead of -10*x^2
    """
    xs = (x-smin)/(smax-smin)
    xs = np.maximum(np.minimum(xs, 1), 0)
    # if xs == 0:
    #     return 1
    # elif xs == 1:
    #     return 0
    # else:
    #     return 1 + xs**3 * (-10 + xs*(15 - 6*xs))
    return 1 + xs**3 * (-10 + xs*(15 - 6*xs))

def switch_sinsin(x, smin=0, smax=1):
    """ Sine of sine switching function with zero first and second derivative at 0 and 1.
    See: Sahnoun et al., J. Phys. Chem. A, 2018, 122, 11, 3004–3012.
    https://doi/10.1021/acs.jpca.8b00150
    """
    xs = (x-smin)/(smax-smin)
    xs = np.maximum(np.minimum(xs, 1), 0)
    # if xs == 0:
    #     return 1
    # elif xs == 1:
    #     return 0
    # else:
    #     return np.sin(np.pi/2 * np.sin(np.pi/2 * xs) )
    return (1 - np.sin(np.pi/2*np.sin(np.pi/2*(2*xs - 1))))/2
    
def RMSE(residuals):
    """ Compute unweighted Root-Mean-Square Error from residuals. """
    return np.sqrt(np.sum(residuals**2)/len(residuals))

def wRMSE(residuals, weights):
    """ Compute weighted Root-Mean-Square Error from residuals and weights. """
    return np.sqrt(np.sum(weights * residuals**2)/np.sum(weights))

def MAE(residuals):
    """ Compute unweighted Mean-Average Error from residuals. """
    return np.sum(np.abs(residuals))/len(residuals)

def wMAE(residuals, weights):
    """ Compute weighted Mean-Average Error from residuals and weights. """
    return np.sum(weights * np.abs(residuals))/np.sum(weights)


### MAIN ###
def main(argv) -> int:
    A = SymPolyMat.eye(2,3)
    A[0,0].grow_degree(3,0.1)
    A[1,0].grow_degree(4,0.2)
    A[1,1].grow_degree(2,0.3)
    # print(A)
    d = SingleDiabatizer(2,3,A)
    print(d.Wguess)
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
