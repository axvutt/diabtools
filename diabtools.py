from __future__ import annotations
import sys
from copy import *
from typing import List, Tuple, Dict, Union
from collections import UserDict
import math
import pickle
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

    # TODO: make slots work so as to avoid extra attribute declaration
    # __slots__ = ('_Nd', '_degree', '_x0', '_zeroPower', '_x0')

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
        s = super().__repr__()
        s += f", ORIGIN: {self._x0}"
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
        result = copy(self)
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
        result = copy(self)
        if isinstance(other, self.__class__):
            # assert np.all(self._x0 == other.x0), "Origins of added polynomials do not match."
            raise(NotImplementedError("Product between NdPoly's not yet implemented"))
        elif isinstance(other, (float, int)):
            for powers in self.powers():
                self[powers] *= other
            return self
        else:
            raise(TypeError)

    @staticmethod
    def empty(Nd):
        """ Return an empty Nd-dimensional polynomial """
        # Dirty way: create a contant zero poly and remove the dict item
        powers = tuple([0 for _ in range(Nd)])
        P = NdPoly.zero(Nd) 
        del P[powers]
        return P

    @staticmethod
    def zero(Nd):
        """ Return a Nd-dimensional polynomial with zero constant term only """
        powers = tuple([0 for _ in range(Nd)])
        return NdPoly({powers: 0})

    @staticmethod
    def one(Nd):
        """ Return a Nd-dimensional polynomial with unit constant term only """
        powers = tuple([0 for _ in range(Nd)])
        return NdPoly({powers: 1})

    @staticmethod
    def zero_like(P: NdPoly):
        """ Return a polynomial with all powers in P but with zero coefficients """
        return NdPoly({ p : 0 for p in list(P.powers()) })

    @staticmethod
    def one_like(P: NdPoly):
        """ Return a polynomial with all powers in P, with unit constant term
        and all the others with zero coefficient """
        return NdPoly({ p : 0 for p in list(P.powers()) }) + 1

    @staticmethod
    def _get_tuples(length, total):
        """ Generator which essentially constructs all the
        tuples of a given length and sum of its integer entries.

        Taken from https://stackoverflow.com/questions/29170953
        """
        # Idea:
        # Make 1D tuples from total to zero
        # Prepend index such that the sum of the 2D tuple is total
        # Make 1D tuples from (total-1) to zero
        # Prepend index such that the sum of the 2D tuple is (total-1)
        # ................... (total-2) to zero
        # .................................................. (total-2)
        # For each generated 2D tuple:
        # Prepend index such that the sum of the 3D tuple is total
        # [... continue if 4D ...]
        # Prepend index such that the sum of the 3D tuple is (total-1)
        # etc...

        if length == 1:
            yield (total,)
            return

        for i in range(total + 1):
            for t in NdPoly._get_tuples(length - 1, total - i):
                yield (i,) + t

    @staticmethod
    def fill_maxdegree(Nd, degree, fill, max_pdeg = None):
        """ Return a polynomial with all powers up to a given degree """
        P = NdPoly.empty(Nd)
        P.grow_degree(degree, fill, max_pdeg)
        return P

    @staticmethod
    def zero_maxdegree(Nd, degree, max_pdeg = None):
        """ Return a polynomial with all powers up to a given degree """
        return NdPoly.fill_maxdegree(Nd, degree, 0, max_pdeg)

    @staticmethod
    def one_maxdegree(Nd, degree):
        return NdPoly.zero_maxdegree(Nd, degree) + 1


class SymPolyMat():
    """ Real symmetric matrix of multivariate polynomial functions """
    def __init__(self, Ns, Nd):
        self._Ns = Ns
        self._Nd = Nd
        self._polys = [ NdPoly.empty(Nd) for i in range(Ns) for j in range(i+1) ]

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
        s = ""
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

    @staticmethod
    def read_from_file(filename):
        with open(filename, "rb") as fin:
            W = pickle.load(fin)
        return W

    @staticmethod
    def zero(Ns, Nd):
        """ Create zero matrix.
        Each matrix term is a constant monomial with zero coefficient.
        """
        M = SymPolyMat(Ns,Nd)
        for i in range(Ns):
            for j in range(Ns):
                M[i,j] = NdPoly.zero(Nd)
        return M

    @staticmethod
    def zero_like(other: SymPolyMat):
        """ Create copy with zero polynomial coefficients.
        Each matrix element is a polynomial with all powers as in other,
        but whose coefficients are all zero.
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
        """
        I = SymPolyMat.zero(Ns, Nd)
        for i in range(Ns):
            I[i,i] = NdPoly({tuple([0 for _ in range(Nd)]): 1})
        return I

    @staticmethod
    def eye_like(other: SymPolyMat):
        """ Create identity matrix, with polynomial coefficients of other.
        Each matrix element is a polynomial with all powers as in other,
        but whose coefficients are all zero except along the diagonal where
        the constant term .
        Warning: If there is no constant term in a diagonal element of other,
        this will be added (with a coefficient 1).
        """
        newmat = SymPolyMat.zero_like(other)
        for i in range(newmat.Ns):
            for j in range(i+1):
                if i == j :
                    newmat[i,j][newmat[i,j].zeroPower] = 1
        return newmat

class DampedSymPolyMat(SymPolyMat):
    """ Symmetric Matrix of Polynomials, with damping functions """
    def __init__(self, Ns, Nd):
        super().__init__(Ns, Nd)
        self._damp = [lambda x: 1 for i in range(Ns) for j in range(i+1)]

    def set_damping(self, pos, dfun: callable):
        i, j = pos
        if j > i :
            i, j = j, i
        self._damp[i*(i+1)//2 + j] = dfun

    def get_damping(self, pos) -> callable:
        i, j = pos
        if j > i :
            i, j = j, i
        return self._damp[i*(i+1)//2 + j]

    def __call__(self, x):
        x = np.atleast_2d(x)
        if self._Nd == 1 and x.shape[0] == 1:   # If x is a row, make it a column
            x = x.T
        Wx = super().__call__(x)
        Dx = np.zeros(Wx.shape)
        for i in range(self._Ns):   # Compute lower triangular part
            for j in range(i+1):
                Dx[:,i,j] = self.get_damping((i,j))(x)
                Dx[:,j,i] = Dx[:,i,j]
        return Wx * Dx

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

class DampingFunction:
    """ Abstract base class for damping functions
    Subclasses should implement __call__"""
    def __init__(self,x0):
        self._x0 = x0

    @property
    def x0(self):
        return self._x0

    @x0.setter
    def x0(self, x0):
        self._x0 = x0

class Gaussian(DampingFunction):
    def __init__(self, x0, sigma):
        super().__init__(x0)
        if math.isclose(sigma,0):
            raise(ValueError("Zero std deviation."))
        self._sigma = sigma

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, sigma):
        self._sigma = sigma

    def __call__(self,x):
        return np.exp(-0.5*((x-self.x0)/self.sigma)**2)

class Lorentzian(DampingFunction):
    def __init__(self, x0, gamma):
        super().__init__(x0)
        if math.isclose(gamma,0):
            raise(ValueError("Zero gamma width parameter."))
        self._gamma = gamma

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        self._gamma = gamma

    def __call__(self,x):
        return 1/(1 + ((x-self.x0)/self.gamma)**2)

class Diabatizer:
    def __init__(self, Ns, Nd, Nm = 1, diab_guess: List[SymPolyMat] = None):
        self._Nd = Nd
        self._Ns = Ns
        self._Nm = Nm
        if diab_guess is not None:
            self._Wguess = diab_guess
        else :
            self._Wguess = [SymPolyMat.eye(Ns, Nd) for _ in range(Nm)]
        self._Wout = self._Wguess
        self._fit = [None for _ in range(Nm)]
        self._x = dict()
        self._energies = dict()
        self._domain_map = {i_matrix : {} for i_matrix in range(Nm)}
        self._domain_IDs = set()
        self._Ndomains = 0
        self._last_domain_ID = 0
        self._auto_fit = True
        self._weight_coord = lambda x: 1
        self._weight_energy = lambda x: 1
        self._switches = None

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
    def domain_map(self):
        """
        Mapping of diabatic matrices to assigned domain and states.
        Returns a dict<int,<dict<int,tuple<int>>>
        * First level keys: id of diabatic matrix
        * Second level keys: id of domain
        * Second level values: states
        """
        return self._domain_map

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
        for dom_id, states in self._domain_map[i_matrix].items():
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

    def rmse(self):
        """ Compute RMSE between reference adiabatic energies and those
        deduced from of all current diabatic matrices in Wout, within
        the associated domains """
        rmse_list = []
        for im, mat_map in enumerate(self.domain_map):
            res = []
            for dom_id, states in mat_map.items():
                x = self._x[dom_id]
                Wx = self._Wout[im](x)
                Vx, Sx = adiabatic(Wx)
                for s in states:
                    res.append(self._energies[dom_id][s] - Vx[:,s])
            rmse = np.sqrt(np.sum(res**2))
            rmse_list.append(rmse)
        return self._rmse

    @property
    def mae(self):
        return self._mae

    @property
    def fit(self):
        return self._fit

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
        self._energies[id_domain] = en
        self._Ndomains += 1
        self._last_domain_ID += 1
        return id_domain

    def remove_domain(id_domain):
        self._domains.remove(id_domain)
        self._x.pop(id_domain)
        self._energies.pop(id_domain)
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

        self._domain_map[n_matrix][id_domain] = states
        self._auto_fit = False

    def set_fit_all_domain(self, n_matrix: int):
        for idd in self._domain_IDs:
            self.set_fit_domain(n_matrix, idd)

    def _coeffs_mapping(self, W: SymPolyMat) -> dict:
        """ Create mapping from a given (i,j,powers) to corresponding
        coefficient. """
        coeffs_map = dict()
        for i in range(self._Ns):
            for j in range(i+1):
                for powers, coeff in W[i,j].items():
                    coeffs_map[(i,j,powers)] = coeff 
        return coeffs_map

    def _rebuild_diabatic(self, keys, coeffs, dict_x0) -> SymPolyMat:
        """ Reconstruct diabatic matrix from flat list of coefficients
        Parameters:
        * keys = list of (i,j,powers)
        * coeffs = [c1,c2,...]
        * x0 = dict of origins {(i,j): x0, ...}
        Return:
        * Diabatic matrix W such that
          W[i,j][powers] = c_{powers}^(i,j)
        """

        W = SymPolyMat(self._Ns, self._Nd)

        # Coefficients
        for n, key in enumerate(keys):
            i, j, powers = key
            W[i,j][powers] = coeffs[n]

        # Shifts
        for idx, x0 in dict_x0.items():
            W[idx].x0 = x0

        return W
    
    @property
    def weight_function(self):
        return self._weight

    def weights(self):
        return {domain: self._weight(e) for domain, energies in self._energies.items()}

    def set_custom_weight(self, wfun: callable):
        """ Set user-defined energy-based weighting function for the residuals """
        self._weight = wfun

    def set_gauss_weights(self, mu, sigma):
        """ Set Gaussian weighting function for the residuals

        w = exp(- 1/2 ((y-mu)/sigma)**2 )
        """
        self._weight = lambda y: np.exp(-0.5*((y-mu)/sigma)**2)
    
    def set_gaussband_weights(self, mu, sigma):
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
        self._weight = lambda y: gaussband(y, mu[0], mu[1], sigma[0], sigma[1])

    def set_exp_weights(self, y0, beta):
        """ Set exponential decaying weighting function for the residuals

        w = exp(-(y-mu)/beta) if y > mu
        w = 1   otherwise
        """
        self._weight = lambda y: np.exp(-(y-y0)/beta) if y>y0 else 1

    def set_expband_weights(self, y0, beta):
        """ Set band weighting function for the residuals, with exponential decaying tails

        w = 1 if mu[0] < y < mu[1]
        w = exp( (y-mu[0])/beta[0]) if y<mu[0]
        w = exp(-(y-mu[1])/beta[1]) if y<mu[1]
        """
        def expband(y, y_down, y_up, beta_down, beta_up):
            if y < y_down:
                return np.exp((y-y_down)/beta_down)
            elif y > y_up:
                return np.exp(-(y-mu_up)/beta_up)
            else:
                return 1
        self._weight = lambda y: expband(y, y0[0], y0[1], beta[0], beta[1])

    def residual(self, c, keys, x0s, i_matrix):
        """ Compute residual for finding optimal diabatic anzats coefficients.

        This method is passed to the optimizer in order to find the optimal coefficients c
        such that the adiabatic surfaces obtained from diagonalization of the
        diabatic ansatz fit the adiabatic data.

        Parameters:
        * c : 1d list/numpy array of coefficients
        * keys : list of keys (i,j,(p1,p2,...)) mapping each of the coefficients
          to a specific matrix element and monomial key
        * x0s : dict of (i,j) matrix indices mapped to the corresponding origin
        * i_matrix : index of the diabatic matrix to fit
        """

        # Construct diabatic matrix by reassigning coefficients to
        # powers of each of the matrix elements
        W = self._rebuild_diabatic(keys, c, x0s)

        # Compute residual function
        residuals = []
        for id_domain, states in self._domain_map[i_matrix].items():
            # Compute W(x) over the domain, diagonalize the obtained point-wise matrices
            x = self._x[id_domain]
            Wx = W(x)
            Vx, Sx = adiabatic(Wx)

            # Retreive data energies and compute (weighted) residuals over the domain
            for s in states:
                Vdata = self._energies[id_domain][:,s]
                if np.any(np.isnan(Vdata)):
                    raise(ValueError(f"Found NaN energies in domain {id_domain}, state {s}. "
                        + "Please deselect from fitting dataset."))
                residuals.append(
                        self._weight_coord(x) * self._weight_energy(Vdata) * (Vx[:,s]-Vdata)
                        )

        # Recast into 1d np.ndarray
        f = np.hstack(tuple(residuals))
        return f 

    def optimize(self, verbose=0, max_nfev=None):
        """ Run optimization

        Find best coefficients for polynomial diabatics and couplings fitting
        given adiabatic data.
        """
        # By default, if no specific domain setting is given, use all the data
        # in the database for the fit
        # NB: autoFit is false if Nm > 1
        if self._auto_fit:
            self.set_fit_all_domain(0)

        # Run a separate optimization for each diabatic matrix
        for i_matrix in range(self._Nm):
            # Here each key in 'keys' refers to a coefficient in 'coeffs' and is
            # used for reconstructing the diabatic ansatzes during the optimization
            # and at the end
            keys2coeffs = self._coeffs_mapping(self._Wguess[i_matrix])
            keys = tuple(keys2coeffs.keys())
            guess_coeffs = list(keys2coeffs.values())
            origins = self._Wguess[i_matrix].get_all_x0()

            lsfit = scipy.optimize.least_squares(
                    self.residual,          # Residual function
                    np.array(guess_coeffs), # Initial guess
                    gtol=1e-10,             # Termination conditions (quality)
                    ftol=1e-10,
                    xtol=1e-10,
                    args=(keys, origins, i_matrix),   # keys and origins of elements of each matrix, reconstructed in residual
                    verbose=verbose,        # Printing option
                    max_nfev=max_nfev)      # Termination condition (# iterations)

            self._fit[i_matrix] = lsfit
            self._rmse[i_matrix] = np.sqrt(np.dot(lsfit.fun, lsfit.fun)/self.n_fitted_points(i_matrix))
            self._mae[i_matrix] = np.sum(np.abs(lsfit.fun))/self.n_fitted_points(i_matrix)
            self._Wout[i_matrix] = self._rebuild_diabatic(
                    keys,
                    lsfit.x,
                    self._Wguess[i_matrix].get_all_x0()
                    )

        return self._Wout


class SingleDiabatizer(Diabatizer):
    def __init__(self, Ns, Nd, diab_guess: SymPolyMat = None, **kwargs):
        super().__init__(Ns, Nd, 1, [diab_guess], **kwargs)

    @property
    def rmse(self):
        return self._rmse[0]

    @property
    def mae(self):
        return self._mae[0]

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
    xs = math.max(math.min(xs, 1), 0)
    # if xs == 0:
    #     return 1
    # elif xs == 1:
    #     return 0
    # else:
    #     return 1 - x**3 * (10 + x*(15 - 6*x))
    return 1 - x**3 * (10 + x*(15 - 6*x))

def switch_sinsin(x, smin=0, smax=1):
    """ Sine of sine switching function with zero first and second derivative at 0 and 1.
    See: Sahnoun et al., J. Phys. Chem. A, 2018, 122, 11, 3004–3012.
    https://doi/10.1021/acs.jpca.8b00150
    """
    xs = (x-smin)/(smax-smin)
    xs = math.max(math.min(xs, 1), 0)
    # if xs == 0:
    #     return 1
    # elif xs == 1:
    #     return 0
    # else:
    #     return np.sin(np.pi/2 * np.sin(np.pi/2 * x) )
    return (1 + np.sin(np.pi/2*np.sin(np.pi/2*(2*x - 1))))/2
    

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
