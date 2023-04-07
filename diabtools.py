from __future__ import annotations
import sys
from copy import *
from typing import List, Tuple, Dict, Union
from collections import UserDict
import math
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
    def __init__(self, data):
        if len(data) == 0:
            raise(ValueError("Cannot make empty polynomial with unknown " \
                    + "dimensionality. Use NdPoly.empty() instead."))

        # Aliases for keys and values
        self.powers = self.keys
        self.coeffs = self.values

        self._Nd = None
        self._degree = None
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
        return self._degree
    
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

    def __setitem__(self, powers, coeff):
        if self._Nd is None and len(powers) != 0 :
            self._Nd = len(powers)
        else:
            assert len(powers) == self._Nd, f"Inappropriate powers {powers}. Expected {self._Nd} integers."
        super().__setitem__(powers, coeff)
        self.data = dict(sorted(self.data.items())) # Rough sorting, may need improvement
        self._degree = max([sum(p) for p in self.powers()])

    def __call__(self, x: np.ndarray):
        """ Calculate value of the polynomial at x.
        
        Parameters:
        x
        N * Nd ndarray containing the values of the coordinates where to evaluate the polynomial.

        Returns:
        P
        np.ndarray containing the N values of the polynomial over the given x meshgrids.
        """
        P = 0
        x = np.atleast_2d(x)
        if self._Nd == 1 and x.shape[0] == 1:   # If x is a row, make it a column
            x = x.T
        assert x.shape[1] == self._Nd, "Wrong point dimensionality "\
                + f"{x.shape[1]}, should be {self._Nd}."
        for powers, coeff in self.items():
            monomial = 1
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

class SymPolyMat():
    """ Real symmetric matrix of multivariate polynomial functions """
    def __init__(self, Ns, Nd):
        self._Ns = Ns
        self._Nd = Nd
        self._polys = [ NdPoly.empty(Nd) for i in range(Ns) for j in range(i+1) ]
        self._x0 = np.array([0 for _ in range(self._Nd)])

    @property
    def Ns(self):
        return self._Ns

    @property
    def Nd(self):
        return self._Nd

    @property
    def x0(self):
        return self._x0

    @x0.setter
    def x0(self, shift):
        for p in self._polys:
            p.x0 = shift
        self._x0 = shift

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
                s += f"({i},{j}): {self[i,j].__str__()}" + "\n"
        return s
   
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

class Diabatizer:
    def __init__(self, Ns, Nd, Nm = 1, diab_guess: List[SymPolyMat] = None, **kwargs):
        self._Nd = Nd
        self._Ns = Ns
        self._Nm = Nm
        if diab_guess is not None:
            self._Wguess = diab_guess
        else :
            self._Wguess = [SymPolyMat.eye(Ns, Nd) for _ in range(Nm)]
        self._Wout = self._Wguess
        self._x = dict()
        self._energies = dict()
        self._domainMap = {i_matrix : {} for i_matrix in range(Nm)}
        self._domainIDs = set()
        self._Ndomains = 0
        self._lastDomainID = 0
        self._verbosity = kwargs.get("verbosity", 0)
        self._autoFit = True

    @property
    def Nd(self):
        """ Get number of dimensions """
        return self._Nd

    @property
    def Ns(self):
        """ Get number of states """
        return self._Ns

    @property
    def Wguess(self):
        return self._Wguess

    @Wguess.setter
    def Wguess(self, guess):
        self._Wguess = guess

    @property
    def Wout(self):
        return self._Wout

    def addDomain(self, x: np.ndarray, en: np.ndarray):
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
        id_domain = self._lastDomainID
        self._domainIDs.add(id_domain)
        self._x[id_domain] = x
        self._energies[id_domain] = en
        self._Ndomains += 1
        self._lastDomainID += 1
        return id_domain

    def removeDomain(id_domain):
        self._domains.remove(id_domain)
        self._x.pop(id_domain)
        self._energies.pop(id_domain)
        for i_matrix in self._Nm:
            self._fitDomains[i_matrix].pop(id_domain)
        self._Ndomains -= 1

    def setFitDomain(self, n_matrix: int, id_domain: int, states: Tuple[int, ...] = None):
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

        self._domainMap[n_matrix][id_domain] = states
        self._autoFit = False

    def setFitAllDomains(self, n_matrix: int):
        for idd in self._domainIDs:
            self.setFitDomain(n_matrix, idd)

    def _coeffsMapping(self, W: SymPolyMat) -> dict:
        coeffs_map = dict()
        for i in range(self._Ns):
            for j in range(i+1):
                for powers, coeff in W[i,j].items():
                    coeffs_map[(i,j,powers)] = coeff 
        return coeffs_map

    def _rebuildDiabatic(self, keys, coeffs) -> SymPolyMat:
        W = SymPolyMat(self._Ns, self._Nd)
        for n, key in enumerate(keys):
            i, j, powers = key
            W[i,j][powers] = coeffs[n]
        return W

    def cost_function(self, c, keys, i_matrix):
        W_iteration = self._rebuildDiabatic(keys, c)
        f = np.array([])
        for id_domain, states in self._domainMap[i_matrix].items():
            W = W_iteration(self._x[id_domain])
            V, S = adiabatic(W)
            for s in states:
                f = np.hstack((f, V[:,s]-self._energies[id_domain][:,s]))
        return f 

    def optimize(self):
        """ Run optimization

        Find best coefficients for polynomial diabatics and couplings fitting
        given adiabatic data.
        """
        # By default, if no specific domain setting is given, use all the data
        # in the database for the fit
        # NB: autoFit is false if Nm > 1
        if self._autoFit:
            self.setFitAllDomains(0)

        # Run a separate optimization for each diabatic matrix
        for i_matrix in range(self._Nm):
            # Here each key in 'keys' refers to a coefficient in 'coeffs' and is
            # used for reconstructing the diabatic ansatzes during the optimization
            # and at the end
            keys = tuple(self._coeffsMapping(self._Wguess[i_matrix]).keys())
            guess_coeffs = list(self._coeffsMapping(self._Wguess[i_matrix]).values())

            lsfit = scipy.optimize.least_squares(
                    self.cost_function,
                    np.array(guess_coeffs),
                    gtol=1e-10,
                    ftol=1e-10,
                    xtol=1e-10,
                    args=(keys, i_matrix),
                    verbose=self._verbosity)

            self._Wout[i_matrix] = self._rebuildDiabatic(keys, lsfit.x)

        return lsfit, self._Wout


class SingleDiabatizer(Diabatizer):
    def __init__(self, Ns, Nd, diab_guess: SymPolyMat = None, **kwargs):
        super().__init__(Ns, Nd, 1, [diab_guess], **kwargs)

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

### MAIN ###
def main(argv) -> int:
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
