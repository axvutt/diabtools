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
        if self._Nd == 1:
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
    def __init__(self, Ns, Nd, diab_guess: SymPolyMat = None, **kwargs):
        self._Nd = Nd
        self._Ns = Ns
        self._Wguess = diab_guess if diab_guess is not None else SymPolyMat.eye(Ns, Nd)
        self._Wout = self._Wguess
        self._x = []
        self._energies = []
        self._states = []
        self._Nchunks = 0
        self._verbosity = kwargs.get("verbosity", 0)

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

    def addPoints(self, x: np.ndarray, en: np.ndarray, states: Tuple[int, ...]):
        """ Add N points to the database of energies to be fitted
        
        Parameters
        x
        np.ndarray of size Nd*N containing coordinates of the points to add

        en
        np.ndarray of Ns*N energy values at the corresponding coordinates

        states
        tuple of integers corresponding to states which should be considered
        Return
        n_chunk
        integer id of the data chunk added to the database
        """
        self._x.append(x)
        self._energies.append(en)
        self._states.append(states)
        self._Nchunks += 1
        return self._Nchunks

    def removeChunk(n_chunk):
        assert n_chunk < self._datachunks, "removeChunk(), chunk id too high."
        self._x[n_chunk] = []
        self._energies[n_chunk] = []
        self._states[n_chunk] = None

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

    def cost_function(self, c, keys):
        W_iteration = self._rebuildDiabatic(keys, c)
        f = np.array([])
        for n in range(self._Nchunks):
            W = W_iteration(self._x[n])
            V, S = adiabatic(W)
            for s in self._states[n]:
                f = np.hstack((f, V[:,s]-self._energies[n][:,s]))
        return f 

    def optimize(self):
        """ Run optimization

        Find best coefficients for polynomial diabatics and couplings fitting
        given adiabatic data.
        """
        
        # Here each key in 'keys' refers to a coefficient in 'coeffs' and is
        # used for reconstructing the diabatic ansatzes during the optimization
        # and at the end
        keys = tuple(self._coeffsMapping(self._Wguess).keys())
        guess_coeffs = list(self._coeffsMapping(self._Wguess).values())

        lsfit = scipy.optimize.least_squares(
                self.cost_function,
                np.array(guess_coeffs),
                gtol=1e-10,
                ftol=1e-10,
                xtol=1e-10,
                args=(keys,),
                verbose=self._verbosity)

        self._Wout = self._rebuildDiabatic(keys, lsfit.x)
        return lsfit, self._Wout


### NON-CLASS FUNCTIONS ###

def adiabatic2(W1, W2, W12, sign):
    """ Return analytical 2-states adiabatic potentials from diabatic ones and coupling. """
    m = 0.5*(W1+W2)
    p = W1*W2 - W12**2
    return m + sign * np.sqrt(m**2 - p)

def adiabatic(W):
    """ Return numerical N-states adiabatic potentials from diabatic ones and couplings. """
    return np.linalg.eigh(W)



### TEST CLASSES ###

class TestPoly:
    testdata = np.linspace(0,1,4)[:,np.newaxis]

    def test_InitEmptyError(self):
        with pytest.raises(ValueError):
            E = NdPoly({})

    def test_Empty(self):
        E = NdPoly.empty(3)
        assert E.Nd == 3
        assert len(E.powers()) == 0
        assert len(E.keys()) == 0
        assert E.zeroPower == (0,0,0)
        assert np.all(E.x0 == np.array([0,0,0]))

    def test_Zero(self):
        P = NdPoly.zero(3)
        X = np.repeat(self.testdata, 3, 1)
        assert P.zeroPower == (0,0,0)
        assert list(P.powers()) == [P.zeroPower]
        assert P[P.zeroPower] == 0
        assert np.all(P(X) == 0)

    def test_One(self):
        P = NdPoly.one(3)
        X = np.repeat(self.testdata, 3, 1)
        assert P.zeroPower == (0,0,0)
        assert list(P.powers()) == [P.zeroPower]
        assert P[P.zeroPower] == 1
        assert np.all(P(X) == 1)

    def test_ZeroLike(self):
        P = NdPoly({(1,0,0): 1, (0,1,0): 3.14, (0,0,1): -1})
        Q = NdPoly.zero_like(P)
        assert all([ p == q for p, q in zip(list(P.powers()), list(Q.powers()))])
        assert all([ c == 0 for c in list(Q.coeffs())])

    def test_OneLike(self):
        P = NdPoly({(0,0,0): 6.66, (1,0,0): 1, (0,1,0): 3.14, (0,0,1): -1})
        Q = NdPoly.one_like(P)
        assert all([ p == q for p, q in zip(list(P.powers()), list(Q.powers()))])
        assert Q[(0,0,0)] == 1
        Q[(0,0,0)] = 0
        assert all([ c == 0 for c in list(Q.coeffs())])

    def test_AddConstant(self):
        P = NdPoly({(1,0,0): 1, (0,1,0): 3.14, (0,0,1): -1})
        Q = P + 1.23
        assert Q[Q.zeroPower] == 1.23
        Q += 1
        assert Q[Q.zeroPower] == 2.23
        P += 1
        assert P[P.zeroPower] == 1

    def test_AddPoly(self):
        P = NdPoly({(1,0,0): 1, (0,1,0): 3.14, (0,0,1): -1})
        Q = NdPoly({(0,0,0): 1, (2,1,0): 0.42})
        R = P + Q
        assert R == NdPoly({(1,0,0): 1, (0,1,0): 3.14, (0,0,1): -1, (0,0,0): 1, (2,1,0): 0.42})

    def test_PolyValues(self):
        X,Y,Z = np.meshgrid(self.testdata, self.testdata, self.testdata)
        data = np.hstack((
            X.flatten()[:,np.newaxis],
            Y.flatten()[:,np.newaxis],
            Z.flatten()[:,np.newaxis],
            ))
        P = NdPoly({(1,0,0): 1, (0,1,0): 3.14, (0,0,1): -1})

    def test_PolyGoodSet(self):
        P = NdPoly({(1,2,3): 0.1, (3,0,0): 3.14})
        P[(1,2,4)] = 3
        assert P == NdPoly({(1,2,3): 0.1, (3,0,0): 3.14, (1,2,4): 3})

    def test_PolyBadSet(self):
        P = NdPoly({(1,2,3): 0.1, (3,0,0): 3.14})
        with pytest.raises(AssertionError):
            P[(1,2)] = 3

    def test_powersCoeffs(self):
        P = NdPoly({
            (1,0,0): 1,
            (0,1,0): 3.14,
            (2,0,1): 0.42,
            (1,2,3): 6.66,
            (0,0,1): -1,
            })
        assert list(P.powers()) == [(0,0,1), (0,1,0), (1,0,0), (1,2,3), (2,0,1)]
        assert list(P.coeffs()) == [-1, 3.14, 1, 6.66, 0.42]

    def test_Derivative(self):
        P = NdPoly({(1,2,3): 0.1, (3,0,0): 3.14})
        Q = P.derivative((1,0,1))
        exp_powers = [(0,2,2)]
        exp_coeffs = [0.3]
        assert all(p == pe for p,pe in zip(list(Q.powers()), exp_powers))
        assert all(abs(c-e) < 1E-10 for c,e in zip(list(Q.coeffs()), exp_coeffs))

    def test_DerivativeNull(self):
        P = NdPoly({(1,2,3): 0.1, (3,0,0): 3.14})
        Q = P.derivative((1,0,4))
        assert Q == NdPoly.zero(3)

    def test_expanded(self):
        P = NdPoly({(2,0): 1, (1,1): 2, (0,2): 0.5})
        P.x0 = [1., 1.]
        Q = P.expanded()
        assert math.isclose(Q([0,0]), 3.5)
        assert Q == NdPoly({
            (0,0): 3.5, (1,0): -4., (2,0): 1.0,
            (0,1): -3., (1,1): 2.0, (2,1): 0.0,
            (0,2): 0.5, (1,2): 0.0, (2,2): 0.0,
            })

class TestSymMat:
    testdata = np.linspace(0,1,4)[:,np.newaxis]
    
    def test_Symmetry(self):
        Ns = 2
        Nd = 3
        M = SymPolyMat(Nd,Ns)
        M[0,0] = NdPoly.zero(Nd)
        M[1,0] = NdPoly.one(Nd)
        M[1,1] = NdPoly({(1,2,3): 4})
        assert M[1,0] == M[0,1]

    def test_Empty(self):
        Ns = 3
        Nd = 4
        M = SymPolyMat(Ns, Nd)
        assert all([len(m.powers()) == 0 for m in M])
        assert all([m.Nd == Nd for m in M])

    def test_Zero(self):
        Ns = 3
        Nd = 4
        M = SymPolyMat.zero(Ns, Nd)
        assert all([list(m.powers()) == [m.zeroPower] for m in M])
        assert all([list(m.coeffs()) == [0] for m in M])
        assert all([m.Nd == Nd for m in M])

    def test_ZeroLike(self):
        Ns = 3
        Nd = 2
        A = SymPolyMat(Ns,Nd)
        A[0,0] = NdPoly({(0,1): 2, (3,4): 5})
        A[1,0] = NdPoly({(5,4): 3, (2,1): 0})
        A[1,1] = NdPoly({(1,1): 1, (2,2): 2})
        A[2,0] = NdPoly({(3,3): 3, (4,4): 4})
        A[2,1] = NdPoly({(3,1): 4, (4,1): 3})
        A[2,2] = NdPoly({(0,4): 2, (6,6): 6})
        B = SymPolyMat.zero_like(A)
        assert all([list(a.powers()) == list(b.powers()) for a,b in zip(A,B)])
        assert all([list(b.coeffs()) == [0 for _ in range(len(b))] for b in B])

    def test_Eye(self):
        Ns = 3
        Nd = 2
        I = SymPolyMat.eye(Ns,Nd)
        assert all([list(i.powers()) == [i.zeroPower] for i in I])
        assert all([list(I[i,j].coeffs()) == [0] for i in range(Ns) for j in range(i)])
        assert all([list(I[i,i].coeffs()) == [1] for i in range(Ns)])

    def test_EyeLike(self):
        Ns = 3
        Nd = 2
        A = SymPolyMat(Ns,Nd)
        A[0,0] = NdPoly({(0,1): 2, (3,4): 5})
        A[1,0] = NdPoly({(5,4): 3, (2,1): 0})
        A[1,1] = NdPoly({(1,1): 1, (2,2): 2})
        A[2,0] = NdPoly({(3,3): 3, (4,4): 4})
        A[2,1] = NdPoly({(3,1): 4, (4,1): 3})
        A[2,2] = NdPoly({(0,4): 2, (6,6): 6})
        B = SymPolyMat.eye_like(A)
        assert all([list(B[i,j].powers()) == list(A[i,j].powers()) for i in range(Ns) for j in range(i)])
        assert all([a_power in B[i,i] for i in range(Ns) for a_power in list(A[i,i].powers())])
        assert all([B[i,i][B[i,i].zeroPower] == 1 for i in range(Ns)])
     
    def test_Call(self):
        Ns = 5
        Nd = 3
        M = SymPolyMat.zero(Ns,Nd)
        P = NdPoly({(1,2,3): 0.1, (3,0,0): 3.14})
        M[0,0] = P
        M[1,1] = NdPoly({(1,0,0): 1, (0,0,1): 2})
        M[2,2] = P
        M[1,0] = NdPoly({(0,0,0): 1})
        M[0,2] = NdPoly({(0,0,0): 0.1})
        M[3,0] = NdPoly({(0,0,1): 6.66})
        X,Y,Z = np.meshgrid(self.testdata, self.testdata, self.testdata)
        data = np.hstack((
            X.flatten()[:,np.newaxis],
            Y.flatten()[:,np.newaxis],
            Z.flatten()[:,np.newaxis],
            ))
        Wx = M(data)
        assert Wx.shape == (len(self.testdata)**Nd, Ns, Ns)
        assert all([np.allclose(Wx[i,:,:], Wx[i,:,:].T) for i in range(len(self.testdata))])

    def test_shift(self):
       pass 

class TestDiabatizer:
    def test_LiF(self, pytestconfig):
        import ConstantsSI as SI
        import matplotlib.pyplot as plt
        from cycler import cycler
        from itertools import cycle
        E_Li_p = 1.8478136
        r0_LiF_gs = 1.563864

        # Load data
        filename = 'test-data/lif_mr_mscaspt2.csv'
        data = np.genfromtxt(filename,delimiter=',',skip_header=4)
        r = data[:,0]
        Npts = r.size
        en = data[:,5:]
        en = (en - en[-1,0])*SI.h2ev

        # Transform coordinate
        r_x = 6.5   # Intersection point
        stretch_factor = 1/r_x
        x = 1 - np.exp(-stretch_factor*(r-r_x))

        # Initial guess
        W = SymPolyMat(2,1)
        W[0,0] = NdPoly({(0,): 0.1, (1,): 1, (2,): -1, (3,): 1, (4,): 1, (5,): 1})
        W[1,1] = NdPoly({(0,): 0.1, (1,): -1, (2,): 1, (3,): 1, (4,): 1, (5,): 1, (6,): 1, (7,): 1})
        W[1,0] = NdPoly({(0,): 0.0})

        # Diabatize
        lif = Diabatizer(2,1, verbosity=2)
        lif.Wguess = W
        lif.addPoints(x,en,(0,1))
        lif.optimize()
        
        if pytestconfig.getoption("verbose") > 0:
            # Make plot using resulting parameters
            r_test = np.logspace(np.log10(2), np.log10(10), 100)
            x_test = 1 - np.exp(-stretch_factor*(r_test-r_x))
            # x_test = x_test[:,np.newaxis]
            yd1_test = lif.Wout[0,0](x_test)
            yd2_test = lif.Wout[1,1](x_test)
            yc_test = lif.Wout[1,0](x_test)
            ya1_test = adiabatic2(yd1_test, yd2_test, yc_test, -1)
            ya2_test = adiabatic2(yd1_test, yd2_test, yc_test, +1)
            
            fig, axs = plt.subplots(2,1)
            _, nplots = en.shape
            labels = [
                    r'$E_1$ XMSCASPT2',
                    r'$E_2$ XMSCASPT2',
                    r'$E_1$ fit',
                    r'$E_2$ fit',
                    r'$V_1$',
                    r'$V_2$',
                    r'$V_{12}$',
                    ]
            mark_cycle = cycler(marker=['+','x'])

            ax = axs[0]
            for n, mc in zip(range(nplots),cycle(mark_cycle)):
                ax.plot(x,en[:,n],linewidth=0.5, **mc)

            ax.plot(x_test, ya1_test)
            ax.plot(x_test, ya2_test)
            ax.plot(x_test, yd1_test)
            ax.plot(x_test, yd2_test)
            ax.plot(x_test, yc_test)
            ax.set_xlabel(r'$x$')
            ax.set_ylabel(r'$E$ / hartree')
            ax.set_title(r'\ce{LiF} avoided crossing : diabatization of XMS-CASPT2 potentials')
            ax.grid(True)

            ax = axs[1]
            for n, mc in zip(range(nplots),cycle(mark_cycle)):
                ax.plot(r,en[:,n],linewidth=0.5, **mc)

            ax.plot(r_test, ya1_test)
            ax.plot(r_test, ya2_test)
            ax.plot(r_test, yd1_test)
            ax.plot(r_test, yd2_test)
            ax.plot(r_test, yc_test)
            ax.set_xlabel(r'$r_{\ce{LiF}}$ / \AA')
            ax.set_ylabel(r'$E$ / hartree')
            ax.legend(labels)
            ax.grid(True)

            fig.set_size_inches((12,9))
            # plt.savefig('lif.pdf')
            plt.show()

    def plot_1d2s_testVSfit(self,x,Wt,Vt,W):
        import matplotlib.pyplot as plt
        # Reshape
        W11_t = Wt[:,0,0].reshape(x.shape)
        W22_t = Wt[:,1,1].reshape(x.shape)
        W21_t = np.abs(Wt[:,1,0].reshape(x.shape))
        V1_t = Vt[:,0].reshape(x.shape)
        V2_t = Vt[:,1].reshape(x.shape)

        W11 = W[:,0,0].reshape(x.shape)
        W22 = W[:,1,1].reshape(x.shape)
        W21 = np.abs(W[:,1,0].reshape(x.shape))
        V1 = adiabatic(W)[0][:,0].reshape(x.shape)
        V2 = adiabatic(W)[0][:,1].reshape(x.shape)

        # Calculate logarithms of abs(differences)
        dW11 = np.abs(W11-W11_t)
        dW22 = np.abs(W22-W22_t)
        dW21 = np.abs(W21-W21_t)
        dV1 = np.abs(V1-V1_t)
        dV2 = np.abs(V2-V2_t)
        dW11[dW11==0] = np.nan
        dW22[dW22==0] = np.nan
        dW21[dW21==0] = np.nan
        dV1[dV1==0] = np.nan
        dV2[dV2==0] = np.nan
       
        ldW11 = np.log10(dW11)
        ldW22 = np.log10(dW22)
        ldW21 = np.log10(dW21)
        ldV1  = np.log10(dV1)
        ldV2  = np.log10(dV2)

        # Plot
        cl = np.linspace(-1,1,21)
        fig, axs = plt.subplots(3,3)
        axs[0,0].plot(x,W11_t,color='b')      
        axs[0,0].plot(x,W22_t,color='r')      
        axs[0,1].plot(x,W21_t,color='magenta')
        axs[0,2].plot(x,V1_t, color='lime')    
        axs[0,2].plot(x,V2_t, color='orange')  

        axs[0,0].set_ylabel("$W_{ii}$ (orig)")
        axs[0,1].set_ylabel("$W_{12}$ (orig)")
        axs[0,2].set_ylabel("$V_{i}$ (orig)")

        axs[1,0].plot(x,W11,color='b')      
        axs[1,0].plot(x,W22,color='r')      
        axs[1,1].plot(x,W21,color='magenta')
        axs[1,2].plot(x,V1, color='lime')    
        axs[1,2].plot(x,V2, color='orange')  

        axs[1,0].set_ylabel("$W_{ii}$ (fit)")
        axs[1,1].set_ylabel("$W_{12}$ (fit)")
        axs[1,2].set_ylabel("$V_{i}$ (fit)")

        axs[2,0].plot(x,ldW11,color='b')
        axs[2,0].plot(x,ldW22,color='r')
        axs[2,1].plot(x,ldW21,color='magenta')
        axs[2,2].plot(x,ldV1, color='lime')
        axs[2,2].plot(x,ldV2, color='orange')
        axs[2,0].set_ylabel("$\log\Delta W_{ii}$")
        axs[2,1].set_ylabel("$\log\Delta W_{12}$")
        axs[2,2].set_ylabel("$\log\Delta V_{i}$")
        axs[2,0].set_xlabel("$x$")
        axs[2,1].set_xlabel("$x$")
        axs[2,2].set_xlabel("$x$")

        plt.show()

    def plot_2d2s_testVSfit(self,X,Y,Wt,Vt,W):
        import matplotlib.pyplot as plt
        # Reshape
        W11_t = Wt[:,0,0].reshape(X.shape)
        W22_t = Wt[:,1,1].reshape(X.shape)
        W21_t = np.abs(Wt[:,1,0].reshape(X.shape))
        V1_t = Vt[:,0].reshape(X.shape)
        V2_t = Vt[:,1].reshape(X.shape)

        W11 = W[:,0,0].reshape(X.shape)
        W22 = W[:,1,1].reshape(X.shape)
        W21 = np.abs(W[:,1,0].reshape(X.shape))
        V1 = adiabatic(W)[0][:,0].reshape(X.shape)
        V2 = adiabatic(W)[0][:,1].reshape(X.shape)

        # Calculate logarithms of abs(differences)
        dW11 = np.abs(W11-W11_t)
        dW22 = np.abs(W22-W22_t)
        dW21 = np.abs(W21-W21_t)
        dV1 = np.abs(V1-V1_t)
        dV2 = np.abs(V2-V2_t)
        dW11[dW11==0] = np.nan
        dW22[dW22==0] = np.nan
        dW21[dW21==0] = np.nan
        dV1[dV1==0] = np.nan
        dV2[dV2==0] = np.nan
       
        ldW11 = np.log10(dW11)
        ldW22 = np.log10(dW22)
        ldW21 = np.log10(dW21)
        ldV1  = np.log10(dV1)
        ldV2  = np.log10(dV2)

        # Plot
        cl = np.linspace(-1,1,21)
        fig, axs = plt.subplots(3,3)
        axs[0,0].contour(X,Y,W11_t,levels=cl,cmap='Blues_r')
        axs[0,0].contour(X,Y,W22_t,levels=cl,cmap='Reds_r')
        axs[0,0].set_ylabel("$y$")
        axs[0,1].contour(X,Y,W21_t,levels=21,cmap='BrBG')
        axs[0,2].contour(X,Y,V1_t,levels=cl,cmap='Greens')
        axs[0,2].contour(X,Y,V2_t,levels=cl,cmap='Oranges_r')

        axs[1,0].contour(X,Y,W11,levels=cl,cmap='Blues_r')
        axs[1,0].contour(X,Y,W22,levels=cl,cmap='Reds_r')
        axs[1,0].set_ylabel("$y$")
        axs[1,1].contour(X,Y,W21,levels=21,cmap='BrBG')
        axs[1,2].contour(X,Y,V1,levels=cl,cmap='Greens')
        axs[1,2].contour(X,Y,V2,levels=cl,cmap='Oranges_r')

        cs = axs[2,0].contour(X,Y,ldW11,levels=10,cmap='Blues_r')
        fig.colorbar(cs, ax=axs[2,0], location='right')
        cs = axs[2,0].contour(X,Y,ldW22,levels=10,cmap='Reds_r')
        fig.colorbar(cs, ax=axs[2,0], location='right')
        axs[2,0].set_xlabel("$x$")
        axs[2,0].set_ylabel("$y$")
        cs = axs[2,1].contour(X,Y,ldW21,levels=10,cmap='BrBG')
        fig.colorbar(cs, ax=axs[2,1], location='right')
        axs[2,1].set_xlabel("$x$")
        cs = axs[2,2].contour(X,Y,ldV1,levels=10,cmap='Greens_r')
        fig.colorbar(cs, ax=axs[2,2], location='right')
        cs = axs[2,2].contour(X,Y,ldV2,levels=10,cmap='Oranges_r')
        fig.colorbar(cs, ax=axs[2,2], location='right')
        axs[2,2].set_xlabel("$x$")

        ax1 = plt.figure().add_subplot(projection='3d')
        ax1.plot_wireframe(X,Y,W11_t,alpha=0.5,color='darkblue', lw=0.5)
        ax1.plot_surface(X,Y,W11,alpha=0.5,cmap='Blues_r')
        ax1.contour(X,Y,W11,alpha=0.5, levels=cl,colors='b')
        ax1.plot_wireframe(X,Y,W22_t,alpha=0.5,color='darkred', lw=0.5)
        ax1.plot_surface(X,Y,W22,alpha=0.5,cmap='Reds_r')
        ax1.contour(X,Y,W22,alpha=0.5, levels=cl,colors='r')
        ax1.set_zlim(ax1.get_zlim()[0], cl[-1])
        ax1.set_xlabel("$x$")
        ax1.set_ylabel("$y$")

        ax2 = plt.figure().add_subplot(projection='3d')
        ax2.plot_wireframe(X,Y,W21_t,alpha=0.5,color='k', lw=0.5)
        ax2.plot_surface(X,Y,W21,alpha=0.5,cmap='BrBG')
        ax2.contour(X,Y,W21,alpha=0.5, levels=21, colors='indigo')
        ax2.set_xlabel("$x$")
        ax2.set_ylabel("$y$")

        ax3 = plt.figure().add_subplot(projection='3d')
        ax3.plot_wireframe(X,Y,V1_t,alpha=0.5,color='darkgreen', lw=0.5)
        ax3.plot_surface(X,Y,V1,alpha=0.5,cmap='Greens')
        ax3.contour(X,Y,V1,alpha=0.5, levels=cl, colors='g')
        ax3.plot_wireframe(X,Y,V2_t,alpha=0.5,color='darkorange', lw=0.5)
        ax3.plot_surface(X,Y,V2,alpha=0.5,cmap='Oranges_r')
        ax3.contour(X,Y,V2,alpha=0.5, levels=cl, colors='orange')
        ax3.set_zlim(ax3.get_zlim()[0], cl[-1])
        ax3.set_xlabel("$x$")
        ax3.set_ylabel("$y$")
        plt.show()

    def plot_2d3s_testVSfit(self,X,Y,Wt,Vt,W):
        import matplotlib.pyplot as plt
        # Reshape
        W11_t = Wt[:,0,0].reshape(X.shape)
        W22_t = Wt[:,1,1].reshape(X.shape)
        W33_t = Wt[:,2,2].reshape(X.shape)
        W21_t = np.abs(Wt[:,1,0].reshape(X.shape))
        W31_t = np.abs(Wt[:,2,0].reshape(X.shape))
        W32_t = np.abs(Wt[:,2,1].reshape(X.shape))
        V1_t = Vt[:,0].reshape(X.shape)
        V2_t = Vt[:,1].reshape(X.shape)
        V3_t = Vt[:,2].reshape(X.shape)

        W11 = W[:,0,0].reshape(X.shape)
        W22 = W[:,1,1].reshape(X.shape)
        W33 = W[:,2,2].reshape(X.shape)
        W21 = np.abs(W[:,1,0].reshape(X.shape))
        W31 = np.abs(W[:,2,0].reshape(X.shape))
        W32 = np.abs(W[:,2,1].reshape(X.shape))
        V1 = adiabatic(W)[0][:,0].reshape(X.shape)
        V2 = adiabatic(W)[0][:,1].reshape(X.shape)
        V3 = adiabatic(W)[0][:,2].reshape(X.shape)

        # Calculate logarithms of abs(differences)
        dW11 = np.abs(W11-W11_t)
        dW22 = np.abs(W22-W22_t)
        dW33 = np.abs(W33-W33_t)
        dW21 = np.abs(W21-W21_t)
        dW31 = np.abs(W31-W31_t)
        dW32 = np.abs(W32-W32_t)
        dV1 = np.abs(V1-V1_t)
        dV2 = np.abs(V2-V2_t)
        dV3 = np.abs(V3-V3_t)
        dW11[dW11==0] = np.nan
        dW22[dW22==0] = np.nan
        dW33[dW33==0] = np.nan
        dW21[dW21==0] = np.nan
        dW31[dW31==0] = np.nan
        dW32[dW32==0] = np.nan
        dV1[dV1==0] = np.nan
        dV2[dV2==0] = np.nan
        dV3[dV3==0] = np.nan
       
        ldW11 = np.log10(dW11)
        ldW22 = np.log10(dW22)
        ldW33 = np.log10(dW33)
        ldW21 = np.log10(dW21)
        ldW31 = np.log10(dW31)
        ldW32 = np.log10(dW32)
        ldV1  = np.log10(dV1)
        ldV2  = np.log10(dV2)
        ldV3  = np.log10(dV3)

        # Plot
        cmapW11='Blues_r'
        cmapW22='Reds_r'
        cmapW33='Greens_r'
        cmapW21='bwr'
        cmapW31='BrBG'
        cmapW32='PiYG'
        cmapV1='Purples'
        cmapV2='plasma'
        cmapV3='Oranges_r'

        # Collect
        Wii_t = [W11_t, W22_t, W33_t]
        Wij_t = [W21_t, W31_t, W32_t]
        V_t = [V1_t, V2_t, V3_t]
        pots_t = Wii_t + Wij_t + V_t
        Wii = [W11, W22, W33]
        Wij = [W21, W31, W32]
        V = [V1, V2, V3]
        pots = Wii + Wij + V
        ldWii = [ldW11, ldW22, ldW33]
        ldWij = [ldW21, ldW31, ldW32]
        ldV = [ldV1, ldV2, ldV3]
        log_diffs = ldWii + ldWij + ldV
        cmapsWii = [cmapW11, cmapW22, cmapW33]
        cmapsWij = [cmapW21, cmapW31, cmapW32]
        cmapsV = [cmapV1, cmapV2, cmapV3]
        cmaps = cmapsWii + cmapsWij + cmapsV

        cl = np.linspace(-1,1.5,21)
        fig, axs = plt.subplots(3,3)
        axs[0,0].contour(X,Y,W11_t,levels=cl,cmap=cmapW11)
        axs[0,0].contour(X,Y,W22_t,levels=cl,cmap=cmapW22)
        axs[0,0].contour(X,Y,W33_t,levels=cl,cmap=cmapW33)
        axs[0,1].contour(X,Y,W21_t,levels=21,cmap=cmapW21)
        axs[0,1].contour(X,Y,W31_t,levels=21,cmap=cmapW31)
        axs[0,1].contour(X,Y,W32_t,levels=21,cmap=cmapW32)
        axs[0,2].contour(X,Y,V1_t,levels=cl,cmap=cmapV1)
        axs[0,2].contour(X,Y,V2_t,levels=cl,cmap=cmapV2)
        axs[0,2].contour(X,Y,V3_t,levels=cl,cmap=cmapV3)

        axs[1,0].contour(X,Y,W11,levels=cl,cmap=cmapW11)
        axs[1,0].contour(X,Y,W22,levels=cl,cmap=cmapW22)
        axs[1,0].contour(X,Y,W33,levels=cl,cmap=cmapW33)
        axs[1,1].contour(X,Y,W21,levels=21,cmap=cmapW21)
        axs[1,1].contour(X,Y,W31,levels=21,cmap=cmapW31)
        axs[1,1].contour(X,Y,W32,levels=21,cmap=cmapW32)
        axs[1,2].contour(X,Y,V1,levels=cl,cmap=cmapV1)
        axs[1,2].contour(X,Y,V2,levels=cl,cmap=cmapV2)
        axs[1,2].contour(X,Y,V3,levels=cl,cmap=cmapV3)

        for n, pair in enumerate(zip(log_diffs, cmaps)):
            ld, cm = pair
            if not np.all(np.isnan(ld)):
                cs = axs[2,n//3].contour(X,Y,ld,levels=10,cmap=cm)
                fig.colorbar(cs, ax=axs[2,n//3], location='right')

        axs[0,0].set_ylabel("$y$")
        axs[1,0].set_ylabel("$y$")
        axs[2,0].set_xlabel("$x$")
        axs[2,0].set_ylabel("$y$")
        axs[2,1].set_xlabel("$x$")
        axs[2,2].set_xlabel("$x$")

        ax1 = plt.figure().add_subplot(projection='3d')
        ax1.plot_wireframe(X,Y,W11_t,alpha=0.5,color='darkblue', lw=0.5)
        ax1.plot_surface(X,Y,W11,alpha=0.5,cmap=cmapW11)
        ax1.contour(X,Y,W11,alpha=0.5, levels=cl,colors='b')
        ax1.plot_wireframe(X,Y,W22_t,alpha=0.5,color='darkred', lw=0.5)
        ax1.plot_surface(X,Y,W22,alpha=0.5,cmap=cmapW22)
        ax1.contour(X,Y,W22,alpha=0.5, levels=cl,colors='r')
        ax1.plot_wireframe(X,Y,W33_t,alpha=0.5,color='darkgreen', lw=0.5)
        ax1.plot_surface(X,Y,W33,alpha=0.5,cmap=cmapW33)
        ax1.contour(X,Y,W33,alpha=0.5, levels=cl,colors='g')
        ax1.set_zlim(ax1.get_zlim()[0], cl[-1])
        ax1.set_xlabel("$x$")
        ax1.set_ylabel("$y$")

        ax2 = plt.figure().add_subplot(projection='3d')
        ax2.plot_wireframe(X,Y,W21_t,alpha=0.5,color='k', lw=0.5)
        ax2.plot_surface(X,Y,W21,alpha=0.5,cmap=cmapW21)
        ax2.contour(X,Y,W21,alpha=0.5, levels=21, colors='purple')
        ax2.plot_wireframe(X,Y,W31_t,alpha=0.5,color='k', lw=0.5)
        ax2.plot_surface(X,Y,W31,alpha=0.5,cmap=cmapW31)
        ax2.contour(X,Y,W31,alpha=0.5, levels=21, colors='teal')
        ax2.plot_wireframe(X,Y,W32_t,alpha=0.5,color='k', lw=0.5)
        ax2.plot_surface(X,Y,W32,alpha=0.5,cmap=cmapW32)
        ax2.contour(X,Y,W32,alpha=0.5, levels=21, colors='olive')
        ax2.set_xlabel("$x$")
        ax2.set_ylabel("$y$")

        ax3 = plt.figure().add_subplot(projection='3d')
        ax3.plot_wireframe(X,Y,V1_t,alpha=0.5,color='indigo', lw=0.5)
        ax3.plot_surface(X,Y,V1,alpha=0.5,cmap=cmapV1)
        ax3.contour(X,Y,V1,alpha=0.5, levels=cl, colors='purple')
        ax3.plot_wireframe(X,Y,V2_t,alpha=0.5,color='crimson', lw=0.5)
        ax3.plot_surface(X,Y,V2,alpha=0.5,cmap=cmapV2)
        ax3.contour(X,Y,V2,alpha=0.5, levels=cl, colors='deeppink')
        ax3.plot_wireframe(X,Y,V3_t,alpha=0.5,color='darkorange', lw=0.5)
        ax3.plot_surface(X,Y,V3,alpha=0.5,cmap=cmapV3)
        ax3.contour(X,Y,V3,alpha=0.5, levels=cl, colors='orange')
        ax3.set_zlim(ax3.get_zlim()[0], cl[-1])
        ax3.set_xlabel("$x$")
        ax3.set_ylabel("$y$")
        plt.show()

    def W_JTCI_2d2s(self):
        dx1 = 1
        dx2 = -1
        W = SymPolyMat(2,2)
        W[0,0] = NdPoly({(2,0): 0.5, (0,2): 0.5,})
        W[0,0][(0,0)] = W[0,0][(2,0)]*dx1**2 #- W[0,0][(1,0)]*dx1 
        W[0,0][(1,0)] = -2*W[0,0][(2,0)]*dx1
        W[1,1] = copy(W[0,0])
        W[1,1][(0,0)] = W[0,0][(2,0)]*dx2**2 #- W[0,0][(1,0)]*dx2 
        W[1,1][(1,0)] = -2*W[0,0][(2,0)]*dx2
        W[0,1] = NdPoly({(0,1): 5E-1})
        return W

    def W_infinityCI_2d2s(self):
        W = SymPolyMat(2,2)
        W[0,0] = NdPoly({
            (0,0): 0,
            (1,0): -1,
            (0,1): 0.2,
            (2,0): 0.3,
            (1,1): 0.1,
            (0,2): 0,
            (3,0): 0.1,
            (2,1): 0,
            (1,2): 0,
            (0,3): 0,
        })
        W[1,1] = NdPoly({
            (0,0): 0,
            (1,0): 0.5,
            (0,1): 0,
            (2,0): 1.5,
            (1,1): 0,
            (0,2): -0.2,
        })
        W[0,1] = NdPoly({(0,1): 5E-2})
        return W

    def W_ho_2d3s(self):
        dx1 = -1
        dx2 = 0
        dx3 = 1
        W = SymPolyMat.zero(3,2)
        W[0,0] = NdPoly({(2,0): 0.1, (0,2): 0.1,})
        W[1,1] = NdPoly({(0,0): 0.5, (1,0): -0.5})
        W[2,2] = NdPoly({(2,0): 0.1, (0,2): 0.1,})
        W[0,0][(0,0)] = W[0,0][(2,0)]*dx1**2 
        W[0,0][(1,0)] = -2*W[0,0][(2,0)]*dx1
        W[2,2][(0,0)] = W[0,0][(2,0)]*dx2**2 + 1
        W[2,2][(1,0)] = -2*W[0,0][(2,0)]*dx2
        W[0,1] = NdPoly({(0,1): 5E-1})
        W[2,1] = NdPoly({(0,1): 1E-1})
        W[2,0] = NdPoly({(0,1): 2E-1})
        return W

    def test_2d2s_infinity(self, pytestconfig):
        """ Two states, two dimensions.
        1: Construct an arbitrary diabatic potential matrix W_test
        2: Diagonalize it in order to get the adiabatic PESs
        3: Fit a new W_fit to the adiabatic data and check if
           W_fit == W_test
        """

        # 0: Prepare coordinates
        x1, x2, y1, y2 = (-1,1,0,1)
        Nx, Ny = (21,21)
        x = np.linspace(x1,x2,Nx)
        y = np.linspace(y1,y2,Ny)

        X, Y = np.mgrid[x1:x2:Nx*1j, y1:y2:Ny*1j]
        x0, y0 = (0,1)
        X = X - x0
        Y = Y - y0

        x_data = np.hstack((
            X.flatten()[:,np.newaxis],
            Y.flatten()[:,np.newaxis],
            ))

        # 1: Make target diabatic surfaces
        W_test = self.W_infinityCI_2d2s()
        W_test_x = W_test(x_data)

        # 2: Make test adiabatic surfaces
        W11_t = W_test_x[:,0,0]
        W22_t = W_test_x[:,1,1]
        W21_t = W_test_x[:,1,0]

        m = 0.5*(W11_t + W22_t)
        d = 0.5*(W11_t - W22_t)
        c = W21_t

        V_t = np.zeros((x_data.shape[0], 2))
        V_t[:,0] = m - np.sqrt(d**2 + c**2)
        V_t[:,1] = m + np.sqrt(d**2 + c**2)
        
        # 3: Fit diabatize test adiabatic surfaces
        W_guess = SymPolyMat.zero_like(W_test)

        test2d2s = Diabatizer(2,2,W_guess)
        test2d2s.addPoints(x_data, V_t, (0,1))
        test2d2s.optimize()
        W = test2d2s.Wout
        for w, wt in zip(W,W_test):
            for c, ct in zip(w.coeffs(), wt.coeffs()):
                assert abs(c-ct) < 1E-10

        # Show the result if verbose test
        if pytestconfig.getoption("verbose") > 0:
            Wx = W(x_data)
            self.plot_2d2s_testVSfit(X,Y,W_test_x,V_t,Wx)

    def test_2d3s(self, pytestconfig):
        """ Three states, two dimensions.
        1: Construct an arbitrary diabatic potential matrix W_test
        2: Diagonalize it in order to get the adiabatic PESs
        3: Fit a new W_fit to the adiabatic data and check if
           W_fit == W_test
        """

        # 0: Prepare coordinates
        x1, x2, y1, y2 = (-2,2,-1,1)
        Nx, Ny = (21,21)
        x = np.linspace(x1,x2,Nx)
        y = np.linspace(y1,y2,Ny)

        X, Y = np.mgrid[x1:x2:Nx*1j, y1:y2:Ny*1j]
        x0, y0 = (0,0)
        X = X - x0
        Y = Y - y0

        x_data = np.hstack((
            X.flatten()[:,np.newaxis],
            Y.flatten()[:,np.newaxis],
            ))

        # 1: Make target diabatic surfaces
        W_test = self.W_ho_2d3s()
        W_test_x = W_test(x_data)

        # 2: Make test adiabatic surfaces
        V_t = adiabatic(W_test_x)[0]

        # 3: Fit diabatize test adiabatic surfaces
        W_guess = SymPolyMat.zero_like(W_test)
        W_guess[0,0][(0,0)] = 0
        W_guess[1,1][(1,0)] = -0.5
        W_guess[2,2][(0,0)] = 1

        test2d3s = Diabatizer(3,2,W_guess)
        test2d3s.addPoints(x_data, V_t, (0,1,2))
        test2d3s.optimize()
        W = test2d3s.Wout

        for i in range(3):
            for c, ct in zip(W[i,i].coeffs(), W_test[i,i].coeffs()):
                assert abs(c-ct) < 1E-10
            for j in range(i):
                for c, ct in zip(W[i,j].coeffs(), W_test[i,j].coeffs()):
                    assert abs(abs(c)-abs(ct)) < 1E-10

        # Show the result if verbose test
        if pytestconfig.getoption("verbose") > 0:
            Wx = W(x_data)
            self.plot_2d3s_testVSfit(X,Y,W_test_x,V_t,Wx)

    def test_1d2s_crossing(self, pytestconfig):
        """ Two states, one dimension, two distinct avoided crossings
        1: Construct two arbitrary diabatic potential matrices
        2: Diagonalize both in order to get the adiabatic PESs
        3: Fit a unique W_fit to the adiabatic data and verify
           W_fit != W_test
        4: Do the fit with two matrices and check if the couplings are
           now correct
        """
        Ns = 2
        Nd = 1

        # 0: Prepare coordinates
        xmin, xmax = (-2,2)
        Nx = 21
        x = np.linspace(xmin,xmax,Nx)
        left = range(Nx//2)
        right = range(Nx//2, Nx)
        x01, x02 = (-1,1)

        # 1: Make target diabatic surfaces
        Wt1 = SymPolyMat(2,1)
        Wt1[0,0] = NdPoly.zero(1)
        Wt1[1,0] = NdPoly.one(1)*0.5
        Wt1[1,1] = NdPoly({(1,):-1})
        Wt1.x0 = -1
        Wt2 = deepcopy(Wt1)
        Wt2[1,0] *= 0.1
        Wt2[1,1] *=  -1
        Wt2.x0 = 1

        W_test_x = np.zeros((Nx, Ns, Ns))
        W_test_x[left,:,:] = Wt1(x[left])
        W_test_x[right,:,:] = Wt2(x[right])

        # 2: Make test adiabatic surfaces
        V_t = adiabatic(W_test_x)[0]

        # 3: Fit diabatize test adiabatic surfaces
        W_guess = SymPolyMat.zero_like(Wt1)
        W_guess[0,0][(0,)] = 0
        W_guess[1,0][(0,)] = 0.2
        W_guess[1,1][(0,)] = 1
        W_guess[1,1][(2,)] = 1

        diab = Diabatizer(2,1,W_guess)
        diab.addPoints(x, V_t, (0,1))
        diab.optimize()
        W = diab.Wout

        # for i in range(3):
        #     for c, ct in zip(W[i,i].coeffs(), W_test[i,i].coeffs()):
        #         assert abs(c-ct) < 1E-10
        #     for j in range(i):
        #         for c, ct in zip(W[i,j].coeffs(), W_test[i,j].coeffs()):
        #             assert abs(abs(c)-abs(ct)) < 1E-10

        # Show the result if verbose test
        if pytestconfig.getoption("verbose") > 0:
            Wx = W(x)
            self.plot_1d2s_testVSfit(x,W_test_x,V_t,Wx)

    def test_3d3s(self):
        pass
    
def main(argv) -> int:
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
