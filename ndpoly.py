from __future__ import annotations
from copy import deepcopy
from typing import Union
from collections import UserDict
import math
import numpy as np
from .jsonutils import _str2tuple

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
            raise ValueError(
                "Cannot make empty polynomial with unknown"
                "dimensionality. Use NdPoly.empty() instead."
                )

        # Aliases for keys and values
        self.powers = self.keys
        self.coeffs = self.values

        self._Nd = None
        self._x0 = None
        super().__init__(data)  # Sets Nd and degree
        self._zero_power = tuple(0 for _ in range(self._Nd))
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
    def zero_power(self):
        """ Return the tuple of powers of constant term, i.e. (0, ..., 0) """
        return self._zero_power

    @property
    def x0(self):
        return self._x0

    @x0.setter
    def x0(self, x0):
        """ Transform the polynomial P(x) into Q(x) = P(x-x0)
        by setting the origin point without changing coefficients. """
        if self._Nd is None:
            raise RuntimeError("Cannot set origin in a polynomial of unknown dimensionality.")
        x0 = np.array(x0).flatten()
        if x0.size != self._Nd:
            raise ValueError(f"{x0} should be a point in {self._Nd} dimensions.")
        self._x0 = x0

    def max_partial_degrees(self):
        return [max(powers[i] for powers in self.powers())
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

    def set_zero_const(self):
        """ Set constant term to zero.

        This has the effect of adding self._zeroPower to the dictionnary
        keys if it was not previously there. Equivalent to self += 0
        """
        self[self._zero_power] = 0

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
            raise ValueError(
                "Invalid differentiation order of length "
                f"{len(orders)}, expecting {self._Nd}")

        D = NdPoly.empty(self._Nd)
        for powers, coeff in self.items():  # Differentiate each monomial
            new_powers = list(powers)
            for dof in range(self._Nd):
                new_powers[dof] -= orders[dof]
            # If monomial is annihilated, go to next
            if any(p < 0 for p in new_powers):
                continue

            # The monomial survived, compute corresponding coefficient
            new_coeff = coeff
            for dof in range(self._Nd):
                for k in range(new_powers[dof]+1, new_powers[dof] + orders[dof] + 1):
                    new_coeff *= k
            D[tuple(new_powers)] = new_coeff

        if len(D.powers()) == 0:
            return NdPoly.zero(self._Nd)

        return D

    def remove(self, key):
        del self[key]

    def __setitem__(self, powers, coeff):
        if self._Nd is None:
            if len(powers) == 0 :
                return
            # If this is the first non-zero-dim monomial to set,
            # it defines the dimension of the polynomial
            self._Nd = len(powers)

        if len(powers) != self._Nd :
            raise ValueError(f"Inappropriate powers {powers}. Expected {self._Nd} integers.")

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
            for k, p in enumerate(powers):
                monomial *= (x[:,k]-self._x0[k])**p
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

        if not self:
            return f"<Empty {self.__class__} of dimension {self._Nd} at {hex(id(self))}.>"

        s = ""

        first = True
        for powers, coeff in self.items():
            if coeff == 0:  # Ignore terms with 0 coeff
                continue

            cstr = ""
            # Write only non "1" coefficients except constant term
            if coeff != 1 or powers == self.zero_power :
                cstr = f"{coeff} "

            # + sign and coefficients
            if first :      # No + sign for the first term
                s += cstr
                first = False
            else:
                s += " + " + cstr

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
            s += "   |   [X0"
            for d in range(1,self._Nd):
                s += f" X{d}"
            s += f"] = {self._x0}"

        if self._Nd < 8:    # Use x,y,z,... instead of x1, x2, x3 if Nd < 8
            xyz = "xyztuvw"
            for d in range(self._Nd):
                s = s.replace(f"x{d}", xyz[d])
                s = s.replace(f"X{d}", xyz.upper()[d])

        return s

    def __add__(self, other: Union[int, float, NdPoly]):
        result = deepcopy(self)
        if isinstance(other, self.__class__):
            assert np.all(self._x0 == other.x0), "Origins of added polynomials do not match."
            for powers, coeff in other.items():
                if powers in result:
                    result[powers] += coeff
                else:
                    result[powers] = coeff
            return result

        if isinstance(other, (float, int)):
            # Convert number to polynomial, then call __add__ recursively
            other = NdPoly({self.zero_power: other})
            return self + other

        raise TypeError(f"Cannot add {self} and {other}.")

    def __mul__(self, other: Union[int, float, NdPoly]):
        # result = deepcopy(self)
        if isinstance(other, self.__class__):
            # assert np.all(self._x0 == other.x0), "Origins of added polynomials do not match."
            raise NotImplementedError("Product between NdPoly's not yet implemented")

        if isinstance(other, (float, int)):
            for powers in self.powers():
                self[powers] *= other
            return self

        raise TypeError(f"Cannot multiply {self} and {other}.")

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
        powers = tuple(0 for _ in range(Nd))
        P = cls.zero(Nd)
        del P[powers]
        return P

    @classmethod
    def zero(cls, Nd):
        """ Return a Nd-dimensional polynomial with zero constant term only """
        powers = tuple(0 for _ in range(Nd))
        return cls({powers: 0})

    @classmethod
    def one(cls, Nd):
        """ Return a Nd-dimensional polynomial with unit constant term only """
        powers = tuple(0 for _ in range(Nd))
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
