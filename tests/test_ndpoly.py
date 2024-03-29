import numpy as np
import math
from copy import copy, deepcopy
import pytest
from ..ndpoly import NdPoly

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
        assert E.zero_power == (0,0,0)
        assert np.all(E.x0 == np.array([0,0,0]))
        assert E([1.,2.,3.]) == 0
        assert np.all(E(np.random.rand(10,3)) == np.zeros((10,)))
        assert E.degree == -1
        assert E.def_degree == -1

    def test_Zero(self):
        P = NdPoly.zero(3)
        X = np.repeat(self.testdata, 3, 1)
        assert P.zero_power == (0,0,0)
        assert list(P.powers()) == [P.zero_power]
        assert P[P.zero_power] == 0
        assert np.all(P(X) == 0)
        assert P.degree == -1
        assert P.def_degree == 0

    def test_One(self):
        P = NdPoly.one(3)
        X = np.repeat(self.testdata, 3, 1)
        assert P.zero_power == (0,0,0)
        assert list(P.powers()) == [P.zero_power]
        assert P[P.zero_power] == 1
        assert np.all(P(X) == 1)
        assert P.degree == 0
        assert P.def_degree == 0

    def test_ZeroLike(self):
        P = NdPoly({(1,0,0): 1, (0,1,0): 3.14, (0,0,1): -1})
        Q = NdPoly.zero_like(P)
        assert all([ p == q for p, q in zip(list(P.powers()), list(Q.powers()))])
        assert all([ c == 0 for c in list(Q.coeffs())])
        assert P.degree == 1
        assert P.def_degree == 1
        assert Q.degree == -1
        assert Q.def_degree == 1

    def test_OneLike(self):
        P = NdPoly({(0,0,0): 6.66, (1,0,0): 1, (0,1,0): 3.14, (0,0,1): -1})
        Q = NdPoly.one_like(P)
        assert all([ p == q for p, q in zip(list(P.powers()), list(Q.powers()))])
        assert Q[(0,0,0)] == 1
        Q[(0,0,0)] = 0
        assert all([ c == 0 for c in list(Q.coeffs())])
        assert P.degree == 1
        assert P.def_degree == 1
        assert Q.degree == -1
        assert Q.def_degree == 1

    def test_AddConstant(self):
        P = NdPoly({(1,0,0): 1, (0,1,0): 3.14, (0,0,1): -1})
        Q = P + 1.23
        assert Q[Q.zero_power] == 1.23
        Q += 1
        assert Q[Q.zero_power] == 2.23
        P += 1
        assert P[P.zero_power] == 1
        assert P.degree == 1
        assert P.def_degree == 1
        assert Q.degree == 1
        assert Q.def_degree == 1

    def test_AddPoly(self):
        P = NdPoly({(1,0,0): 1, (0,1,0): 3.14, (0,0,1): -1})
        Q = NdPoly({(0,0,0): 1, (2,1,0): 0.42})
        R = P + Q
        assert R == NdPoly({(1,0,0): 1, (0,1,0): 3.14, (0,0,1): -1, (0,0,0): 1, (2,1,0): 0.42})
        assert P.degree == 1
        assert P.def_degree == 1
        assert Q.degree == 3
        assert Q.def_degree == 3
        assert R.degree == 3
        assert R.def_degree == 3

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
        with pytest.raises(ValueError):
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
        P = NdPoly({(1,2,3): 0.1, (3,0,0): 3.14, (0,0,6): 0})
        Q = P.derivative((1,0,1))
        R = P.derivative((0,0,4))
        exp_powers = [(0,2,2)]
        exp_coeffs = [0.3]
        assert all(p == pe for p,pe in zip(list(Q.powers()), exp_powers))
        assert all(abs(c-e) < 1E-10 for c,e in zip(list(Q.coeffs()), exp_coeffs))
        assert P.degree == 6
        assert P.def_degree == 6
        assert Q.degree == 4
        assert Q.def_degree == 4
        assert R.degree == -1
        assert R.def_degree == 2

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

    def test_maxdegree(self):
        P = NdPoly.zero_maxdegree(3, 4)
        powers = [
            (0, 0, 0),
            (0, 0, 1), (0, 1, 0),
            (1, 0, 0), 
            (0, 0, 2), (0, 1, 1), (0, 2, 0),
            (1, 0, 1), (1, 1, 0),
            (2, 0, 0), 
            (0, 0, 3), (0, 1, 2), (0, 2, 1), (0, 3, 0), 
            (1, 0, 2), (1, 1, 1), (1, 2, 0),
            (2, 0, 1), (2, 1, 0),
            (3, 0, 0), 
            (0, 0, 4), (0, 1, 3), (0, 2, 2), (0, 3, 1), (0, 4, 0),
            (1, 0, 3), (1, 1, 2), (1, 2, 1), (1, 3, 0),
            (2, 0, 2), (2, 1, 1), (2, 2, 0),
            (3, 0, 1), (3, 1, 0), 
            (4, 0, 0)]
        assert all([p in P for p in powers])
        assert P.degree == -1
        assert P.def_degree == 4

    def test_maxdegree_limited(self):
        P = NdPoly.fill_maxdegree(3, 4, fill=0.1, max_pdeg=(3,2,None))
        powers = [
            (0, 0, 0),
            (0, 0, 1), (0, 1, 0),
            (1, 0, 0), 
            (0, 0, 2), (0, 1, 1), (0, 2, 0),
            (1, 0, 1), (1, 1, 0),
            (2, 0, 0), 
            (0, 0, 3), (0, 1, 2), (0, 2, 1), 
            (1, 0, 2), (1, 1, 1), (1, 2, 0),
            (2, 0, 1), (2, 1, 0),
            (3, 0, 0), 
            (0, 0, 4), (0, 1, 3), (0, 2, 2), 
            (1, 0, 3), (1, 1, 2), (1, 2, 1), 
            (2, 0, 2), (2, 1, 1), (2, 2, 0),
            (3, 0, 1), (3, 1, 0), 
            ]
        excluded_powers = [
            (0, 3, 0), 
            (0, 3, 1), (0, 4, 0),
            (1, 3, 0),
            (4, 0, 0)]
        assert all([coeff == 0.1 for coeff in P.coeffs()])
        assert all([p in P for p in powers])
        assert all([p not in P for p in excluded_powers])
        assert P.degree == 4
        assert P.def_degree == 4

    def test_grow(self):
        P = NdPoly({(1,0,1): 0.1, (3,0,0): 3.14})
        P.grow_degree(4)
        powers = [
            (0, 0, 0),
            (0, 0, 1), (0, 1, 0),
            (1, 0, 0), 
            (0, 0, 2), (0, 1, 1), (0, 2, 0),
            (1, 0, 1), (1, 1, 0),
            (2, 0, 0), 
            (0, 0, 3), (0, 1, 2), (0, 2, 1), (0, 3, 0), 
            (1, 0, 2), (1, 1, 1), (1, 2, 0),
            (2, 0, 1), (2, 1, 0),
            (3, 0, 0), 
            (0, 0, 4), (0, 1, 3), (0, 2, 2), (0, 3, 1), (0, 4, 0),
            (1, 0, 3), (1, 1, 2), (1, 2, 1), (1, 3, 0),
            (2, 0, 2), (2, 1, 1), (2, 2, 0),
            (3, 0, 1), (3, 1, 0), 
            (4, 0, 0)]
        powers_diff = copy(powers)
        powers_diff.remove((1,0,1))
        powers_diff.remove((3,0,0))
        assert all([p in P for p in powers])
        assert P[(1,0,1)] == 0.1
        assert P[(3,0,0)] == 3.14
        assert all([P[p] == 0 for p in powers_diff])
        assert P.degree == 3
        assert P.def_degree == 4

    def test_degree(self):
        P = NdPoly({(1,2,3): 4, (0,1,2): 3, (3,0,2): 3, (0,4,0): 3, (2,2,2): 3, (6,6,6): 0})
        assert P.degree == 6
        assert P.def_degree == 18

    def test_coeffs_array(self):
        P = NdPoly({(1,2,3): 1, (0,1,2): 2, (3,0,2): 3, (0,4,0): 4, (2,2,2): 5, (6,6,6): 0})
        assert np.all(P.coeffs_to_array() == np.array([2,4,1,5,3,0]))

    def test_powers_list(self):
        P = NdPoly({(1,2,3): 1, (0,1,2): 2, (3,0,2): 3, (0,4,0): 4, (2,2,2): 5, (6,6,6): 0})
        assert all([a == b for a,b in zip(
            P.powers_to_list(),
            [(0,1,2),(0,4,0),(1,2,3),(2,2,2),(3,0,2),(6,6,6)]
            )
        ])

    def test_reconstruct(self):
        P = NdPoly({(1,2,3): 1, (0,1,2): 2, (3,0,2): 3, (0,4,0): 4, (2,2,2): 5, (6,6,6): 0})
        Q = NdPoly({powers: coeff for powers, coeff in zip(P.powers_to_list(), P.coeffs_to_array())})
        assert P == Q

