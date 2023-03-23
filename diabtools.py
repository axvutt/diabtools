from __future__ import annotations
import sys
from copy import *
from typing import List, Tuple, Dict, Union
from collections import UserDict
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
        self._Nd = None
        super().__init__(data)
        self._zeroPower = tuple([0 for _ in range(self._Nd)])

        # Aliases for keys and values
        self.powers = self.keys
        self.coeffs = self.values

    @property
    def Nd(self):
        """ Number of variables of the polynomial """
        return self._Nd

    @property
    def zeroPower(self):
        """ Return the tuple of powers of constant term, i.e. (0, ..., 0) """
        return self._zeroPower

    def setZeroConst(self):
        """ Set constant term to zero.

        This has the effect of adding self._zeroPower to the dictionnary
        keys if it was not previously there. Equivalent to self += 0
        """
        self[self._zeroPower] = 0

    def __setitem__(self, powers, coeff):
        if self._Nd is None and len(powers) != 0 :
            self._Nd = len(powers)
        else:
            assert len(powers) == self._Nd, f"Inappropriate powers {powers}. Expected {self._Nd} integers."
        super().__setitem__(powers, coeff)

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
        for powers, coeff in self.items():
            monomial = 1
            for k in range(len(powers)):
                monomial *= x[:,k]**powers[k]
            P += coeff * monomial
        return P
    
    def __str__(self):
        """ Explicit representation of the polynomial in terms of its variables """

        if self._Nd is None:
            return ""

        s = ""
        first = True
        for powers, coeff in self.items():
            if coeff == 0:  # Ignore terms with 0 coeff
                continue
            if first :      # No + sign for the first term
                s += f"{coeff} "
                first = False
            else:
                s += f" + {coeff} "
            for d, p in enumerate(powers):
                if p == 0 : # Do not show variables to the 0th power
                    continue
                if p == 1 : # Do not show ^1
                    s += f"x{d}"
                else :
                    s += f"x{d}^{p}"
        
        if self._Nd < 8:    # Use x,y,z,... instead of x1, x2, x3 if Nd < 8
            xyz = "xyztuvw"
            for d in range(self._Nd):
                s = s.replace(f"x{d}", xyz[d])

        return s

    def __add__(self, other: Union[int,float, self.__class__]):
        result = copy(self)
        if isinstance(other, self.__class__):
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
        self._polys = [ NdPoly.zero(Nd) for i in range(Ns) for j in range(i+1) ]

    @property
    def Ns(self):
        return self._Ns

    @property
    def Nd(self):
        return self._Nd

    def __iter__(self):
        return self._polys.__iter__()

    def __getitem__(self, pos) -> NdPoly:
        i, j = pos
        if j > i :
            i, j = j, i
        return self._polys[i*(i+1)//2 + j]

    def __setitem__(self, pos, value: NdPoly ):
        i, j = pos
        if j > i :
            i, j = j, i
        self._polys[i*(i+1)//2 + j] = value

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """ Returns a len(x)*Ns*Ns ndarray of values at x. """
        W = np.zeros((x.shape[0], self._Ns, self._Ns))
        for i in range(self._Ns):   # Compute lower triangular part
            for j in range(i+1):
                W[:,i,j] = self[i,j](x)
        for i in range(1, self._Ns):   # Copy to upper triangular off-diagonal part
            for j in range(i, self._Ns):
                W[:,i,j] = W[:,j,i]
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
        return SymPolyMat(Ns, Nd)

    @staticmethod
    def zero_like(other: SymPolyMat):
        """ Create copy with zero polynomial coefficients.
        Each matrix element is a polynomial with all powers as in other,
        but whose coefficients are all zero.
        """
        newmat = SymPolyMat(other.Ns, other.Nd)
        for i in range(Ns):
            for j in range(i+1):
                newmat[i,j] = NdPoly.zero_like(other[i,j])
        return newmat

    @staticmethod
    def eye(Ns, Nd):
        """ Create identity matrix
        Each matrix term is a constant monomial with coefficient 1 along
        the diagonal and 0 otherwise.
        """
        I = SymPolyMat(Ns, Nd)
        for i in range(Ns):
            I[i,i] = NdPoly({tuple([0 for _ in range(Nd)]): 1})
        return I

class Diabatizer:
    def __init__(self, Ns, Nd, diab_guess: SymPolyMat = None, **kwargs):
        self._Nd = Nd
        self._Ns = Ns
        self._Wguess = diab_guess if diab_guess is not None else SymPolyMat.eye(Ns, Nd)
        self._Wout = SymPolyMat.zero(Ns,Nd)
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

    def test_Zero(self):
        P = NdPoly.zero(3)
        X = np.repeat(self.testdata, 3, 1)
        assert np.all(P(X) == 0)

    def test_One(self):
        P = NdPoly.one(3)
        X = np.repeat(self.testdata, 3, 1)
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

    def test_PolyBadSet(self):
        P = NdPoly({(1,2,3): 0.1, (3,0,0): 3.14})
        with pytest.raises(AssertionError):
            P[(1,2)] = 3

    def test_powersCoeffs(self):
        P = NdPoly({(1,0,0): 1, (0,1,0): 3.14, (0,0,1): -1})
        assert list(P.powers()) == [(1,0,0), (0,1,0), (0,0,1)]
        assert list(P.coeffs()) == [1,3.14,-1]

class TestSymMat:
    testdata = np.linspace(0,1,4)[:,np.newaxis]

    def test_SymMat(self):
        Ns = 5
        Nd = 3
        M = SymPolyMat(Ns,Nd)
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
        lif.addPoints(x[:,np.newaxis],en,(0,1))
        lif.optimize()
        
        if pytestconfig.getoption("verbose") > 0:
            # Make plot using resulting parameters
            r_test = np.logspace(np.log10(2), np.log10(10), 100)
            x_test = 1 - np.exp(-stretch_factor*(r_test-r_x))
            x_test = x_test[:,np.newaxis]
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

    def test_2d2s(self, pytestconfig):
        """ Two states, two dimensions.
        1: Construct an arbitrary diabatic potential matrix W_test
        2: Diagonalize it in order to get the adiabatic PESs
        3: Fit a new W_fit to the adiabatic data and check if
           W_fit == W_test
        """
        import matplotlib.pyplot as plt

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
        W_test = SymPolyMat(2,2)
        W_test[0,0] = NdPoly({
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
        W_test[1,1] = NdPoly({
            (0,0): 0,
            (1,0): 0.5,
            (0,1): 0,
            (2,0): 1.5,
            (1,1): 0,
            (0,2): -0.2,
        })
        W_test[0,1] = NdPoly({(0,1): 5E-2})

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
        W_guess = SymPolyMat(2,2)
        # W_guess[0,0] = NdPoly({
        #     (0,0): 0,
        #     (1,0): -1,
        # })
        # W_guess[1,1] = NdPoly({
        #     (0,0): 0,
        #     (1,0): 0.5,
        #     (0,1): 0,
        #     (2,0): 1.5,
        # })
        # W_guess[0,1] = NdPoly({(0,1): 0})
        W_guess[0,0] = NdPoly.zero_like(W_test[0,0])
        W_guess[1,1] = NdPoly.zero_like(W_test[1,1])
        W_guess[1,0] = NdPoly.zero_like(W_test[1,0])
        
        test2d2s = Diabatizer(2,2,W_guess)
        test2d2s.addPoints(x_data, V_t, (0,1))
        test2d2s.optimize()
        W = test2d2s.Wout
        # for w, wt in zip(W,W_test):
        #     for c, ct in zip(w.coeffs(), wt.coeffs()):
        #         assert abs(c-ct) < 1E-10
                
        Wx = W(x_data)

        if pytestconfig.getoption("verbose") > 0:
            # Finally, show the result
            W11_t = W11_t.reshape(X.shape)
            W22_t = W22_t.reshape(X.shape)
            W21_t = W21_t.reshape(X.shape)
            V1_t = V_t[:,0].reshape(X.shape)
            V2_t = V_t[:,1].reshape(X.shape)

            W11 = Wx[:,0,0].reshape(X.shape)
            W22 = Wx[:,1,1].reshape(X.shape)
            W21 = Wx[:,1,0].reshape(X.shape)
            V1 = adiabatic2(W11,W22,W21,-1)
            V2 = adiabatic2(W11,W22,W21,1)

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

    def test_2d3s(self):
        pass

    def test_3d3s(self):
        pass
    
def main(argv) -> int:
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
