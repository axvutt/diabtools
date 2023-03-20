import sys
from copy import *
from typing import List, Tuple, Dict
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

    @property
    def Nd(self):
        """ Number of variables of the polynomial """
        return self._Nd

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

    @staticmethod
    def zero(Nd):
        powers = tuple([0 for _ in range(Nd)])
        return NdPoly({powers: 0})



class SymPolyMat():
    """ Real symmetric matrix of multivariate polynomial functions """
    def __init__(self, Ns, Nd):
        self._Ns = Ns
        self._Nd = Nd
        self._polys = [[ NdPoly.zero(Nd) for j in range(i+1) ] for i in range(Ns) ]

    @property
    def Ns(self):
        return self._Ns

    @property
    def Nd(self):
        return self._Nd

    def __getitem__(self, pos) -> NdPoly:
        i, j = pos
        if j > i :
            i, j = j, i
        return self._polys[i][j]

    def __setitem__(self, pos, value: NdPoly ):
        i, j = pos
        if j > i :
            i, j = j, i
        self._polys[i][j] = value

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """ Returns a len(x)*Ns*Ns ndarray of values at x. """
        W = np.zeros((x.shape[0], self._Ns, self._Ns))
        for i in range(self._Ns):   # Compute lower triangular part
            for j in range(i+1):
                W[:,i,j] = self._polys[i][j](x)
        for i in range(1, self._Ns):   # Copy to upper triangular off-diagonal part
            for j in range(i, self._Ns):
                W[:,i,j] = W[:,j,i]
        return W

    def __repr__(self):
        s = ""
        for i in range(self._Ns):
            for j in range(i+1):
                s += f"({i},{j}): {self._polys[i][j].__repr__()}" + "\n"
        return s

    def __str__(self):
        s = ""
        for i in range(self._Ns):
            for j in range(i+1):
                s += f"({i},{j}): {self._polys[i][j].__str__()}" + "\n"
        return s
   
    @staticmethod
    def zero(Ns, Nd):
        return SymPolyMat(Ns, Nd)

    @staticmethod
    def eye(Ns, Nd):
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

    def test_ZeroPoly(self):
        P = NdPoly.zero(3)
        X = np.repeat(self.testdata, 3, 1)
        assert np.all(P(X) == 0)

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
    def test_LiF(self):
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
        print(lif.Wout)
        
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

    def test_2d2s(self):
        pass

    def test_2d3s(self):
        pass

    def test_3d3s(self):
        pass
    
def main(argv) -> int:
    TestDiabatizer().test_LiF()
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
