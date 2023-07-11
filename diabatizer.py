from __future__ import annotations
from typing import Tuple
from copy import deepcopy
import math
import numpy as np
import scipy
from .sympolymat import SymPolyMat
from .dampedsympolymat import DampedSymPolyMat
from .results import Results
from .jsonutils import _str2tuple
from .diagnostics import RMSE, wRMSE, MAE, wMAE

class Diabatizer:
    def __init__(self, Ns, Nd, diab_guess = None):
        self._Nd = Nd
        self._Ns = Ns
        if diab_guess:
            self._Wguess = diab_guess
        else :
            self._Wguess = SymPolyMat.eye(Ns, Nd)
        self._Wout = deepcopy(self._Wguess)
        self._x = {}
        self._energies = {}
        self._states_by_domain = {}
        self._domain_IDs = set()
        self._Ndomains = 0
        self._last_domain_ID = 0
        self._auto_fit = True
        self._wfun_coord = None
        self._wfun_energy = None
        self._manually_weighted_domains = set()
        self._weights = {}
        self._weights_coord = {}
        self._weights_energy = {}
        self._weights_flat = None
        self._print_every = 5
        self._results = Results()

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

    def n_fitted_points(self):
        Npts = 0
        for dom_id, states in self._states_by_domain.items():
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

        if x.shape[1] != self._Nd:
            raise ValueError(
                "Wrong coord. array dimensions, "
                f"{x.shape[1]} vars, expected {self._Nd}."
                )
        if en.shape[1] != self._Ns:
            raise ValueError(
                "Wrong energy array dimensions "
                f"{en.shape[1]} states, expected {self._Ns}."
                )
        if x.shape[0] != en.shape[0]:
            raise ValueError("Coordinates vs energies dimensions mismatch.")

        id_ = self._last_domain_ID
        self._domain_IDs.add(id_)
        self._x[id_] = x
        self._weights_coord[id_] = np.ones((x.shape[0],1))
        self._energies[id_] = en
        self._weights_energy[id_] = np.ones_like(en)
        self._weights[id_] = self._weights_coord[id_] * self._weights_energy[id_]
        self._Ndomains += 1
        self._last_domain_ID += 1
        return id_

    def remove_domain(self, id_):
        self._domain_IDs.remove(id_)
        self._x.pop(id_)
        self._energies.pop(id_)
        self._weights_coord.pop(id_)
        self._weights_energy.pop(id_)
        self._weights.pop(id_)
        self._states_by_domain.pop(id_)
        self._Ndomains -= 1

    def set_fit_domain(self, id_: int, states: Tuple[int, ...] = None):
        """
        Specify the states that the diabatic potential matrix should fit in a given domain.
        If not called ever by the user, the Diabatizer instance is set in autoFit mode,
        i.e. all states in all the included domains are taken into account in the fitting
        procedure.

        When called, the autoFit mode is disabled.

        Args:
            id_ :
                domain identifier. Integer.
            states :
                tuple of states indices (integers), or
                "all" literal string, in which case all states are selected, or
                None, same effect as "all"

        Raises:
            IndexError: invalid state selection.
            IndexError: invalid domain selection.
        """
        if states is None or states == "all":
            states = tuple(s for s in range(self._Ns))

        if id_ not in self._domain_IDs:
            raise IndexError(
                f"Chosen domain identifier {id_} has not been assigned yet. "
                f"Choose among: {self._domain_IDs}."
            )

        if any(s < 0 or self._Ns <= s for s in states):
            raise IndexError(
                f"One of specified states {states} is out of range, "
                f"should be 0 <= s < {self._Ns}."
            )

        self._states_by_domain[id_] = states
        self._auto_fit = False

    def set_fit_all_domain(self):
        for idd in self._domain_IDs:
            self.set_fit_domain(idd, "all")

    def set_domain_weight(self, id_domain, weight):
        """ Assign a fixed weight to a coordinate domain. """
        self._manually_weighted_domains.add(id_domain)
        self._weights[id_domain] = np.full_like(self._energies[id_domain], weight)

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
            raise ValueError(
                    "Weighting function must be applied to either \'coord\' or \'energy\'."
                    )

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
            if y > mu_up:
                return np.exp(-0.5*((y-mu_up)/sigma_up)**2)
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
            if y > y_up:
                return np.exp(-(y-y_up)/beta_up)
            return 1
        self.set_weight_function(lambda y: expband(y, x0[0], x0[1], beta[0], beta[1]), apply_to)

    def compute_weights(self):
        """
        Precompute weights assigned to points in coordinate space or the corresponding energies
        if weighting functions have been specified.
        Also, cache the results in a flat 1d-array
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

        # Cache to flat 1d-array
        weights = []
        for id_, states in self._states_by_domain.items():
            for s in states:
                weights.append(self._weights[id_][:,s])
        self._weights_flat = np.hstack(tuple(weights))


    def compute_errors(self):
        """
        Compute weighted and unweighted RMSE and MAE between
        reference adiabatic energies and those deduced from of all
        current diabatic matrices in Wout, within the associated domains
        """

        res = self._compute_residuals(self._Wout, self.states_by_domain)
        self._results.rmse = RMSE(res)
        self._results.wrmse = wRMSE(res,self._weights_flat)
        self._results.mae = MAE(res)
        self._results.wmae = wMAE(res,self._weights_flat)


    def _compute_residuals(self, W: SymPolyMat, states_by_domain: dict):
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
        for id_, states in states_by_domain.items():
            # Compute adiabatic potential from diabatic ansatz
            x = self._x[id_]
            Wx = W(x)
            Vx, _ = adiabatic(Wx)

            # Retreive data energies and compute (weighted) residuals over the domain
            for s in states:
                Vdata = self._energies[id_][:,s]
                if np.any(np.isnan(Vdata)):
                    raise(ValueError(f"Found NaN energies in domain {id_}, state {s}. "
                        + "Please deselect from fitting dataset."))

                residuals.append(Vx[:,s]-Vdata)

        return np.hstack(tuple(residuals))

    def _cost(self, c):
        """
        Compute cost function for finding optimal diabatic anzats coefficients.

        This method is passed to the optimizer in order to find the optimal coefficients c
        such that the adiabatic surfaces obtained from diagonalization of the
        diabatic ansatz fit the adiabatic data.

        Args:
            c : 1d list/numpy array of coefficients. Assumed to be sorted as expected by
            SymPolyMat.update() does, see related documentation.

        Returns:
            cost: value of the cost function (weighted RMSE).
        """
        self._Wout.update(self._Wguess.keys(),c)
        res = self._compute_residuals(self._Wout, self._states_by_domain)
        wrmse = wRMSE(res, self._weights_flat)
        return wrmse


    def _cache_results(self,c):
        self._Wout.update(self._Wguess.keys(),c)
        res = self._compute_residuals(self._Wout, self._states_by_domain)
        self._results.residuals = res

        tmp_rmse = RMSE(res)
        tmp_wrmse = wRMSE(res, self._weights_flat)
        tmp_mae = MAE(res)
        tmp_wmae = wMAE(res, self._weights_flat)

        if self._results.coeffs.size != 0:
            self._results.dc2 = Results.dist_l2(self._results.coeffs, c)
        self._results.coeffs = c

        self._results.delta_rmse   = tmp_rmse  - self._results.rmse
        self._results.delta_wrmse  = tmp_wrmse - self._results.wrmse
        self._results.delta_mae    = tmp_mae   - self._results.mae
        self._results.delta_wmae   = tmp_wmae  - self._results.wmae

        self._results.rmse  = tmp_rmse
        self._results.wrmse = tmp_wrmse
        self._results.mae   = tmp_mae
        self._results.wmae  = tmp_wmae

        self._results.cost  = self._results.wrmse
        self._results.delta_cost  = self._results.delta_wrmse


    def _cost_verbose_callback(self, c, state=None):
        """ Callback function called at end of each call of _cost, if verbose option is turned on. """

        self._results.increment_nit()
        n_it = self._results.n_it

        # Compute and store coeffs and errors at print iteration
        # (and preceding iteration, for deltas)
        if ((n_it + 1) % self._print_every == 0
            or n_it % self._print_every == 0):
            self._cache_results(c)

        if n_it % self._print_every == 0:
            print(
                "{:<6d}".format(self._results.n_it)
                + (" {: 12.4e}"*9).format(
                    self._results.dc2,
                    self._results.wrmse,
                    self._results.delta_wrmse,
                    self._results.rmse,
                    self._results.delta_rmse,
                    self._results.wmae,
                    self._results.delta_wmae,
                    self._results.mae,
                    self._results.delta_mae,
                )
            )


    def optimize(self, method="l-bfgs-b", method_options=None, verbose=False, print_every=5):
        """ Run optimization

        Find best coefficients for polynomial diabatics and couplings fitting
        given adiabatic data.
        """
        # By default, if no specific domain setting is given, use all the data
        # in the database for the fit
        if self._auto_fit:
            self.set_fit_all_domain()

        # Initialize
        self._results.reset()
        self._Wout = deepcopy(self._Wguess)
        self.compute_weights()
        coeffs = self._Wguess.coeffs()

        # Set verbose callback if desired
        verb_callback = None
        if verbose:
            verb_callback = self._cost_verbose_callback
            self._print_every = max(print_every,1)
            print(
                "{:<6s}".format("I")
                + (" {:>12s}"*9).format(
                    "dC",
                    "wRMSE",
                    "dwRMSE",
                    "RMSE",
                    "dRMSE",
                    "wMAE",
                    "dwMAE",
                    "MAE",
                    "dMAE",
                )
            )

        optres = scipy.optimize.minimize(
            self._cost,    # Objective function to minimize
            coeffs,                 # Initial guess
            method=method,
            options=method_options,
            callback=verb_callback
        )

        self._Wout.update(self._Wguess.keys(), optres.x)
        self._results.from_OptimizeResult(optres)

        self.compute_errors()

        return self._Wout

    def to_JSON_dict(self):
        dct = {
                "__Diabatizer__" : True,
                "Nd"                        : self._Nd,
                "Ns"                        : self._Ns,
                "Wguess"                    : self._Wguess.to_JSON_dict(),
                "Wout"                      : self._Wout.to_JSON_dict(),
                "x"                         : {id_ : x.tolist() for id_,x in self._x.items()},
                "energies"                  : {id_ : e.tolist()
                                                    for id_,e in self._energies.items()},
                "states_by_domain"          : {id_ : str(states)
                                                    for id_, states in
                                                        self._states_by_domain.items()},
                "domain_IDs"                : list(self._domain_IDs),
                "Ndomains"                  : self._Ndomains,
                "last_domain_ID"            : self._last_domain_ID,
                "auto_fit"                  : self._auto_fit,
                "wfun_coord"                : None, # self._wfun_coord,
                "wfun_energy"               : None, # self._wfun_energy,
                "manually_weighted_domains" : list(self._manually_weighted_domains),
                "weights"                   : {id_: w.tolist()
                                                    for id_, w in self._weights.items()},
                "weights_coord"             : {id_: w.tolist()
                                                    for id_, w in self._weights_coord.items()},
                "weights_energy"            : {id_: w.tolist()
                                                    for id_, w in self._weights_energy.items()},
                "print_every"               : self._print_every,
                "results"                   : self.results.to_JSON_dict(),
                }
        return dct

    @staticmethod
    def from_JSON_dict(dct):
        """
        Deserialize Diabatizer in an intrusive way, i.e. setting some private members manually.
        ! This may become dangerous if the structure of the Diabatizer class change.
        """

        if "__DampedSymPolyMat__" in dct["Wguess"]:
            Wguess = DampedSymPolyMat.from_JSON_dict(dct["Wguess"])
        else:
            Wguess = SymPolyMat.from_JSON_dict(dct["Wguess"])

        if "__DampedSymPolyMat__" in dct["Wout"]:
            Wout = DampedSymPolyMat.from_JSON_dict(dct["Wout"])
        else:
            Wout = SymPolyMat.from_JSON_dict(dct["Wout"])

        diab = Diabatizer(dct["Nd"], dct["Ns"], Wguess)
        diab._Wguess = Wguess
        diab._Wout = Wout
        diab._x = {id_: np.array(xlist) for id_, xlist in dct["x"].items()}
        diab._energies = {
            id_: np.array(elist) for id_, elist in dct["energies"].items()
        }
        diab._states_by_domain = {
            id_: _str2tuple(str_states) for id_, str_states in dct["states_by_domain"].items()
        }
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
        diab._results = Results.from_JSON_dict(dct["results"])


def adiabatic2(W1, W2, W12, sign):
    """ Return analytical 2-states adiabatic potentials from diabatic ones and coupling. """
    m = 0.5*(W1+W2)
    p = W1*W2 - W12**2
    return m + sign * np.sqrt(m**2 - p)

def adiabatic(W):
    """ Return numerical N-states adiabatic potentials from diabatic ones and couplings. """
    return np.linalg.eigh(W)
