from __future__ import annotations
from typing import List, Tuple
import numpy as np
import scipy
from .sympolymat import SymPolyMat
from .dampedsympolymat import DampedSymPolyMat
from .results import Results
from .jsonutils import _str2tuple
from .diagnostics import RMSE, wRMSE, MAE, wMAE

class Diabatizer:
    def __init__(self, Ns, Nd, Nm, diab_guess: List[SymPolyMat] = None):
        self._Nd = Nd
        self._Ns = Ns
        self._Nm = Nm
        if diab_guess is not None:
            if len(diab_guess) != Nm:
                raise ValueError(
                    f"Inconsistent number of matrices {Nm}"
                    f"vs number of given guesses {len(diab_guess)}."
                    )
            self._Wguess = diab_guess
        else :
            self._Wguess = [SymPolyMat.eye(Ns, Nd) for _ in range(Nm)]
        self._Wout = self._Wguess
        self._x = {}
        self._energies = {}
        self._states_by_domain = [{} for _ in range(Nm)]
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
        for states_selection_per_matrix in self._states_by_domain:
            if id_ in states_selection_per_matrix:
                states_selection_per_matrix.pop(id_)
        self._Ndomains -= 1

    def set_fit_domain(self, n_matrix: int, id_domain: int, states: Tuple[int, ...] = None):
        """ Specify the domain and states that a diabatic potential matrix
        should fit.
        """
        if states is None:
            states = tuple(s for s in range(self._Ns))

        if n_matrix >= self._Nm:
            raise ValueError(
                "Matrix index should be less than "
                f"{self._Nm}, got {n_matrix}."
                )
        if any(s < 0 or self._Ns <= s for s in states):
            raise ValueError(
                f"One of specified states {states} is out of range, "
                f"should be 0 <= s < {self._Ns}."
                )

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

        # Compute errors for each matrix
        for im in range(self._Nm):
            res = []
            w = []

            # Evaluate adiabatic matrices over each domain
            for id_, states in self.states_by_domain[im].items():
                x = self._x[id_]
                Wx = self._Wout[im](x)
                Vx, _ = adiabatic(Wx)

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
            Vx, _ = adiabatic(Wx)

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

    def optimize(self, method="l-bfgs-b", verbose=0, maxiter=1000):
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
                    args=(
                        keys,
                        origins,
                        this_matrix_domains,
                        weights
                    ),   # other arguments passed to objective function
                    method="l-bfgs-b",
                    options={
                        "gtol": 1e-08,      # Termination conditions (quality)
                        # "xtol": 1e-08,
                        "maxiter": maxiter, # Termination condition (# iterations)
                        # "verbose": verbose, # Printing option
                        }
                    )

            self._Wout[i_matrix] = SymPolyMat.construct(
                    self._Ns, self._Nd, keys, optres.x, origins
            )
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
                "energies"                  : {id_ : e.tolist()
                                                    for id_,e in self._energies.items()},
                "states_by_domain"          : [{id_ : str(states) for id_,states in dct.items()}
                                                    for dct in self._states_by_domain],
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
    def Wguess(self):
        return self._Wguess[0]

    @Wguess.setter
    def Wguess(self, guess):
        self._Wguess[0] = guess

    @property
    def Wout(self):
        return self._Wout[0]


def adiabatic2(W1, W2, W12, sign):
    """ Return analytical 2-states adiabatic potentials from diabatic ones and coupling. """
    m = 0.5*(W1+W2)
    p = W1*W2 - W12**2
    return m + sign * np.sqrt(m**2 - p)

def adiabatic(W):
    """ Return numerical N-states adiabatic potentials from diabatic ones and couplings. """
    return np.linalg.eigh(W)
