import json
from . import diabtools

class DiabJSONEncoder(json.JSONEncoder):
    def encode_NdPoly(self, obj): 
        poly = {str(p) : c for p,c in obj.items()}
        out = {
                "__NdPoly__" : True,
                "Nd" : obj.Nd,
                "x0" : obj.x0.tolist(),
                "coeffs_by_powers" : poly,
                }
        return out

    def encode_SymPolyMat(self, obj):
        Wij = dict()
        for i in range(obj.Ns):
            for j in range(i+1):
                Wij[f"({i}, {j})"] = self.default(obj[i,j])
        out = {
                "__SymPolyMat__" : True,
                "Nd" : obj.Nd,
                "Ns" : obj.Ns,
                "elements" : Wij
                }
        return out

    def encode_DampedSymPolyMat(self, obj: diabtools.DampedSymPolyMat):
        out = self.encode_SymPolyMat(obj)
        damping = dict()
        for i in range(obj.Ns):
            for j in range(i+1):
                damping[f"({i}, {j})"] = self.default(obj.get_damping(i,j))
        out["damping"] = damping
        return out

    def encode_Damping(self, obj: diabtools.DampingFunction):
        out = {
                "__DampingFunction__": True,
                "x0": obj.x0
                }
        return out

    def encode_Gaussian(self, obj: diabtools.Gaussian):
        out = self.encode_Damping(obj)
        out.update({
                "__Gaussian__": True,
                "sigma" : obj.sigma
                })
        return out

    def encode_Lorenzian(self, obj: diabtools.Lorentzian):
        out = self.encode_Damping(obj)
        out.update({
                "__Lorentzian__": True,
                "gamma" : obj.gamma
                })
        return out

    def encode_Results(self, obj: diabtools.Results):
        out = {
                "__Results__" : True,
                "success"     : obj.success
                "coeffs"      : obj.coeffs
                "n_it"        : obj.n_it
                "n_fev"       : obj.n_fev
                "n_jev"       : obj.n_jev
                "n_hev"       : obj.n_hev
                "rmse"        : obj.rmse
                "wrmse"       : obj.wrmse
                "mae"         : obj.mae
                "wmae"        : obj.wmae
                "delta_rmse"  : obj.delta_rmse
                "delta_wrmse" : obj.delta_wrmse
                "delta_mae"   : obj.delta_mae
                "delta_wmae"  : obj.delta_wmae
                "residuals"   : obj.residuals
                "cost"        : obj.cost
                "delta_cost"  : obj.delta_cost
                }
        return out

    def encode_Diabatizer(self, obj: diabtools.Diabatizer):
        out = {
                "__Diabatizer__" : True,
                "Nd"                        : obj._Nd
                "Ns"                        : obj._Ns
                "Nm"                        : obj._Nm
                "Wguess"                    : [self.default(Wg) for Wg in range(obj._Wguess)]
                "Wout"                      : [self.default(Wo) for Wo in range(obj._Wout)]
                "x"                         : {id_ : x.tolist() for id_,x in obj._x.items()}
                "energies"                  : {id_ : e.tolist() for id_,e in obj._e.items()}
                "states_by_domain"          : [{id_ : str(states) for id_,states in dct.items()} \
                                                    for dct in obj._states_by_domain]
                "domain_IDs"                : list(obj._domain_IDs)
                "Ndomains"                  : obj._Ndomains
                "last_domain_ID"            : obj._last_domain_ID
                "auto_fit"                  : obj._auto_fit
                "wfun_coord"                : None # obj._wfun_coord
                "wfun_energy"               : None # obj._wfun_energy
                "manually_weighted_domains" : list(obj._manually_weighted_domains)
                "weights"                   : [w.tolist() for w in obj._weights]
                "weights_coord"             : [w.tolist() for w in obj._weights_coord]
                "weights_energy"            : [w.tolist() for w in obj._weights_energy]
                "print_every"               : obj._print_every
                "n_cost_calls"              : obj._n_cost_calls
                "last_residuals"            : obj._last_residuals.tolist()
                "results"                   : [self.default(r) for r in obj.results]
                }
        return out

    def default(self, obj):
        if isinstance(obj, diabtools.NdPoly):
            return self.encode_NdPoly(obj)

        if isinstance(obj, diabtools.SymPolyMat):
            if isinstance(obj, diabtools.DampedSymPolyMat):
                return self.encode_DampedSymPolyMat(obj)
            return self.encode_SymPolyMat(obj)

        if isinstance(obj, diabtools.DampingFunction):
            if isinstance(obj, diabtools.Gaussian):
                return self.encode_Gaussian(obj)
            if isinstance(obj, diabtools.Lorenzian):
                return self.encode_Lorenzian(obj)
            raise Warning("Serialized abstract DampingFunction instance.")

        if isinstance(obj, diabtools.Results):
            return self.encode_Results(obj)
        
        if isinstance(obj, diabtools.Diabatizer):
            return self.encode_Diabatizer(obj)

        # If not a diabtools type, use default parser
        return json.JSONEncoder.default(self, obj)


