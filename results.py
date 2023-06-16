from dataclasses import dataclass
import numpy as np
import scipy

@dataclass
class Results:
    success : bool = False
    coeffs : np.ndarray = np.array([])
    n_it : int = 0
    n_fev : int = 0
    n_jev : int = 0
    n_hev : int = 0
    rmse : float = 0.0
    wrmse : float = 0.0
    mae : float = 0.0
    wmae : float = 0.0
    delta_rmse : float = 0.0
    delta_wrmse : float = 0.0
    delta_mae : float = 0.0
    delta_wmae : float = 0.0
    residuals : np.ndarray = np.array([])
    cost : float = 0.0
    delta_cost : float = 0.0

    def reset(self):
        self.success = False
        self.coeffs = np.array([])
        self.n_it = 0
        self.rmse = 0.0
        self.wrmse = 0.0
        self.mae = 0.0
        self.wmae = 0.0
        self.delta_rmse = 0.0
        self.delta_wrmse = 0.0
        self.delta_mae = 0.0
        self.delta_wmae = 0.0
        self.residual = np.array([])
        self.cost = 0.0
        self.delta_cost = 0.0

    def from_OptimizeResult(self, optres : scipy.optimize.OptimizeResult):
        self.success = optres.success
        self.coeffs = optres.x
        self.cost = optres.fun
        if "nit" in optres:
            self.n_it = optres.nit
        if "nfev" in optres:
            self.n_fev = optres.nfev
        if "njev" in optres:
            self.n_jev = optres.njev
    
    def to_JSON_dict(self):
        out = {
                "__Results__" : True,
                "success"     : self.success,
                "coeffs"      : self.coeffs.tolist(),
                "n_it"        : self.n_it,
                "n_fev"       : self.n_fev,
                "n_jev"       : self.n_jev,
                "n_hev"       : self.n_hev,
                "rmse"        : self.rmse,
                "wrmse"       : self.wrmse,
                "mae"         : self.mae,
                "wmae"        : self.wmae,
                "delta_rmse"  : self.delta_rmse,
                "delta_wrmse" : self.delta_wrmse,
                "delta_mae"   : self.delta_mae,
                "delta_wmae"  : self.delta_wmae,
                "residuals"   : self.residuals.tolist(),
                "cost"        : self.cost,
                "delta_cost"  : self.delta_cost,
                }
        return out

    @staticmethod
    def from_JSON_dict(dct):
        if "__Results__" not in dct:
            raise KeyError("The JSON object is not a Results object.")

        return Results(
                success     = dct["success"    ],
                coeffs      = np.array(dct["coeffs"     ]),
                n_it        = dct["n_it"       ],
                n_fev       = dct["n_fev"      ],
                n_jev       = dct["n_jev"      ],
                n_hev       = dct["n_hev"      ],
                rmse        = dct["rmse"       ],
                wrmse       = dct["wrmse"      ],
                mae         = dct["mae"        ],
                wmae        = dct["wmae"       ],
                delta_rmse  = dct["delta_rmse" ],
                delta_wrmse = dct["delta_wrmse"],
                delta_mae   = dct["delta_mae"  ],
                delta_wmae  = dct["delta_wmae" ],
                residuals   = np.array(dct["residuals"  ]),
                cost        = dct["cost"       ],
                delta_cost  = dct["delta_cost" ],
                )
