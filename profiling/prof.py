#!/usr/bin/env python3

"""
Testing various optimizers in scipy.optimize.minimize in the diabatools package
"""
import sys
from datetime import datetime
import os
import cProfile
import pstats
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import ConstantsSI as SI
from diabtools.damping import Gaussian, Lorentzian
from diabtools.ndpoly import NdPoly
from diabtools.sympolymat import SymPolyMat
from diabtools.dampedsympolymat import DampedSymPolyMat
from diabtools.diabatizer import Diabatizer, adiabatic
from diabtools.diagnostics import MAE, RMSE, wMAE, wRMSE
from diabtools.results import Results
from diabtools.jsonutils import save_to_JSON, load_from_JSON
import prdata
import prplot

######## CONSTANTS #############

METHODS = (
        'BFGS'        ,
        )
# METHODS = (
#         # 'Nelder-Mead' ,
#         'Powell'      ,
#         'CG'          ,
#         'BFGS'        ,
#         # 'Newton-CG'   ,
#         'L-BFGS-B'    ,
#         'TNC'         ,
#         'COBYLA'      ,
#         'SLSQP'       ,
#         'trust-constr',
#         # 'dogleg'      ,
#         # 'trust-ncg'   ,
#         # 'trust-exact' ,
#         # 'trust-krylov',
#         )

CASES = (
    # "LiF",
    "JTCI",
)

NDNS = {
    "LiF": (1,2),
    "JTCI": (2,2),
}

DAMP_AXIS = {
    "LiF": (0,),
    "JTCI": (1,),
}

# DAMPING_TYPES = ("none", "gspread", "gtight", "lspread", "ltight")
DAMPING_TYPES = ("gtight",)

PLOT_FUNCTION = {
    "LiF": prplot.compare_data_1d,
    "JTCI": prplot.compare_data_2d,
}

########## END CONSTANTS ###########

def get_data(case):
    if case == "LiF":
        return prdata.LiF()

    if case == "JTCI":
        return prdata.JTCI_2d2s()
    # if case == (2,3):
    #     return case_2d3s()

    # if case == (3,2):
    #     return case_3d2s()
    # if case == (3,3):
    #     return case_3d3s()

    raise ValueError(f"No valid test case \"{case}\".")

def set_damping(W, case, type_):
    def damp(x0, damptype):
        if damptype == "gspread":
            return Gaussian(x0, 1)
        if damptype == "gtight":
            return Gaussian(x0, 0.1)
        if damptype == "lspread":
            return Lorentzian(x0, 1)
        if damptype == "ltight":
            return Lorentzian(x0, 0.1)
        if not damptype:
            return None
        raise ValueError("Unknown damping type")

    for i in range(1,W.Ns):
        for j in range(i):
            if case == "LiF":
                W.set_damping((i,j), 0, damp(float(W[i,j].x0[0]), type_))
            if case == "JTCI":
                W.set_damping((i,j), 1, damp(float(W[i,j].x0[1]), type_))

    return W


def make_dir():
    dirpath = "profiling_{}__0".format(datetime.now().strftime("%y%m%d-%H%M"))
    while os.path.exists(dirpath):
        ndir = int(dirpath.split("__")[-1])
        dirpath = dirpath.split("__")[0] + f"__{ndir+1}"
    os.mkdir(dirpath)
    return dirpath

def main():
    results = {}
    profile_stats = {}
    Wout = {}
    for case in CASES:
        Nd, Ns = NDNS[case]
        X, Y, Wguess = get_data(case)
        Wguess = DampedSymPolyMat.from_SymPolyMat(Wguess)
        for type_ in DAMPING_TYPES:
            Wguess = set_damping(Wguess, DAMP_AXIS[case], type_)
            diab = Diabatizer(Ns, Nd, Wguess)
            diab.add_domain(X, Y)
            for method in METHODS:
                print("#############{:^15s}##############".format(str((Nd, Ns))))
                print("#############{:^15s}##############".format(type_))
                print("#############{:^15s}##############".format(method))
                with cProfile.Profile() as profiler:
                    diab.optimize(method, verbose=True, print_every=1)
                    profiler.create_stats()
                    stats = pstats.Stats(profiler)
                stats.strip_dirs().sort_stats('cumulative', 'tottime')
                profile_stats[(case,type_,method)] = stats
                results[(case,type_,method)] = diab.results
                Wout[(case,type_,method)] = diab.Wout

    dirpath = make_dir()
    for case in CASES:
        Nd, Ns = NDNS[case]
        for type_ in DAMPING_TYPES:
            for method in METHODS:
                fname = f"{Nd}{Ns}_{type_}_{method}"
                X, Y, _ = get_data(case)
                save_to_JSON(results[(case,type_,method)], dirpath + "/res_" + fname + ".json")
                save_to_JSON(Wout[(case,type_,method)], dirpath + "/W_" + fname + ".json")
                profile_stats[(case,type_,method)].dump_stats(dirpath + "/pr_" + fname + ".stat")
                with open(dirpath + "/pr_" + fname + ".txt", 'w') as stream:
                    stats = pstats.Stats(dirpath + "/pr_" + fname + ".stat", stream=stream)
                    stats.print_stats()

                cost = results[(case,type_,method)].wrmse
                PLOT_FUNCTION[case](
                    X,Y,Wout[(case,type_,method)],
                    title = f"case={case} Nd={Nd} Ns={Ns} damp={type_} meth={method} wRMSE={cost}",
                    # fname = dirpath + "/plot_" + fname + ".png"
                )
    return 0

if __name__ == "__main__":
    sys.exit(main())
