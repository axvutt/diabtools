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

def case_LiF():
    # Load data
    filename = 'lif_mr_mscaspt2.csv'
    data = np.genfromtxt(filename,delimiter=',',skip_header=4)

    # Coordinates and energies
    X = data[:,0]
    X = 1 - np.exp(-(X-6.5)/6.5)
    Y = data[:,5:]
    Y = (Y - Y[-1,0])*SI.h2ev

    # Initial guess
    Wref = SymPolyMat(2,1)
    Wref[0,0] = NdPoly({(0,): 0.1, (1,): 1, (2,): -1, (3,): 1, (4,): 1, (5,): 1})
    Wref[1,1] = NdPoly({(0,): 0.1, (1,): -1, (2,): 1, (3,): 1, (4,): 1, (5,): 1, (6,): 1, (7,): 1})
    Wref[1,0] = NdPoly({(0,): 0.1, (2,): 0.1})

    # Intersection location
    Wref[1,0].x0 = 0

    return X, Y, Wref

def test_case(Nd, Ns):
    case = (Nd, Ns)
    if case == (1,2):
        return case_LiF()

    if case == (2,2):
        return case_2d2s()
    if case == (2,3):
        return case_2d3s()

    if case == (3,2):
        return case_3d2s()
    if case == (3,3):
        return case_3d3s()

    raise ValueError(f"No test case for Nd={Nd} and Ns={Ns}.")

def set_damping(W, axis, type_):
    def damp(x0, damptype):
        if damptype == "gspread":
            return Gaussian(x0, 1)
        if damptype == "gtight":
            return Gaussian(x0, 0.1)
        if damptype == "lspread":
            return Lorentzian(x0, 1)
        if damptype == "ltight":
            return Lorentzian(x0, 0.1)
        return

    for i in range(1,W.Ns):
        for j in range(i):
            W.set_damping((i,j), axis, damp(float(W[i,j].x0[axis]), type_))

    return W

def plot_LiF(x_data, y_data, W, title = None):
    x = np.linspace(x_data.min(), x_data.max(), 10*x_data.size)

    fig, ax = plt.subplots()

    Wx = W(x)
    ax.plot(x, Wx[:,0,0])
    ax.plot(x, Wx[:,1,1])
    ax.plot(x, Wx[:,0,1])

    V, _ = adiabatic(W(x))
    ax.plot(x, V[:,0], lw=0.5)
    ax.plot(x, V[:,1], lw=0.5)
    ax.plot(x_data, y_data[:,0], lw=0, marker='x')
    ax.plot(x_data, y_data[:,1], lw=0, marker='+')

    ax.set_title(title)
    ax.grid(True)
    plt.show()

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

# CASES = ((1,2), (2,2), (2,3), (3,2), (3,3))
CASES = ((1,2),)
DAMP_AXIS = {
    (1,2): 0,
    (2,2): 0,
    (2,3): 0,
    (3,2): 0,
    (3,2): 0,
}

DAMPING_TYPES = ("none", "gspread", "gtight", "lspread", "ltight")
# DAMPING_TYPES = ("gspread",)

PLOT_FUNCTION = {
    (1,2): plot_LiF
}



def main():
    results = {}
    profile_stats = {}
    Wout = {}
    for Nd, Ns in list(CASES):
        X, Y, Wref = test_case(Nd, Ns)
        Wguess = DampedSymPolyMat.from_SymPolyMat(Wref)
        for type_ in DAMPING_TYPES:
            Wguess = set_damping(Wguess, DAMP_AXIS[(Nd,Ns)], type_)
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
                profile_stats[(Nd,Ns,type_,method)] = stats
                results[(Nd,Ns,type_,method)] = diab.results
                Wout[(Nd,Ns,type_,method)] = diab.Wout
                stats.print_stats()
                print(diab.results)

    dirpath = "profiling_{}__0".format(datetime.now().strftime("%y%m%d-%H%M"))
    while os.path.exists(dirpath):
        ndir = int(dirpath.split("__")[-1])
        dirpath = dirpath.split("__")[0] + f"__{ndir+1}"
    os.mkdir(dirpath)

    for Nd, Ns in list(CASES):
        for type_ in DAMPING_TYPES:
            for method in METHODS:
                fname = f"{Nd}{Ns}_{type_}_{method}"
                X, Y, _ = test_case(Nd, Ns)
                save_to_JSON(results[(Nd,Ns,type_,method)], dirpath + "/res_" + fname + ".json")
                save_to_JSON(Wout[(Nd,Ns,type_,method)], dirpath + "/W_" + fname + ".json")
                profile_stats[(Nd,Ns,type_,method)].dump_stats(dirpath + "/pr_" + fname + ".stat")
                with open(dirpath + "/pr_" + fname + ".txt", 'w') as stream:
                    stats = pstats.Stats(dirpath + "/pr_" + fname + ".stat", stream=stream)
                    stats.print_stats()

                cost = results[(Nd,Ns,type_,method)].wrmse
                PLOT_FUNCTION[(Nd, Ns)](
                    X,Y,Wout[(Nd,Ns,type_,method)],
                    title=f"Nd={Nd} Ns={Ns} damp={type_} meth={method} wRMSE={cost}"
                )
    return 0

if __name__ == "__main__":
    sys.exit(main())
