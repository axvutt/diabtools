#!/usr/bin/env python3

"""
Testing various optimizers in scipy.optimize.minimize in the diabatools package
"""
import sys
import cProfile
import pstats
import numpy as np
import ConstantsSI as SI
from diabtools.damping import Gaussian, Lorentzian
from diabtools.ndpoly import NdPoly
from diabtools.sympolymat import SymPolyMat
from diabtools.dampedsympolymat import DampedSymPolyMat
from diabtools.diabatizer import Diabatizer
from diabtools.diagnostics import MAE, RMSE, wMAE, wRMSE
from diabtools.results import Results

METHODS = (
        # 'Nelder-Mead' ,
        'Powell'      ,
        'CG'          ,
        'BFGS'        ,
        # 'Newton-CG'   ,
        'L-BFGS-B'    ,
        'TNC'         ,
        'COBYLA'      ,
        'SLSQP'       ,
        'trust-constr',
        # 'dogleg'      ,
        # 'trust-ncg'   ,
        # 'trust-exact' ,
        # 'trust-krylov',
        )


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
    Wref[1,0] = NdPoly({(0,): 0.0})

    # Intersection location
    x0 = {(0,1): 6.5}

    return X, Y, Wref, x0


# CASES = ((1,2), (2,2), (2,3), (3,2), (3,3))
CASES = ((1,2),)

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


DAMPING_TYPES = ("none", "gspread", "gtight", "lspread", "ltight")

def set_damping(W, x0, type_):
    if type_ == "gspread":
        for i in range(1,W.Ns):
            for j in range(i):
                if (i,j) in x0:
                    xint = x0[(i,j)]
                    W.set_damping((i,j), 0, Gaussian(xint, 1))
    if type_ == "gtight":
        for i in range(1,W.Ns):
            for j in range(i):
                if (i,j) in x0:
                    xint = x0[(i,j)]
                    W.set_damping((i,j), 0, Gaussian(xint, 0.1))
    if type_ == "lspread":
        for i in range(1,W.Ns):
            for j in range(i):
                if (i,j) in x0:
                    xint = x0[(i,j)]
                    W.set_damping((i,j), 0, Lorentzian(xint, 1))
    if type_ == "ltight":
        for i in range(1,W.Ns):
            for j in range(i):
                if (i,j) in x0:
                    xint = x0[(i,j)]
                    W.set_damping((i,j), 0, Lorentzian(xint, 0.1))
    return W


def main():
    results = {}
    profile_stats = {}
    for Nd, Ns in list(CASES):
        X, Y, Wref, x0 = test_case(Nd, Ns)
        Wguess = DampedSymPolyMat.from_SymPolyMat(
            SymPolyMat.zero_like(Wref)
            )
        for type_ in DAMPING_TYPES:
            Wguess = set_damping(Wguess, x0, type_)
            diab = Diabatizer(Ns, Nd, 1, [Wguess,])
            diab.add_domain(X, Y)
            for method in METHODS:
                print("#############{:^15s}##############".format(method))
                with cProfile.Profile() as profiler:
                    diab.optimize(method)
                    profiler.create_stats()
                    stats = pstats.Stats(profiler)
                stats.strip_dirs().sort_stats('cumulative', 'tottime')
                profile_stats[(Nd,Ns,type_,method)] = stats
                results[(Nd,Ns,type_,method)] = diab.results[0]
                stats.print_stats()
                print(diab.results[0])
    return 0

if __name__ == "__main__":
    sys.exit(main())
