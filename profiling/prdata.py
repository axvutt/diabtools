#!/usr/bin/env python3

"""
Data used to profile the diabatools package
"""
from copy import deepcopy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import ConstantsSI as SI
from diabtools.ndpoly import NdPoly
from diabtools.sympolymat import SymPolyMat

def LiF():
    # Load data
    filename = 'lif_mr_mscaspt2.csv'
    data = np.genfromtxt(filename,delimiter=',',skip_header=4)

    # Coordinates and energies
    X = data[:,0]
    X = 1 - np.exp(-(X-6.5)/6.5)
    Y = data[:,5:]
    Y = (Y - Y[-1,0])*SI.h2ev

    # Initial guess
    Wguess = SymPolyMat(2,1)
    Wguess[0,0] = NdPoly({(0,): 0.1, (1,): 1, (2,): -1, (3,): 1, (4,): 1, (5,): 1})
    Wguess[1,1] = NdPoly({(0,): 0.1, (1,): -1, (2,): 1, (3,): 1, (4,): 1, (5,): 1, (6,): 1, (7,): 1})
    Wguess[1,0] = NdPoly({(0,): 0.1, (2,): 0.1})

    # Intersection location
    Wguess[1,0].x0 = 0

    return X, Y, Wguess


def JTCI_2d2s():
    dx1 = 1
    dx2 = -1
    W = SymPolyMat(2,2)
    W[0,0] = NdPoly({(2,0): 0.5, (0,2): 0.5,})
    W[0,0][(0,0)] = W[0,0][(2,0)]*dx1**2 #- W[0,0][(1,0)]*dx1
    W[0,0][(1,0)] = -2*W[0,0][(2,0)]*dx1
    W[1,1] = deepcopy(W[0,0])
    W[1,1][(0,0)] = W[0,0][(2,0)]*dx2**2 #- W[0,0][(1,0)]*dx2
    W[1,1][(1,0)] = -2*W[0,0][(2,0)]*dx2
    W[0,1] = NdPoly({(0,1): 5E-1})
    
    Xg, Yg = np.mgrid[-2:2:51j, -2:2:51j]
    X = np.vstack((Xg.ravel(), Yg.ravel())).transpose()
    E = W(X).diagonal(axis1=1, axis2=2)

    Wguess = SymPolyMat.zero_like(W)
    Wguess[0,0][(1,0)] = 0.1
    Wguess[1,1][(1,0)] = -0.1

    return X, E, Wguess


def main():
    fig, ax = plt.subplots()
    x, y, _ = LiF()
    ax.scatter(x,y[:,0], marker='+')
    ax.scatter(x,y[:,1], marker='+')
    plt.show()

if __name__ == "__main__":
    main()
