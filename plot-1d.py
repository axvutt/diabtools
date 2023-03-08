#!/usr/bin/python3

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from cycler import cycler as cyc
import sys
import copy
import ConstantsSI as SI
from dataclasses import dataclass

line_cycler = cyc(
        color=['red', 'green', 'blue', 'magenta', 'cyan', 'y'], 
        marker=['1','2','3','4','x','+'],
        )
plt.rc('lines', linewidth=0.5)
plt.rc('lines', linestyle='--')
plt.rc('axes', prop_cycle=line_cycler)

@dataclass
class PlotProps:
    x_lim_min : float
    x_lim_max : float
    x_step : float
    x_label : str
    y_lim_min : float
    y_lim_max : float
    y_step : float
    y_label : float
    legend_labels : str
    title : str

def makeFigure(rc, l_en, r_vlines, l_w, pprops):
    fig, lax = plt.subplots(2,1,gridspec_kw=dict(height_ratios=[3,1]))
    lsc = [[None] * 4 ] * 2
    
    ax = lax[0]
    sc = lsc[0]
    for n, en in enumerate(l_en):
        sc[n], = ax.plot(rc, en)
    ax.axhline(linewidth=0.5, color='gray')
    for rvl in r_vlines:
        ax.axvline(rvl, linewidth=0.5, color='gray', linestyle='dashed')
    ax.grid(True)
    ax.set_xlim(pprops.x_lim_min, pprops.x_lim_max)
    ax.set_ylim(pprops.y_lim_min, pprops.y_lim_max)
    ax.set_ylabel(pprops.y_label)
    ax.set_title(pprops.title)
    ax.legend(sc, pprops.legend_labels, loc="upper right")

    ax = lax[1]
    sc = lsc[1]
    for n, w in enumerate(l_w):
        sc[n], = ax.plot(rc, w)
    ax.axhline(linewidth=0.5, color='gray')
    for rvl in r_vlines:
        ax.axvline(rvl, linewidth=0.5, color='gray', linestyle='dashed')
    ax.grid(True)
    ax.set_xlim(pprops.x_lim_min, pprops.x_lim_max)
    ax.set_ylim(0, 1)
    ax.set_xlabel(pprops.x_label)
    ax.set_ylabel('Weights')
    ax.legend(sc, pprops.legend_labels, loc="upper right")

    fig.set_size_inches(11.69,8.27)     # A4 paper size

    return fig, lax, lsc

def parseChoice(indices_or_x):
    assert(indices_or_x.count("x") == 1)
    choice_list = [[] for l in range(3) ]
    for n,s in enumerate(indices_or_x):
        if s == "x":
            plot_coord = n
            continue
        comma_separated = s.split(",")
        for t in comma_separated:
            if ":" in t:
                colon_separated = t.split(":")
                start = int(colon_separated[0])
                end = int(colon_separated[1])
                for i in range(start,end+1):
                    choice_list[n].append(i)
            else:
                choice_list[n].append(int(t))
        
    return plot_coord, choice_list

def main():
    assert(len(sys.argv)==5 or len(sys.argv)==6)
    filename = sys.argv[1]
    coord_x, idx = parseChoice(sys.argv[2:5])
    saveFig = False
    if sys.argv[-1] == "-s":
        saveFig = True
    
    # Read reference energies
    e_ref = np.loadtxt('ref_en.txt')

    # Read data
    data = np.loadtxt(filename, skiprows=1)
    with open(filename, 'r') as f:
        strHeader = f.readline()
    headers = strHeader.split()
    
    ii = data[:,0].astype(int)
    jj = data[:,1].astype(int)
    kk = data[:,2].astype(int)
    coords = data[:,3:6]
    en_cols = [6,7,8,9]
    w_cols = [10,11,12,13]
    
    # Transform coordinates
    r_NN_2 = 0.5945
    r0 = 1.56814067
    R0 = 5
    coords[:,0] = (coords[:,0]+0.54725)/1.56814067 - 1
    coords[:,1] = 1 - np.exp(-(coords[:,1]-R0)/R0)

    iu = np.unique(ii)
    ju = np.unique(jj)
    ku = np.unique(kk)
    en = data[:,en_cols]
    w = data[:,w_cols]
    Nstates = len(en_cols)

    # Common plotting options
    leg_labels = [r"1 $A'$", r"2 $A'$", r"3 $A'$", r"1 $A''$"] 
    ap = dict(facecolor='black', arrowstyle="->")

    ### Plot against the first coordinate
    if coord_x == 0:
        r_vlines = [
                # 1.0341,  # NH in N2H+
                # 2.822,   # NH in tetratomic N2HRb+
                ]

        props = PlotProps(
                x_lim_min = -0.2 ,
                x_lim_max =  0.4 ,
                x_step = 0.05, 
                x_label = r"$x$" ,
                y_lim_min = -2.0 ,
                y_lim_max =  2.0 ,
                y_step = 0.5 ,
                y_label = r"Energy / eV" ,
                legend_labels = leg_labels ,
                title = "",
                )

        for j in idx[1] :
            for k in idx[2] :
                select = np.logical_and(jj == j, kk == k)
                if np.any(select) is False:
                    continue
                x = coords[select,0]
                sort_idx = np.argsort(x)
                x = x[sort_idx]
                y = coords[select,1][0]
                z = coords[select,2][0]
                l_en = []
                l_w = []
                for iz in range(Nstates):
                    en_i = (en[select,iz]-e_ref)*SI.h2ev
                    en_i = en_i[sort_idx]
                    l_en.append(en_i)
                    w_i = w[select,iz]
                    w_i = w_i[sort_idx]
                    l_w.append(w_i)

                title = r"Scan \ce{[Rb + N2H]+} with "\
                        + r"$y = "      + "{:5.3f}".format(y) + "$, "\
                        + r"$\theta = " + "{:4.1f}".format(z) + r"{}^{\circ}$, "\
                        + "DWSA-CASSCF with df=5" + r"\;hartree$^{-1}$"

                props.title = title
                fig, lax, lsc = makeFigure(x, l_en, r_vlines, l_w, props)

                if saveFig:
                    plt.savefig("rb-n2-h+_nh_j{:d}k{:d}.png".format(j,k))
                else:
                    plt.show()
                plt.close(fig)

    ### Plot against the second coordinate
    if coord_x == 1:
        r_vlines = [
                # 1.0341,  # NH in N2H+
                # 2.822,   # NH in tetratomic N2HRb+
                ]

        props = PlotProps(
                x_lim_min = -0.5 ,
                x_lim_max =  1.0 ,
                x_step = 0.1, 
                x_label = r"$y$" ,
                y_lim_min = -2.0 ,
                y_lim_max =  2.0 ,
                y_step = 0.5 ,
                y_label = r"Energy / eV" ,
                legend_labels = leg_labels ,
                title = "",
                )

        for i in idx[0] :
            for k in idx[2] :
                select = np.logical_and(ii == i, kk == k)
                if np.any(select) is False:
                    continue
                x = coords[select,1]
                sort_idx = np.argsort(x)
                x = x[sort_idx]
                y = coords[select,0][0]
                z = coords[select,2][0]
                l_en = []
                l_w = []
                for iz in range(Nstates):
                    en_i = (en[select,iz]-e_ref)*SI.h2ev
                    en_i = en_i[sort_idx]
                    l_en.append(en_i)
                    w_i = w[select,iz]
                    w_i = w_i[sort_idx]
                    l_w.append(w_i)

                title = r"Scan \ce{[Rb + N2H]+} with "\
                        + r"$x = "      + "{:5.3f}".format(y) + r"$, "\
                        + r"$\theta = " + "{:4.1f}".format(z) + r"{}^{\circ}$, "\
                        + "DWSA-CASSCF with df=5" + r"\;hartree$^{-1}$"

                props.title = title
                fig, lax, lsc = makeFigure(x, l_en, r_vlines, l_w, props)

                if saveFig:
                    plt.savefig("rb-n2-h+_grb_i{:d}k{:d}.png".format(i,k))
                else:
                    plt.show()
                plt.close(fig)

    ### Plot against the third coordinate
    if coord_x == 2:
        r_vlines = [
                # 1.0341,  # NH in N2H+
                # 2.822,   # NH in tetratomic N2HRb+
                ]

        props = PlotProps(
                x_lim_min = 0.0 ,
                x_lim_max =  180.0 ,
                x_step = 30.0, 
                x_label = r"$\theta$ / ${}^{\circ}$" ,
                y_lim_min = -2.0 ,
                y_lim_max =  2.0 ,
                y_step = 0.5 ,
                y_label = r"Energy / eV" ,
                legend_labels = leg_labels ,
                title = "",
                )

        for i in idx[0] :
            for j in idx[1] :
                select = np.logical_and(ii == i, jj == j)
                if np.any(select) is False:
                    continue
                x = coords[select,2]
                sort_idx = np.argsort(x)
                x = x[sort_idx]
                y = coords[select,0][0]
                z = coords[select,1][0]
                l_en = []
                l_w = []
                for iz in range(Nstates):
                    en_i = (en[select,iz]-e_ref)*SI.h2ev
                    en_i = en_i[sort_idx]
                    l_en.append(en_i)
                    w_i = w[select,iz]
                    w_i = w_i[sort_idx]
                    l_w.append(w_i)

                title = r"Scan \ce{[Rb + N2H]+} with "\
                        + r"$x = "  + "{:5.3f}".format(y) + r"$\;\AA{}, "\
                        + r"$y = "  + "{:5.3f}".format(z) + r"$\;\AA{}, "\
                        + "DWSA-CASSCF with df=5" + r"\;hartree$^{-1}$"

                props.title = title
                fig, lax, lsc = makeFigure(x, l_en, r_vlines, l_w, props)

                if saveFig:
                    plt.savefig("rb-n2-h+_theta_i{:d}j{:d}.png".format(i,j))
                else:
                    plt.show()
                plt.close(fig)


if __name__ == '__main__':
    main()
