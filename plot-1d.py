#!/usr/bin/python3

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from cycler import cycler as cyc
import sys
from copy import *
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
    x_label : str
    y_lim_min : float
    y_lim_max : float
    y_label : float
    legend_labels : str
    title : str

def MakeFigure(x, energies_a, energies_d, pprops):
    fig, ax = plt.subplots()
    lines_a = [None] * (len(energies_a))
    lines_d = [None] * (len(energies_d))
    
    for n, en in enumerate(energies_a):
        lines_a[n], = ax.plot(x, en)

    for n, en in enumerate(energies_d):
        lines_d[n], = ax.plot(x, en)

    ax.axhline(linewidth=0.5, color='gray')

    ax.grid(True)
    ax.set_xlim(pprops.x_lim_min, pprops.x_lim_max)
    ax.set_ylim(pprops.y_lim_min, pprops.y_lim_max)
    ax.set_ylabel(pprops.y_label)
    ax.set_title(pprops.title)
    ax.legend(lines_a, pprops.legend_labels, loc="upper right")

    fig.set_size_inches(11.69,8.27)     # A4 paper size

    return fig, ax, lines_a

def ParseChoice(indices_or_x):
    assert(indices_or_x.count("x") == 1)
    cut_coord_indices = [[] for _ in range(len(indices_or_x)) ]
    for n,s in enumerate(indices_or_x):
        if s == "x":
            axis_dim = n
            continue
        comma_separated = s.split(",")
        for t in comma_separated:
            if ":" in t:
                colon_separated = t.split(":")
                start = int(colon_separated[0])
                end = int(colon_separated[1])
                for i in range(start,end+1):
                    cut_coord_indices[n].append(i)
            else:
                cut_coord_indices[n].append(int(t))
        
    return axis_dim, cut_coord_indices

def GetPlotProperties(d):
    leg_labels = [r"1 $A'$", r"2 $A'$", r"3 $A'$", r"1 $A''$"] 
    if d == 0:
        props = PlotProps(
                x_lim_min = -0.2 ,
                x_lim_max =  0.4 ,
                x_label = r"$x$" ,
                y_lim_min = -2.0 ,
                y_lim_max =  2.0 ,
                y_label = r"Energy / eV" ,
                legend_labels = leg_labels ,
                title = "",
                )
    elif d == 1:
        props = PlotProps(
                x_lim_min = -0.5 ,
                x_lim_max =  1.0 ,
                x_label = r"$y$" ,
                y_lim_min = -2.0 ,
                y_lim_max =  2.0 ,
                y_label = r"Energy / eV" ,
                legend_labels = leg_labels ,
                title = "",
                )
    elif d == 2:
        props = PlotProps(
                x_lim_min = 0.0 ,
                x_lim_max =  180.0 ,
                x_label = r"$\theta$ / ${}^{\circ}$" ,
                y_lim_min = -2.0 ,
                y_lim_max =  2.0 ,
                y_label = r"Energy / eV" ,
                legend_labels = leg_labels ,
                title = "",
                )
    return props

def main(argv):
    assert(len(argv)==5 or len(argv)==6)
    file_name = argv[1]
    axis_dim, cut_coord_indices = ParseChoice(argv[2:5])
    save_figure = False
    if argv[-1] == "-s":
        save_figure = True
    
    # Read reference energies
    e_ref = np.loadtxt('ref_en.txt')

    # Read data
    data = np.loadtxt(file_name, skiprows=1)
    with open(file_name, 'r') as f:
        header = f.readline()
    header_names = header.split()
    
    indices = data[:,0:3].astype(int)
    coords = data[:,3:6]
    dims = 3
    energies_cols = [6,7,8,9]
    energies_a = data[:,energies_cols]
    n_states = len(energies_cols)
    n_points = len(indices[:,0])
    
    # Transform coordinates
    r_NN_2 = 0.5945
    r0 = 1.56814067
    R0 = 5
    coords[:,0] = coords[:,0]+r_NN_2
    coords[:,0] = 1 - np.exp(-(coords[:,0]-r0)/r0)
    coords[:,1] = 1 - np.exp(-(coords[:,1]-R0)/R0)

    # Common plotting options
    coord_names = [r"$r$",r"$R$",r"$\theta$"]
    coord_units = [r"~\AA{}",r"~\AA{}",r"${}^{\circ}$"]

    props = GetPlotProperties(axis_dim)

    # Diabatization
    


    # Dimensions of the cut coordinates
    dims_cut = []
    for d in range(dims):
        if d != axis_dim :
            dims_cut.append(d)
   
    # List of all indices where we have a cut
    # First, count them
    n_cuts = 1
    n_cuts_per_dim = [[] for _ in range(dims)]
    for d in dims_cut:
        n_cuts_d = len(cut_coord_indices[d])
        n_cuts_per_dim[d] = n_cuts_d
        n_cuts *= n_cuts_d

    # Form the list: each sublist gets an index from each 
    cuts_list = []
    for i_cut in range(n_cuts):
        i_cut_rem = i_cut
        indices_single_cut = []
        for d in dims_cut:
            i_cut_d = i_cut_rem % n_cuts_per_dim[d]
            i_cut_rem = (i_cut_rem - i_cut_d) // n_cuts_per_dim[d]
            indices_single_cut.append(cut_coord_indices[d][i_cut_d])
        cuts_list.append(indices_single_cut)

    # For each cut stride, make a plot
    for indices_cut in cuts_list:
        select = np.full((n_points,1), True, dtype=bool)
        for d, i in zip(dims_cut, indices_cut):
            select = np.logical_and(select, (indices[:,d] == i)[:, np.newaxis])

        if np.any(select) is False:
            break

        x = coords[:,axis_dim][select.flatten()]
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        coords_cut = []
        for d in dims_cut:
            coords_cut.append(coords[:,d][select.flatten()][0])

        energies_a_cut = []
        energies_d_cut = []
        for state in range(n_states):
            en_i = (energies_a[:,state][select.flatten()]-e_ref)*SI.h2ev
            en_i = en_i[sort_idx]
            energies_a_cut.append(en_i)

        title = r"Scan \ce{[Rb + N2H]+} with "
        for n, d in enumerate(dims_cut):
            title += coord_names[d] + r" $= " + "{:5.3f}".format(coords_cut[n]) + r"$ " + coord_units[d] + ", "
        title = title[:-2]
        props.title = title

        fig, ax, lines = MakeFigure(x, energies_a_cut, [], props)

        if save_figure:
            saved_file_name = f"cut_{axis_dim}_"
            for d, i in zip(dims_cut, indices_cut):
                saved_file_name += f"{i}{chr(105 + d)}_" 
            saved_file_name = saved_file_name[:-1]
            saved_file_name += ".pdf"
            plt.savefig(saved_file_name)
        else:
            plt.show()
        plt.close(fig)


if __name__ == '__main__':
    main(sys.argv)
