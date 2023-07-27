#!/usr/bin/env python3

"""
Plotting the results of profiling run of the diabatools package
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmasher as cmr
from cycler import cycler
import ConstantsSI as SI
from diabtools.diabatizer import adiabatic

color_dark_cyc = cycler("color", [mpl.colormaps.get_cmap("tab20")(2*n) for n in range(10)])
color_light_cyc = cycler("color", [mpl.colormaps.get_cmap("tab20")(2*n+1) for n in range(10)])
color_alt_cyc = cycler("color", [mpl.colormaps.get_cmap("Accent")(n) for n in range(8)])
cmap_cyc1 = cycler("cmap", [
    cmr.get_sub_cmap("Blues_r", 0.0, 0.7),
    cmr.get_sub_cmap("Reds_r", 0.0, 0.7),
    cmr.get_sub_cmap("Greens_r", 0.0, 0.7),
    ])
cmap_cyc2 = cycler("cmap", [
    cmr.get_sub_cmap("Purples_r", 0.0, 0.7),
    cmr.get_sub_cmap("Oranges_r", 0.0, 0.7),
    cmr.get_sub_cmap("GnBu", 0.0, 1.0),
    ])
cmap_cyc3 = cycler("cmap", ["bwr", "PuOr", "BrBG_r", "PiYG"])
cmap_cyc4 = cycler("cmap", ["spring", "summer", "autumn", "winter"])

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def compare_data_1d(x_data, V_data, W, title = None, fname = None):
    Npts, Ns = V_data.shape
    x = np.linspace(x_data.min(), x_data.max(), 10*x_data.size)

    Wx = W(x)
    Wxd = W(x_data)
    Wxd_diag = Wxd.diagonal(axis1=1,axis2=2)
    Vx, _ = adiabatic(Wx)
    Vxd, _ = adiabatic(Wxd)

    dV = np.abs(Vxd - V_data)
    dWV = np.zeros(V_data.shape)
    for i in range(Ns):
        dWV[:,i] = np.amin(np.abs(Wxd_diag[:,i,np.newaxis] - V_data), axis=1)

    fig, axs = plt.subplots(3, 1, height_ratios=[2,1,1], sharex=True)
    ax = axs[0]
    for i, sty_a, sty_d in zip(range(Ns), color_dark_cyc, color_light_cyc):
        ax.plot(x, Wx[:,i,i], label=(r"$W_{" f"{i}{i}" r"}$"), **sty_d, lw=2)
        ax.plot(x, Vx[:,i], label=(r"$V_{" f"{i}" r"}$"), **sty_a, ls="--")
        ax.scatter(x_data, V_data[:,i], label=(r"$E_{" f"{i}" r"}$"), **sty_a, marker="+")

    iter_color_alt_cyc = iter(color_alt_cyc)
    for i in range(1,Ns):
        for j in range(i):
            sty = next(iter_color_alt_cyc)
            ax.plot(x, Wx[:,i,j], label=(r"$V_{" f"{i}{j}" r"}$"), **sty)

    ax.set_title(title)
    ax.grid(True)
    ax.set_ylim(-1,1)
    ax.legend()

    ax = axs[1]
    for i, sty in zip(range(Ns), color_dark_cyc):
        if np.any(~np.isnan(dV[:,i])):
            stems = ax.stem(x_data, dV[:,i],
                label=r"$|E_{" f"{i}" r"} - V_{" f"{i}" r"}|$")
            stems.markerline.set_color(sty["color"])
            stems.markerline.set_marker('+')
            stems.stemlines.set_color(sty["color"])
            stems.baseline.set_color("none")
    ax.axhline(color='k', lw=0.5, ls='--')
    ax.legend()

    ax = axs[2]
    for i, sty in zip(range(Ns), color_dark_cyc):
        if np.any(~np.isnan(dWV[:,i])):
            stems = ax.stem(x_data, dWV[:,i],
                label=r"$\displaystyle\min_{i}|E_{i} - W_{" f"{i}{i}" r"}|$")
            stems.markerline.set_color(sty["color"])
            stems.markerline.set_marker('+')
            stems.stemlines.set_color(sty["color"])
            stems.baseline.set_color("none")
    ax.axhline(color='k', lw=0.5, ls='--')
    ax.legend()

    if fname:
        fig.savefig(fname)
    else:
        plt.show()


def compare_data_2d(x_data, V_data, W, title = None, fname = None):
    Npts, Ns = V_data.shape
    _, Nd = x_data.shape
    x = x_data[:,0]
    y = x_data[:,1]

    Wx = W(x_data)
    Wx_diag = Wx.diagonal(axis1=1,axis2=2)
    Vx, _ = adiabatic(Wx)

    dV = np.abs(Vx - V_data)
    dWV = np.zeros(V_data.shape)
    for i in range(Ns):
        dWV[:,i] = np.amin(np.abs(Wx_diag[:,i,np.newaxis] - V_data), axis=1)

    # Plot
    cl = 21
    fig, axs = plt.subplots(3,2,sharex=True,sharey=False)
    fig.suptitle(title)

    for s, sty in zip(range(Ns), cmap_cyc1):
        cs = axs[0,0].tricontour(x,y,Wx_diag[:,s],levels=cl,**sty)
        fig.colorbar(cs, ax=axs[0,0], location='right')
    axs[0,0].set_ylabel("$y$")
    axs[0,0].set_title("Diabatic (fit)")

    for s, sty in zip(range(Ns), cmap_cyc2):
        cs = axs[0,1].tricontour(x,y,Vx[:,s],levels=cl,**sty)
        fig.colorbar(cs, ax=axs[0,1], location='right')
    axs[0,1].set_title("Adiabatic (fit)")

    icyc = iter(cmap_cyc3)
    for i in range(1,Ns):
        for j in range(Ns*(Ns-1)//2):
            sty = next(icyc)
            cs = axs[1,0].tricontour(x,y,Wx[:,i,j],levels=21,**sty)
            fig.colorbar(cs, ax=axs[1,0], location='right')
    axs[1,0].set_ylabel("$y$")
    axs[1,0].set_title("Couplings (fit)")

    for s, sty in zip(range(Ns), cmap_cyc2):
        cs = axs[1,1].tricontour(x,y,V_data[:,s],levels=cl,**sty)
        fig.colorbar(cs, ax=axs[1,1], location='right')
    axs[1,1].set_title("Adiabatic (orig.)")

    for s, sty in zip(range(Ns), cmap_cyc1):
        cs = axs[2,0].tricontour(x,y,dWV[:,s],levels=cl,**sty)
        fig.colorbar(cs, ax=axs[2,0], location='right')
    axs[2,0].set_xlabel("$x$")
    axs[2,0].set_ylabel("$y$")
    axs[2,0].set_title("Least diff. Dia vs. Adia.")

    for s, sty in zip(range(Ns), cmap_cyc4):
        cs = axs[2,1].tricontour(x,y,dV[:,s],levels=21,**sty)
        fig.colorbar(cs, ax=axs[2,1], location='right')
    axs[2,1].set_xlabel("$x$")
    axs[2,1].set_title("Diff. Adiabatic")

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1,projection='3d')
    for i, sty in zip(range(Ns), cmap_cyc1):
        ax.plot_trisurf(x, y, Wx_diag[:,i], **sty, alpha=0.5)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1,projection='3d')
    icyc = iter(cmap_cyc3)
    for i in range(1,Ns):
        for j in range(Ns*(Ns-1)//2):
            sty = next(icyc)
        ax.plot_trisurf(x, y, Wx[:,i,j], **sty)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1,projection='3d')
    for i in range(Ns):
        ax.scatter(x, y, V_data[:,i])

    plt.show()
