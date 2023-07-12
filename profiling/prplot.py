#!/usr/bin/env python3

"""
Plotting the results of profiling run of the diabatools package
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
import ConstantsSI as SI
from diabtools.diabatizer import adiabatic

color_dark_cyc = cycler("color", [mpl.cm.get_cmap("tab20")(2*n) for n in range(10)])
color_light_cyc = cycler("color", [mpl.cm.get_cmap("tab20")(2*n+1) for n in range(10)])
color_alt_cyc = cycler("color", [mpl.cm.get_cmap("Accent")(n) for n in range(8)])
cmap_cyc = cycler("cmap", [mpl.cm.get_cmap("tab20")(2*n) for n in range(10)])

def compare_data_1d(x_data, y_data, W, title = None, fname = None):
    Npts, Ns = y_data.shape
    x = np.linspace(x_data.min(), x_data.max(), 10*x_data.size)

    Wx = W(x)
    Wxd = W(x_data)
    Wxd_diag = Wxd.diagonal(axis1=1,axis2=2)
    Vx, _ = adiabatic(Wx)
    Vxd, _ = adiabatic(Wxd)

    dV = np.abs(Vxd - y_data)
    dWV = np.zeros(y_data.shape)
    for i in range(Ns):
        dWV[:,i] = np.amin(np.abs(Wxd_diag[:,i,np.newaxis] - y_data), axis=1)

    fig, axs = plt.subplots(3, 1, height_ratios=[2,1,1], sharex=True)
    ax = axs[0]
    for i, sty_a, sty_d in zip(range(Ns), color_dark_cyc, color_light_cyc):
        ax.plot(x, Wx[:,i,i], label=(r"$W_{" f"{i}{i}" r"}$"), **sty_d, lw=2)
        ax.plot(x, Vx[:,i], label=(r"$V_{" f"{i}" r"}$"), **sty_a, ls="--")
        ax.scatter(x_data, y_data[:,i], label=(r"$E_{" f"{i}" r"}$"), **sty_a, marker="+")
    
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
            stems = ax.stem(x_data, dV[:,i], label=(r"$|E_{" f"{i}" r"} - V_{" f"{i}" r"}|$"))
            stems.markerline.set_color(sty["color"])
            stems.markerline.set_marker('+')
            stems.stemlines.set_color(sty["color"])
            stems.baseline.set_color("none")
    ax.axhline(color='k', lw=0.5, ls='--')
    ax.legend()

    ax = axs[2]
    for i, sty in zip(range(Ns), color_dark_cyc):
        if np.any(~np.isnan(dWV[:,i])):
            stems = ax.stem(x_data, dWV[:,i], label=(r"$\displaystyle\min_{i}|E_{i} - W_{" f"{i}{i}" r"}|$"))
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


def compare_data_2d(x_data, y_data, W, title = None, fname = None):
# def plot_2d2s_testVSfit(self,X,Y,Wt,Vt,W):
    Npts, Ns = y_data.shape
    _, Nd = x_data.shape

    x_by_dim = [None for _ in range(Nd)]
    for d in range(Nd):
        x_by_dim[d] = np.linspace(x_data[:,d].min(), x_data[:,d].max(), int(5*np.sqrt(Npts)))
    X_by_dim = tuple(np.meshgrid(*x_by_dim))

    breakpoint()
    x = np.vstack(tuple(Xd.ravel() for Xd in X_by_dim)).transpose()
    
    Wx = W(x)
    Wxd = W(x_data)
    Wxd_diag = Wxd.diagonal(axis1=1,axis2=2)
    Vx, _ = adiabatic(Wx)
    Vxd, _ = adiabatic(Wxd)

    dV = np.abs(Vxd - y_data)
    dWV = np.zeros(y_data.shape)
    for i in range(Ns):
        dWV[:,i] = np.amin(np.abs(Wxd_diag[:,i,np.newaxis] - y_data), axis=1)

    # Plot
    cl = np.linspace(-1,1,21)
    fig = plt.figure()
    ax = fig.add_subplot(2,2,1,projection='3d')
    for i, sty in zip(range(Ns):
        ax.plot_trisurf(x[:,0],x[:,1],Wx[:,i,i],cmap='Blues_r')
    # axs[0,0].tricontour(X,Y,W22_t,levels=cl,cmap='Reds_r')
    # axs[0,0].set_ylabel("$y$")
    # axs[0,1].tricontour(X,Y,W21_t,levels=21,cmap='BrBG')
    # axs[0,2].tricontour(X,Y,V1_t,levels=cl,cmap='Greens')
    # axs[0,2].tricontour(X,Y,V2_t,levels=cl,cmap='Oranges_r')

    # axs[1,0].tricontour(X,Y,W11,levels=cl,cmap='Blues_r')
    # axs[1,0].tricontour(X,Y,W22,levels=cl,cmap='Reds_r')
    # axs[1,0].set_ylabel("$y$")
    # axs[1,1].tricontour(X,Y,W21,levels=21,cmap='BrBG')
    # axs[1,2].tricontour(X,Y,V1,levels=cl,cmap='Greens')
    # axs[1,2].tricontour(X,Y,V2,levels=cl,cmap='Oranges_r')

    # cs = axs[2,0].contour(X,Y,ldW11,levels=10,cmap='Blues_r')
    # fig.colorbar(cs, ax=axs[2,0], location='right')
    # cs = axs[2,0].contour(X,Y,ldW22,levels=10,cmap='Reds_r')
    # fig.colorbar(cs, ax=axs[2,0], location='right')
    # axs[2,0].set_xlabel("$x$")
    # axs[2,0].set_ylabel("$y$")
    # cs = axs[2,1].contour(X,Y,ldW21,levels=10,cmap='BrBG')
    # fig.colorbar(cs, ax=axs[2,1], location='right')
    # axs[2,1].set_xlabel("$x$")
    # cs = axs[2,2].contour(X,Y,ldV1,levels=10,cmap='Greens_r')
    # fig.colorbar(cs, ax=axs[2,2], location='right')
    # cs = axs[2,2].contour(X,Y,ldV2,levels=10,cmap='Oranges_r')
    # fig.colorbar(cs, ax=axs[2,2], location='right')
    # axs[2,2].set_xlabel("$x$")

    # ax1 = plt.figure().add_subplot(projection='3d')
    # ax1.plot_wireframe(X,Y,W11_t,alpha=0.5,color='darkblue', lw=0.5)
    # ax1.plot_surface(X,Y,W11,alpha=0.5,cmap='Blues_r')
    # ax1.contour(X,Y,W11,alpha=0.5, levels=cl,colors='b')
    # ax1.plot_wireframe(X,Y,W22_t,alpha=0.5,color='darkred', lw=0.5)
    # ax1.plot_surface(X,Y,W22,alpha=0.5,cmap='Reds_r')
    # ax1.contour(X,Y,W22,alpha=0.5, levels=cl,colors='r')
    # ax1.set_zlim(ax1.get_zlim()[0], cl[-1])
    # ax1.set_xlabel("$x$")
    # ax1.set_ylabel("$y$")

    # ax2 = plt.figure().add_subplot(projection='3d')
    # ax2.plot_wireframe(X,Y,W21_t,alpha=0.5,color='k', lw=0.5)
    # ax2.plot_surface(X,Y,W21,alpha=0.5,cmap='BrBG')
    # ax2.contour(X,Y,W21,alpha=0.5, levels=21, colors='indigo')
    # ax2.set_xlabel("$x$")
    # ax2.set_ylabel("$y$")

    # ax3 = plt.figure().add_subplot(projection='3d')
    # ax3.plot_wireframe(X,Y,V1_t,alpha=0.5,color='darkgreen', lw=0.5)
    # ax3.plot_surface(X,Y,V1,alpha=0.5,cmap='Greens')
    # ax3.contour(X,Y,V1,alpha=0.5, levels=cl, colors='g')
    # ax3.plot_wireframe(X,Y,V2_t,alpha=0.5,color='darkorange', lw=0.5)
    # ax3.plot_surface(X,Y,V2,alpha=0.5,cmap='Oranges_r')
    # ax3.contour(X,Y,V2,alpha=0.5, levels=cl, colors='orange')
    # ax3.set_zlim(ax3.get_zlim()[0], cl[-1])
    # ax3.set_xlabel("$x$")
    # ax3.set_ylabel("$y$")
    plt.show()
