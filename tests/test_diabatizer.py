import os
from ..diabtools import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

class TestDiabatizer:
    def test_LiF(self, pytestconfig):
        import ConstantsSI as SI
        import matplotlib.pyplot as plt
        from cycler import cycler
        from itertools import cycle
        E_Li_p = 1.8478136
        r0_LiF_gs = 1.563864

        # Load data
        filename = 'tests/lif_mr_mscaspt2.csv'
        data = np.genfromtxt(filename,delimiter=',',skip_header=4)
        r = data[:,0]
        Npts = r.size
        en = data[:,5:]
        en = (en - en[-1,0])*SI.h2ev

        # Transform coordinate
        r_x = 6.5   # Intersection point
        stretch_factor = 1/r_x
        x = 1 - np.exp(-stretch_factor*(r-r_x))

        # Initial guess
        W = SymPolyMat(2,1)
        W[0,0] = NdPoly({(0,): 0.1, (1,): 1, (2,): -1, (3,): 1, (4,): 1, (5,): 1})
        W[1,1] = NdPoly({(0,): 0.1, (1,): -1, (2,): 1, (3,): 1, (4,): 1, (5,): 1, (6,): 1, (7,): 1})
        W[1,0] = NdPoly({(0,): 0.0})

        # Diabatize
        lif = SingleDiabatizer(2,1, verbosity=2)
        lif.Wguess = W
        lif.addDomain(x,en)
        lif.optimize()
        
        if pytestconfig.getoption("verbose") > 0:
            # Make plot using resulting parameters
            r_test = np.logspace(np.log10(2), np.log10(10), 100)
            x_test = 1 - np.exp(-stretch_factor*(r_test-r_x))
            # x_test = x_test[:,np.newaxis]
            yd1_test = lif.Wout[0,0](x_test)
            yd2_test = lif.Wout[1,1](x_test)
            yc_test = lif.Wout[1,0](x_test)
            ya1_test = adiabatic2(yd1_test, yd2_test, yc_test, -1)
            ya2_test = adiabatic2(yd1_test, yd2_test, yc_test, +1)
            
            fig, axs = plt.subplots(2,1)
            _, nplots = en.shape
            labels = [
                    r'$E_1$ XMSCASPT2',
                    r'$E_2$ XMSCASPT2',
                    r'$E_1$ fit',
                    r'$E_2$ fit',
                    r'$V_1$',
                    r'$V_2$',
                    r'$V_{12}$',
                    ]
            mark_cycle = cycler(marker=['+','x'])

            ax = axs[0]
            for n, mc in zip(range(nplots),cycle(mark_cycle)):
                ax.plot(x,en[:,n],linewidth=0.5, **mc)

            ax.plot(x_test, ya1_test)
            ax.plot(x_test, ya2_test)
            ax.plot(x_test, yd1_test)
            ax.plot(x_test, yd2_test)
            ax.plot(x_test, yc_test)
            ax.set_xlabel(r'$x$')
            ax.set_ylabel(r'$E$ / hartree')
            ax.set_title(r'\ce{LiF} avoided crossing : diabatization of XMS-CASPT2 potentials')
            ax.grid(True)

            ax = axs[1]
            for n, mc in zip(range(nplots),cycle(mark_cycle)):
                ax.plot(r,en[:,n],linewidth=0.5, **mc)

            ax.plot(r_test, ya1_test)
            ax.plot(r_test, ya2_test)
            ax.plot(r_test, yd1_test)
            ax.plot(r_test, yd2_test)
            ax.plot(r_test, yc_test)
            ax.set_xlabel(r'$r_{\ce{LiF}}$ / \AA')
            ax.set_ylabel(r'$E$ / hartree')
            ax.legend(labels)
            ax.grid(True)

            fig.set_size_inches((12,9))
            # plt.savefig('lif.pdf')
            plt.show()

    def plot_1d2s_testVSfit(self,x,Wt,Vt,W):
        import matplotlib.pyplot as plt
        # Reshape
        W11_t = Wt[:,0,0].reshape(x.shape)
        W22_t = Wt[:,1,1].reshape(x.shape)
        W21_t = np.abs(Wt[:,1,0].reshape(x.shape))
        V1_t = Vt[:,0].reshape(x.shape)
        V2_t = Vt[:,1].reshape(x.shape)

        W11 = W[:,0,0].reshape(x.shape)
        W22 = W[:,1,1].reshape(x.shape)
        W21 = np.abs(W[:,1,0].reshape(x.shape))
        V1 = adiabatic(W)[0][:,0].reshape(x.shape)
        V2 = adiabatic(W)[0][:,1].reshape(x.shape)

        # Calculate logarithms of abs(differences)
        dW11 = np.abs(W11-W11_t)
        dW22 = np.abs(W22-W22_t)
        dW21 = np.abs(W21-W21_t)
        dV1 = np.abs(V1-V1_t)
        dV2 = np.abs(V2-V2_t)
        dW11[dW11==0] = np.nan
        dW22[dW22==0] = np.nan
        dW21[dW21==0] = np.nan
        dV1[dV1==0] = np.nan
        dV2[dV2==0] = np.nan
       
        ldW11 = np.log10(dW11)
        ldW22 = np.log10(dW22)
        ldW21 = np.log10(dW21)
        ldV1  = np.log10(dV1)
        ldV2  = np.log10(dV2)

        # Plot
        cl = np.linspace(-1,1,21)
        fig, axs = plt.subplots(3,3)
        axs[0,0].plot(x,W11_t,color='b')      
        axs[0,0].plot(x,W22_t,color='r')      
        axs[0,1].plot(x,W21_t,color='magenta')
        axs[0,2].plot(x,V1_t, color='lime')    
        axs[0,2].plot(x,V2_t, color='orange')  

        axs[0,0].set_ylabel(r"$W_{ii}$ (orig)")
        axs[0,1].set_ylabel(r"$W_{12}$ (orig)")
        axs[0,2].set_ylabel(r"$V_{i}$ (orig)")

        axs[1,0].plot(x,W11,color='b')      
        axs[1,0].plot(x,W22,color='r')      
        axs[1,1].plot(x,W21,color='magenta')
        axs[1,2].plot(x,V1, color='lime')    
        axs[1,2].plot(x,V2, color='orange')  

        axs[1,0].set_ylabel(r"$W_{ii}$ (fit)")
        axs[1,1].set_ylabel(r"$W_{12}$ (fit)")
        axs[1,2].set_ylabel(r"$V_{i}$ (fit)")

        axs[2,0].plot(x,ldW11,color='b')
        axs[2,0].plot(x,ldW22,color='r')
        axs[2,1].plot(x,ldW21,color='magenta')
        axs[2,2].plot(x,ldV1, color='lime')
        axs[2,2].plot(x,ldV2, color='orange')
        axs[2,0].set_ylabel(r"$\log\Delta W_{ii}$")
        axs[2,1].set_ylabel(r"$\log\Delta W_{12}$")
        axs[2,2].set_ylabel(r"$\log\Delta V_{i}$")
        axs[2,0].set_xlabel(r"$x$")
        axs[2,1].set_xlabel(r"$x$")
        axs[2,2].set_xlabel(r"$x$")

        plt.show()

    def plot_2d2s_testVSfit(self,X,Y,Wt,Vt,W):
        import matplotlib.pyplot as plt
        # Reshape
        W11_t = Wt[:,0,0].reshape(X.shape)
        W22_t = Wt[:,1,1].reshape(X.shape)
        W21_t = np.abs(Wt[:,1,0].reshape(X.shape))
        V1_t = Vt[:,0].reshape(X.shape)
        V2_t = Vt[:,1].reshape(X.shape)

        W11 = W[:,0,0].reshape(X.shape)
        W22 = W[:,1,1].reshape(X.shape)
        W21 = np.abs(W[:,1,0].reshape(X.shape))
        V1 = adiabatic(W)[0][:,0].reshape(X.shape)
        V2 = adiabatic(W)[0][:,1].reshape(X.shape)

        # Calculate logarithms of abs(differences)
        dW11 = np.abs(W11-W11_t)
        dW22 = np.abs(W22-W22_t)
        dW21 = np.abs(W21-W21_t)
        dV1 = np.abs(V1-V1_t)
        dV2 = np.abs(V2-V2_t)
        dW11[dW11==0] = np.nan
        dW22[dW22==0] = np.nan
        dW21[dW21==0] = np.nan
        dV1[dV1==0] = np.nan
        dV2[dV2==0] = np.nan
       
        ldW11 = np.log10(dW11)
        ldW22 = np.log10(dW22)
        ldW21 = np.log10(dW21)
        ldV1  = np.log10(dV1)
        ldV2  = np.log10(dV2)

        # Plot
        cl = np.linspace(-1,1,21)
        fig, axs = plt.subplots(3,3)
        axs[0,0].contour(X,Y,W11_t,levels=cl,cmap='Blues_r')
        axs[0,0].contour(X,Y,W22_t,levels=cl,cmap='Reds_r')
        axs[0,0].set_ylabel("$y$")
        axs[0,1].contour(X,Y,W21_t,levels=21,cmap='BrBG')
        axs[0,2].contour(X,Y,V1_t,levels=cl,cmap='Greens')
        axs[0,2].contour(X,Y,V2_t,levels=cl,cmap='Oranges_r')

        axs[1,0].contour(X,Y,W11,levels=cl,cmap='Blues_r')
        axs[1,0].contour(X,Y,W22,levels=cl,cmap='Reds_r')
        axs[1,0].set_ylabel("$y$")
        axs[1,1].contour(X,Y,W21,levels=21,cmap='BrBG')
        axs[1,2].contour(X,Y,V1,levels=cl,cmap='Greens')
        axs[1,2].contour(X,Y,V2,levels=cl,cmap='Oranges_r')

        cs = axs[2,0].contour(X,Y,ldW11,levels=10,cmap='Blues_r')
        fig.colorbar(cs, ax=axs[2,0], location='right')
        cs = axs[2,0].contour(X,Y,ldW22,levels=10,cmap='Reds_r')
        fig.colorbar(cs, ax=axs[2,0], location='right')
        axs[2,0].set_xlabel("$x$")
        axs[2,0].set_ylabel("$y$")
        cs = axs[2,1].contour(X,Y,ldW21,levels=10,cmap='BrBG')
        fig.colorbar(cs, ax=axs[2,1], location='right')
        axs[2,1].set_xlabel("$x$")
        cs = axs[2,2].contour(X,Y,ldV1,levels=10,cmap='Greens_r')
        fig.colorbar(cs, ax=axs[2,2], location='right')
        cs = axs[2,2].contour(X,Y,ldV2,levels=10,cmap='Oranges_r')
        fig.colorbar(cs, ax=axs[2,2], location='right')
        axs[2,2].set_xlabel("$x$")

        ax1 = plt.figure().add_subplot(projection='3d')
        ax1.plot_wireframe(X,Y,W11_t,alpha=0.5,color='darkblue', lw=0.5)
        ax1.plot_surface(X,Y,W11,alpha=0.5,cmap='Blues_r')
        ax1.contour(X,Y,W11,alpha=0.5, levels=cl,colors='b')
        ax1.plot_wireframe(X,Y,W22_t,alpha=0.5,color='darkred', lw=0.5)
        ax1.plot_surface(X,Y,W22,alpha=0.5,cmap='Reds_r')
        ax1.contour(X,Y,W22,alpha=0.5, levels=cl,colors='r')
        ax1.set_zlim(ax1.get_zlim()[0], cl[-1])
        ax1.set_xlabel("$x$")
        ax1.set_ylabel("$y$")

        ax2 = plt.figure().add_subplot(projection='3d')
        ax2.plot_wireframe(X,Y,W21_t,alpha=0.5,color='k', lw=0.5)
        ax2.plot_surface(X,Y,W21,alpha=0.5,cmap='BrBG')
        ax2.contour(X,Y,W21,alpha=0.5, levels=21, colors='indigo')
        ax2.set_xlabel("$x$")
        ax2.set_ylabel("$y$")

        ax3 = plt.figure().add_subplot(projection='3d')
        ax3.plot_wireframe(X,Y,V1_t,alpha=0.5,color='darkgreen', lw=0.5)
        ax3.plot_surface(X,Y,V1,alpha=0.5,cmap='Greens')
        ax3.contour(X,Y,V1,alpha=0.5, levels=cl, colors='g')
        ax3.plot_wireframe(X,Y,V2_t,alpha=0.5,color='darkorange', lw=0.5)
        ax3.plot_surface(X,Y,V2,alpha=0.5,cmap='Oranges_r')
        ax3.contour(X,Y,V2,alpha=0.5, levels=cl, colors='orange')
        ax3.set_zlim(ax3.get_zlim()[0], cl[-1])
        ax3.set_xlabel("$x$")
        ax3.set_ylabel("$y$")
        plt.show()

    def plot_2d3s_testVSfit(self,X,Y,Wt,Vt,W):
        import matplotlib.pyplot as plt
        # Reshape
        W11_t = Wt[:,0,0].reshape(X.shape)
        W22_t = Wt[:,1,1].reshape(X.shape)
        W33_t = Wt[:,2,2].reshape(X.shape)
        W21_t = np.abs(Wt[:,1,0].reshape(X.shape))
        W31_t = np.abs(Wt[:,2,0].reshape(X.shape))
        W32_t = np.abs(Wt[:,2,1].reshape(X.shape))
        V1_t = Vt[:,0].reshape(X.shape)
        V2_t = Vt[:,1].reshape(X.shape)
        V3_t = Vt[:,2].reshape(X.shape)

        W11 = W[:,0,0].reshape(X.shape)
        W22 = W[:,1,1].reshape(X.shape)
        W33 = W[:,2,2].reshape(X.shape)
        W21 = np.abs(W[:,1,0].reshape(X.shape))
        W31 = np.abs(W[:,2,0].reshape(X.shape))
        W32 = np.abs(W[:,2,1].reshape(X.shape))
        V1 = adiabatic(W)[0][:,0].reshape(X.shape)
        V2 = adiabatic(W)[0][:,1].reshape(X.shape)
        V3 = adiabatic(W)[0][:,2].reshape(X.shape)

        # Calculate logarithms of abs(differences)
        dW11 = np.abs(W11-W11_t)
        dW22 = np.abs(W22-W22_t)
        dW33 = np.abs(W33-W33_t)
        dW21 = np.abs(W21-W21_t)
        dW31 = np.abs(W31-W31_t)
        dW32 = np.abs(W32-W32_t)
        dV1 = np.abs(V1-V1_t)
        dV2 = np.abs(V2-V2_t)
        dV3 = np.abs(V3-V3_t)
        dW11[dW11==0] = np.nan
        dW22[dW22==0] = np.nan
        dW33[dW33==0] = np.nan
        dW21[dW21==0] = np.nan
        dW31[dW31==0] = np.nan
        dW32[dW32==0] = np.nan
        dV1[dV1==0] = np.nan
        dV2[dV2==0] = np.nan
        dV3[dV3==0] = np.nan
       
        ldW11 = np.log10(dW11)
        ldW22 = np.log10(dW22)
        ldW33 = np.log10(dW33)
        ldW21 = np.log10(dW21)
        ldW31 = np.log10(dW31)
        ldW32 = np.log10(dW32)
        ldV1  = np.log10(dV1)
        ldV2  = np.log10(dV2)
        ldV3  = np.log10(dV3)

        # Plot
        cmapW11='Blues_r'
        cmapW22='Reds_r'
        cmapW33='Greens_r'
        cmapW21='bwr'
        cmapW31='BrBG'
        cmapW32='PiYG'
        cmapV1='Purples'
        cmapV2='plasma'
        cmapV3='Oranges_r'

        # Collect
        Wii_t = [W11_t, W22_t, W33_t]
        Wij_t = [W21_t, W31_t, W32_t]
        V_t = [V1_t, V2_t, V3_t]
        pots_t = Wii_t + Wij_t + V_t
        Wii = [W11, W22, W33]
        Wij = [W21, W31, W32]
        V = [V1, V2, V3]
        pots = Wii + Wij + V
        ldWii = [ldW11, ldW22, ldW33]
        ldWij = [ldW21, ldW31, ldW32]
        ldV = [ldV1, ldV2, ldV3]
        log_diffs = ldWii + ldWij + ldV
        cmapsWii = [cmapW11, cmapW22, cmapW33]
        cmapsWij = [cmapW21, cmapW31, cmapW32]
        cmapsV = [cmapV1, cmapV2, cmapV3]
        cmaps = cmapsWii + cmapsWij + cmapsV

        cl = np.linspace(-1,1.5,21)
        fig, axs = plt.subplots(3,3)
        axs[0,0].contour(X,Y,W11_t,levels=cl,cmap=cmapW11)
        axs[0,0].contour(X,Y,W22_t,levels=cl,cmap=cmapW22)
        axs[0,0].contour(X,Y,W33_t,levels=cl,cmap=cmapW33)
        axs[0,1].contour(X,Y,W21_t,levels=21,cmap=cmapW21)
        axs[0,1].contour(X,Y,W31_t,levels=21,cmap=cmapW31)
        axs[0,1].contour(X,Y,W32_t,levels=21,cmap=cmapW32)
        axs[0,2].contour(X,Y,V1_t,levels=cl,cmap=cmapV1)
        axs[0,2].contour(X,Y,V2_t,levels=cl,cmap=cmapV2)
        axs[0,2].contour(X,Y,V3_t,levels=cl,cmap=cmapV3)

        axs[1,0].contour(X,Y,W11,levels=cl,cmap=cmapW11)
        axs[1,0].contour(X,Y,W22,levels=cl,cmap=cmapW22)
        axs[1,0].contour(X,Y,W33,levels=cl,cmap=cmapW33)
        axs[1,1].contour(X,Y,W21,levels=21,cmap=cmapW21)
        axs[1,1].contour(X,Y,W31,levels=21,cmap=cmapW31)
        axs[1,1].contour(X,Y,W32,levels=21,cmap=cmapW32)
        axs[1,2].contour(X,Y,V1,levels=cl,cmap=cmapV1)
        axs[1,2].contour(X,Y,V2,levels=cl,cmap=cmapV2)
        axs[1,2].contour(X,Y,V3,levels=cl,cmap=cmapV3)

        for n, pair in enumerate(zip(log_diffs, cmaps)):
            ld, cm = pair
            if not np.all(np.isnan(ld)):
                cs = axs[2,n//3].contour(X,Y,ld,levels=10,cmap=cm)
                fig.colorbar(cs, ax=axs[2,n//3], location='right')

        axs[0,0].set_ylabel("$y$")
        axs[1,0].set_ylabel("$y$")
        axs[2,0].set_xlabel("$x$")
        axs[2,0].set_ylabel("$y$")
        axs[2,1].set_xlabel("$x$")
        axs[2,2].set_xlabel("$x$")

        ax1 = plt.figure().add_subplot(projection='3d')
        ax1.plot_wireframe(X,Y,W11_t,alpha=0.5,color='darkblue', lw=0.5)
        ax1.plot_surface(X,Y,W11,alpha=0.5,cmap=cmapW11)
        ax1.contour(X,Y,W11,alpha=0.5, levels=cl,colors='b')
        ax1.plot_wireframe(X,Y,W22_t,alpha=0.5,color='darkred', lw=0.5)
        ax1.plot_surface(X,Y,W22,alpha=0.5,cmap=cmapW22)
        ax1.contour(X,Y,W22,alpha=0.5, levels=cl,colors='r')
        ax1.plot_wireframe(X,Y,W33_t,alpha=0.5,color='darkgreen', lw=0.5)
        ax1.plot_surface(X,Y,W33,alpha=0.5,cmap=cmapW33)
        ax1.contour(X,Y,W33,alpha=0.5, levels=cl,colors='g')
        ax1.set_zlim(ax1.get_zlim()[0], cl[-1])
        ax1.set_xlabel("$x$")
        ax1.set_ylabel("$y$")

        ax2 = plt.figure().add_subplot(projection='3d')
        ax2.plot_wireframe(X,Y,W21_t,alpha=0.5,color='k', lw=0.5)
        ax2.plot_surface(X,Y,W21,alpha=0.5,cmap=cmapW21)
        ax2.contour(X,Y,W21,alpha=0.5, levels=21, colors='purple')
        ax2.plot_wireframe(X,Y,W31_t,alpha=0.5,color='k', lw=0.5)
        ax2.plot_surface(X,Y,W31,alpha=0.5,cmap=cmapW31)
        ax2.contour(X,Y,W31,alpha=0.5, levels=21, colors='teal')
        ax2.plot_wireframe(X,Y,W32_t,alpha=0.5,color='k', lw=0.5)
        ax2.plot_surface(X,Y,W32,alpha=0.5,cmap=cmapW32)
        ax2.contour(X,Y,W32,alpha=0.5, levels=21, colors='olive')
        ax2.set_xlabel("$x$")
        ax2.set_ylabel("$y$")

        ax3 = plt.figure().add_subplot(projection='3d')
        ax3.plot_wireframe(X,Y,V1_t,alpha=0.5,color='indigo', lw=0.5)
        ax3.plot_surface(X,Y,V1,alpha=0.5,cmap=cmapV1)
        ax3.contour(X,Y,V1,alpha=0.5, levels=cl, colors='purple')
        ax3.plot_wireframe(X,Y,V2_t,alpha=0.5,color='crimson', lw=0.5)
        ax3.plot_surface(X,Y,V2,alpha=0.5,cmap=cmapV2)
        ax3.contour(X,Y,V2,alpha=0.5, levels=cl, colors='deeppink')
        ax3.plot_wireframe(X,Y,V3_t,alpha=0.5,color='darkorange', lw=0.5)
        ax3.plot_surface(X,Y,V3,alpha=0.5,cmap=cmapV3)
        ax3.contour(X,Y,V3,alpha=0.5, levels=cl, colors='orange')
        ax3.set_zlim(ax3.get_zlim()[0], cl[-1])
        ax3.set_xlabel("$x$")
        ax3.set_ylabel("$y$")
        plt.show()

    def W_JTCI_2d2s(self):
        dx1 = 1
        dx2 = -1
        W = SymPolyMat(2,2)
        W[0,0] = NdPoly({(2,0): 0.5, (0,2): 0.5,})
        W[0,0][(0,0)] = W[0,0][(2,0)]*dx1**2 #- W[0,0][(1,0)]*dx1 
        W[0,0][(1,0)] = -2*W[0,0][(2,0)]*dx1
        W[1,1] = copy(W[0,0])
        W[1,1][(0,0)] = W[0,0][(2,0)]*dx2**2 #- W[0,0][(1,0)]*dx2 
        W[1,1][(1,0)] = -2*W[0,0][(2,0)]*dx2
        W[0,1] = NdPoly({(0,1): 5E-1})
        return W

    def W_infinityCI_2d2s(self):
        W = SymPolyMat(2,2)
        W[0,0] = NdPoly({
            (0,0): 0,
            (1,0): -1,
            (0,1): 0.2,
            (2,0): 0.3,
            (1,1): 0.1,
            (0,2): 0,
            (3,0): 0.1,
            (2,1): 0,
            (1,2): 0,
            (0,3): 0,
        })
        W[1,1] = NdPoly({
            (0,0): 0,
            (1,0): 0.5,
            (0,1): 0,
            (2,0): 1.5,
            (1,1): 0,
            (0,2): -0.2,
        })
        W[0,1] = NdPoly({(0,1): 5E-2})
        return W

    def W_ho_2d3s(self):
        dx1 = -1
        dx2 = 0
        dx3 = 1
        W = SymPolyMat.zero(3,2)
        W[0,0] = NdPoly({(2,0): 0.1, (0,2): 0.1,})
        W[1,1] = NdPoly({(0,0): 0.5, (1,0): -0.5})
        W[2,2] = NdPoly({(2,0): 0.1, (0,2): 0.1,})
        W[0,0][(0,0)] = W[0,0][(2,0)]*dx1**2 
        W[0,0][(1,0)] = -2*W[0,0][(2,0)]*dx1
        W[2,2][(0,0)] = W[0,0][(2,0)]*dx2**2 + 1
        W[2,2][(1,0)] = -2*W[0,0][(2,0)]*dx2
        W[0,1] = NdPoly({(0,1): 5E-1})
        W[2,1] = NdPoly({(0,1): 1E-1})
        W[2,0] = NdPoly({(0,1): 2E-1})
        return W

    def test_2d2s_infinity(self, pytestconfig):
        """ Two states, two dimensions.
        1: Construct an arbitrary diabatic potential matrix W_test
        2: Diagonalize it in order to get the adiabatic PESs
        3: Fit a new W_fit to the adiabatic data and check if
           W_fit == W_test
        """

        # 0: Prepare coordinates
        x1, x2, y1, y2 = (-1,1,0,1)
        Nx, Ny = (21,21)
        x = np.linspace(x1,x2,Nx)
        y = np.linspace(y1,y2,Ny)

        X, Y = np.mgrid[x1:x2:Nx*1j, y1:y2:Ny*1j]
        x0, y0 = (0,1)
        X = X - x0
        Y = Y - y0

        x_data = np.hstack((
            X.flatten()[:,np.newaxis],
            Y.flatten()[:,np.newaxis],
            ))

        # 1: Make target diabatic surfaces
        W_test = self.W_infinityCI_2d2s()
        W_test_x = W_test(x_data)

        # 2: Make test adiabatic surfaces
        W11_t = W_test_x[:,0,0]
        W22_t = W_test_x[:,1,1]
        W21_t = W_test_x[:,1,0]

        m = 0.5*(W11_t + W22_t)
        d = 0.5*(W11_t - W22_t)
        c = W21_t

        V_t = np.zeros((x_data.shape[0], 2))
        V_t[:,0] = m - np.sqrt(d**2 + c**2)
        V_t[:,1] = m + np.sqrt(d**2 + c**2)
        
        # 3: Fit diabatize test adiabatic surfaces
        W_guess = SymPolyMat.zero_like(W_test)

        test2d2s = SingleDiabatizer(2,2,W_guess)
        test2d2s.addDomain(x_data, V_t)
        test2d2s.optimize()
        W = test2d2s.Wout
        for w, wt in zip(W,W_test):
            for c, ct in zip(w.coeffs(), wt.coeffs()):
                assert abs(c-ct) < 1E-10

        # Show the result if verbose test
        if pytestconfig.getoption("verbose") > 0:
            Wx = W(x_data)
            self.plot_2d2s_testVSfit(X,Y,W_test_x,V_t,Wx)

    def test_2d3s(self, pytestconfig):
        """ Three states, two dimensions.
        1: Construct an arbitrary diabatic potential matrix W_test
        2: Diagonalize it in order to get the adiabatic PESs
        3: Fit a new W_fit to the adiabatic data and check if
           W_fit == W_test
        """

        # 0: Prepare coordinates
        x1, x2, y1, y2 = (-2,2,-1,1)
        Nx, Ny = (21,21)
        x = np.linspace(x1,x2,Nx)
        y = np.linspace(y1,y2,Ny)

        X, Y = np.mgrid[x1:x2:Nx*1j, y1:y2:Ny*1j]
        x0, y0 = (0,0)
        X = X - x0
        Y = Y - y0

        x_data = np.hstack((
            X.flatten()[:,np.newaxis],
            Y.flatten()[:,np.newaxis],
            ))

        # 1: Make target diabatic surfaces
        W_test = self.W_ho_2d3s()
        W_test_x = W_test(x_data)

        # 2: Make test adiabatic surfaces
        V_t = adiabatic(W_test_x)[0]

        # 3: Fit diabatize test adiabatic surfaces
        W_guess = SymPolyMat.zero_like(W_test)
        W_guess[0,0][(0,0)] = 0
        W_guess[1,1][(1,0)] = -0.5
        W_guess[2,2][(0,0)] = 1

        test2d3s = SingleDiabatizer(3,2,W_guess)
        test2d3s.addDomain(x_data, V_t)
        test2d3s.optimize()
        W = test2d3s.Wout

        for i in range(3):
            for c, ct in zip(W[i,i].coeffs(), W_test[i,i].coeffs()):
                assert abs(c-ct) < 1E-10
            for j in range(i):
                for c, ct in zip(W[i,j].coeffs(), W_test[i,j].coeffs()):
                    assert abs(abs(c)-abs(ct)) < 1E-10

        # Show the result if verbose test
        if pytestconfig.getoption("verbose") > 0:
            Wx = W(x_data)
            self.plot_2d3s_testVSfit(X,Y,W_test_x,V_t,Wx)

    def test_1d2s_crossing(self, pytestconfig):
        """ Two states, one dimension, two distinct avoided crossings
        1: Construct two arbitrary diabatic potential matrices
        2: Diagonalize both in order to get the adiabatic PESs
        3: Fit a unique W_fit to the adiabatic data and verify
           W_fit != W_test
        4: Do the fit with two matrices and check if the couplings are
           now correct
        """
        Ns = 2
        Nd = 1

        # 0: Prepare coordinates
        xmin, xmax = (-2,2)
        Nx = 21
        x = np.linspace(xmin,xmax,Nx)
        left = range(Nx//2)
        right = range(Nx//2, Nx)
        x01, x02 = (-1,1)

        # 1: Make target diabatic surfaces
        Wt1 = SymPolyMat(2,1)
        Wt1[0,0] = NdPoly.zero(1)
        Wt1[1,0] = NdPoly.one(1)*0.1
        Wt1[1,1] = NdPoly({(1,):-1})
        Wt1.x0 = -1
        Wt2 = deepcopy(Wt1)
        Wt2[1,0] *= 2
        Wt2[1,1] *=  -1
        Wt2.x0 = 1

        W_test_x = np.zeros((Nx, Ns, Ns))
        W_test_x[left,:,:] = Wt1(x[left])
        W_test_x[right,:,:] = Wt2(x[right])

        # 2: Make test adiabatic surfaces
        V_t = adiabatic(W_test_x)[0]

        # 3: Fit diabatize test adiabatic surfaces
        W_guess = SymPolyMat.zero_like(Wt1)
        W_guess[0,0][(0,)] = 0
        W_guess[1,0][(0,)] = 0.2
        W_guess[1,1][(0,)] = 1
        W_guess[1,1][(2,)] = 1

        diab = SingleDiabatizer(Ns,Nd,W_guess)
        diab.addDomain(x, V_t)
        diab.optimize()
        W = diab.Wout

        # Diabatize using two matrices
        W_guess2 = [SymPolyMat.zero_like(Wt1) for _ in range(2)]
        W_guess2[0][0,0][(0,)] = 0
        W_guess2[0][1,0][(0,)] = 0.2
        W_guess2[0][1,1][(0,)] = -1
        W_guess2[0][1,1][(1,)] = -1
        W_guess2[1][0,0][(0,)] = 0
        W_guess2[1][1,0][(0,)] = 0.7
        W_guess2[1][1,1][(0,)] = 1
        W_guess2[1][1,1][(1,)] = 1
        diab2 = Diabatizer(Ns,Nd,2,W_guess2)
        diab2.addDomain(x[left], V_t[left,:])
        diab2.addDomain(x[right], V_t[right,:])
        diab2.setFitDomain(0,0) # matrix 0 to domain 0, all states
        diab2.setFitDomain(1,1) # matrix 1 to domain 1, all states
        diab2.optimize()
        W2 = diab2.Wout

        # Show the result if verbose test
        if pytestconfig.getoption("verbose") > 0:
            Wx = W(x)
            self.plot_1d2s_testVSfit(x,W_test_x,V_t,Wx)
            W2x = [w(x) for w in W2]
            self.plot_1d2s_testVSfit(x,W_test_x,V_t,W2x[0])
            self.plot_1d2s_testVSfit(x,W_test_x,V_t,W2x[1])


    def test_3d3s(self):
        pass
