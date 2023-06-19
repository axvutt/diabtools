import numpy as np
import matplotlib.pyplot as plt
from ..dampedsympolymat import DampedSymPolyMat
from ..damping import Gaussian, Lorentzian

class TestDampedSymMat:
    
    def test_Gaussian(self):
        G = Gaussian(0,1)
        assert G(0) == 1
        assert np.isclose(G(1),1/np.sqrt(np.e))
        G = Gaussian(1,2)
        assert G(1) == 1
        assert np.isclose(G(0), np.exp(-0.5*(-1/2)**2))

    def test_Lorenzian(self):
        L = Lorentzian(0,1)
        assert L(0) == 1
        assert np.isclose(L(1), 0.5)
        L = Lorentzian(1,2)
        assert L(1) == 1
        assert np.isclose(L(0), 4/5)

    def test_Call(self, pytestconfig):
        Ns = 3
        Nd = 2
        W = DampedSymPolyMat.zero(Ns,Nd)

        W[0,0].x0 = [0., 0.]
        W[0,0][0,0] = -1.
        W[0,0][2,0] = 1.
        W[0,0][0,2] = 1.
        W[1,1].x0 = [1., 0.]
        W[1,1][0,0] = 0
        W[1,1][2,0] = 0.5
        W[1,1][0,2] = 1.
        W[2,2].x0 = [-1., 0.]
        W[2,2][0,0] = 0
        W[2,2][2,0] = 0.5
        W[2,2][0,2] = 1.
        W[1,2].x0 = [0., 0.] 
        W[1,2][1,0] = 1.
        W.set_damping((1,2), 0, Gaussian(0,1))
        W.set_damping((1,2), 1, Gaussian(0,0.5))

        x = np.linspace(-2,2,101)
        y = np.linspace(-2,2,101)
        X,Y = np.meshgrid(x,y)
        data = np.hstack((
            X.flatten()[:,np.newaxis],
            Y.flatten()[:,np.newaxis],
            ))
        Wx = W(data)
        W11 = Wx[:,0,0].reshape(X.shape)
        W22 = Wx[:,1,1].reshape(X.shape)
        W33 = Wx[:,2,2].reshape(X.shape)
        W23 = Wx[:,1,2].reshape(X.shape)

        if pytestconfig.getoption("verbose") > 0:
            ax1 = plt.figure().add_subplot(projection='3d')
            ax1.plot_surface(X,Y,W11,alpha=0.5,cmap='Blues_r')
            ax1.contour(X,Y,W11,alpha=0.5, levels=21,colors='b')
            ax1.plot_surface(X,Y,W22,alpha=0.5,cmap='Reds_r')
            ax1.contour(X,Y,W22,alpha=0.5, levels=21,colors='r')
            ax1.plot_surface(X,Y,W33,alpha=0.5,cmap='Greens_r')
            ax1.contour(X,Y,W33,alpha=0.5, levels=21,colors='g')
            ax1.set_xlabel("$x$")
            ax1.set_ylabel("$y$")

            ax2 = plt.figure().add_subplot(projection='3d')
            ax2.plot_surface(X,Y,W23,alpha=0.5,cmap='BrBG')
            ax2.contour(X,Y,W23,alpha=0.5, levels=21, colors='indigo')
            ax2.set_xlabel("$x$")
            ax2.set_ylabel("$y$")

            plt.show()
