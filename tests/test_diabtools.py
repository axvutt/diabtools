from ..diabtools import *
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def test_switchPoly(pytestconfig):
    x = np.linspace(-2,2,101)
    y = switch_poly(x)
    z = switch_poly(x, -1, 1)
    if pytestconfig.getoption("verbose") > 0:
        plt.plot(x,y)
        plt.plot(x,z)
        plt.show()
    assert np.all(np.isclose(y[np.isclose(x,0)], 1))
    assert np.all(np.isclose(y[np.isclose(x,0.5)], 0.5))
    assert np.all(np.isclose(y[np.isclose(x,1)], 0))
    assert np.all(np.isclose(z[np.isclose(x,-1)], 1))
    assert np.all(np.isclose(z[np.isclose(x,0)], 0.5))
    assert np.all(np.isclose(z[np.isclose(x,1)], 0))

def test_switchSinSin(pytestconfig):
    x = np.linspace(-2,2,101)
    y = switch_sinsin(x)
    z = switch_sinsin(x, -1, 1)
    if pytestconfig.getoption("verbose") > 0:
        plt.plot(x,y)
        plt.plot(x,z)
        plt.show()
    assert np.all(np.isclose(y[np.isclose(x,0)], 1))
    assert np.all(np.isclose(y[np.isclose(x,0.5)], 0.5))
    assert np.all(np.isclose(y[np.isclose(x,1)], 0))
    assert np.all(np.isclose(z[np.isclose(x,-1)], 1))
    assert np.all(np.isclose(z[np.isclose(x,0)], 0.5))
    assert np.all(np.isclose(z[np.isclose(x,1)], 0))

