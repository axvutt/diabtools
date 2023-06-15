import numpy as np

def switch_poly(x, smin=0, smax=1):
    """ Polynomial switching function with zero first and second derivatives at 0 and 1.
    See: Zhang et al., J. Phys. Chem. A, 2004, 108, 41, 8980–8986.
    https://doi.org/10.1021/jp048339l
    Beware of the typo: should be -10*x^3 instead of -10*x^2
    """
    xs = (x-smin)/(smax-smin)
    xs = np.maximum(np.minimum(xs, 1), 0)
    # if xs == 0:
    #     return 1
    # elif xs == 1:
    #     return 0
    # else:
    #     return 1 + xs**3 * (-10 + xs*(15 - 6*xs))
    return 1 + xs**3 * (-10 + xs*(15 - 6*xs))

def switch_sinsin(x, smin=0, smax=1):
    """ Sine of sine switching function with zero first and second derivative at 0 and 1.
    See: Sahnoun et al., J. Phys. Chem. A, 2018, 122, 11, 3004–3012.
    https://doi/10.1021/acs.jpca.8b00150
    """
    xs = (x-smin)/(smax-smin)
    xs = np.maximum(np.minimum(xs, 1), 0)
    # if xs == 0:
    #     return 1
    # elif xs == 1:
    #     return 0
    # else:
    #     return np.sin(np.pi/2 * np.sin(np.pi/2 * xs) )
    return (1 - np.sin(np.pi/2*np.sin(np.pi/2*(2*xs - 1))))/2
