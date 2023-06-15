import numpy as np

def RMSE(residuals):
    """ Compute unweighted Root-Mean-Square Error from residuals. """
    return np.sqrt(np.sum(residuals**2)/len(residuals))

def wRMSE(residuals, weights):
    """ Compute weighted Root-Mean-Square Error from residuals and weights. """
    return np.sqrt(np.sum(weights * residuals**2)/np.sum(weights))

def MAE(residuals):
    """ Compute unweighted Mean-Average Error from residuals. """
    return np.sum(np.abs(residuals))/len(residuals)

def wMAE(residuals, weights):
    """ Compute weighted Mean-Average Error from residuals and weights. """
    return np.sum(weights * np.abs(residuals))/np.sum(weights)
