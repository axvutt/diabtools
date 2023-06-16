import math
from abc import ABC, abstractmethod 
import numpy as np

class DampingFunction(ABC):
    """ Abstract base class for damping functions
    Subclasses should implement __call__"""
    def __init__(self, x0):
        self._x0 = x0

    @property
    def x0(self):
        return self._x0

    @x0.setter
    def x0(self, x0):
        self._x0 = x0

    @abstractmethod
    def __call__(self, x):
        pass

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if self._x0 == other._x0:
                return True
        return False

    def to_JSON_dict(self):
        return {"__DampingFunction__": True, "x0": self._x0}

    @staticmethod
    def from_JSON_dict(dct):
        return None


class One(DampingFunction):
    def __init__(self):
        super().__init__(0)

    def __call__(self, x):
        return 1

    def to_JSON_dict(self):
        dct = super().to_JSON_dict()
        dct.update({"__One__": True})
        return dct

    @staticmethod
    def from_JSON_dict(dct):
        if "__One__" not in dct:
            raise KeyError("The JSON object is not a One.")

        return One()


class Gaussian(DampingFunction):
    def __init__(self, x0, sigma):
        super().__init__(x0)
        if math.isclose(sigma,0):
            raise ValueError("Zero std deviation.")
        self._sigma = sigma

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, sigma):
        self._sigma = sigma

    def __call__(self,x):
        return np.exp(-0.5*((x-self.x0)/self.sigma)**2)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if super().__eq__(other):
                if self.sigma == other.sigma:
                    return True
        return False

    def to_JSON_dict(self):
        dct = super().to_JSON_dict()
        dct.update({"__Gaussian__": True, "sigma": self._sigma})
        return dct

    @staticmethod
    def from_JSON_dict(dct):
        if "__Gaussian__" not in dct:
            raise KeyError("The JSON object is not a Gaussian.")

        return Gaussian(dct["x0"], dct["sigma"])


class Lorentzian(DampingFunction):
    def __init__(self, x0, gamma):
        super().__init__(x0)
        if math.isclose(gamma,0):
            raise ValueError("Zero gamma width parameter.")
        self._gamma = gamma

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        self._gamma = gamma

    def __call__(self,x):
        return 1/(1 + ((x-self.x0)/self.gamma)**2)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if super().__eq__(other):
                if self.gamma == other.gamma:
                    return True
        return False

    def to_JSON_dict(self):
        dct = super().to_JSON_dict()
        dct.update({"__Lorentzian__": True, "gamma": self._gamma})
        return dct

    @staticmethod
    def from_JSON_dict(dct):
        if "__Lorentzian__" not in dct:
            raise KeyError("The JSON object is not a Lorentzian.")

        return Lorentzian(dct["x0"], dct["gamma"])
