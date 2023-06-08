import json
from . import diabtools

def str2tuple(s):
    """
    Convert string to original tuple of integers.
    Assuming that the string was obtained with str(t),
    t being the tuple.
    """
    return tuple(map(int,s.strip('()').split(', ')))


class NdPolyJSONDecoder(json.JSONDecoder):
    def __init__(self):
        super().__init__(object_hook=self.object_hook)

    def object_hook(self, dct):
        if "__NdPoly__" in dct:
            P = diabtools.NdPoly.empty(dct["Nd"])
            P.x0 = dct["x0"]
            
            # Get polynomial coefficients
            for raw_power, coeff in dct["coeffs_by_powers"].items():
                P[str2tuple(raw_power)] = coeff

            return P

        else:
            return dct


class SymPolyMatJSONDecoder(json.JSONDecoder):
    def __init__(self):
        super().__init__(object_hook=self.object_hook)

    def object_hook(self, dct):
        if "__SymPolyMat__" in dct:
            poly_decoder = NdPolyJSONDecoder()

            W = diabtools.SymPolyMat(dct["Ns"],dct["Nd"])
            for ij, poly in dct["elements"].items():
                W[str2tuple(ij)] = poly_decoder.object_hook(poly)
            
            return W

        else:
            return dct


