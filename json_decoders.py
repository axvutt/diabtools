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
    def decode(self, s):
        raw = super().decode(s)
        P = diabtools.NdPoly.empty(raw["Nd"])
        P.x0 = raw["x0"]
        
        # Get polynomial coefficients
        for raw_power, coeff in raw["coeffs_by_powers"].items():
            P[str2tuple(raw_power)] = coeff

        return P


class SymPolyMatJSONDecoder(json.JSONDecoder):
    def decode(self, s):
        raw = super().decode(s)
        poly_decoder = NdPolyJSONDecoder()

        W = diabtools.SymPolyMat(raw["Ns"],raw["Nd"])
        for ij, poly in raw["elements"].items():
            pass
            # W[str2tuple(ij)] = poly_decoder.decode(json_poly)
        
        return W


