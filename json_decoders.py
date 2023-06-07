import json
from . import diabtools

class NdPolyJSONdecoder(json.JSONDecoder):
    def decode(self, s):
        raw = super().decode(s)
        P = diabtools.NdPoly.empty(raw["Nd"])
        P.x0 = raw["x0"]
        
        # Get polynomial coefficients
        def str2powers(s):
            """
            Convert string to original tuple of integers.
            Assuming that the string was obtained with str(t),
            t being the tuple.
            """
            return tuple(map(int,s.strip('()').split(', ')))

        for raw_power, coeff in raw["coeffs_by_powers"].items():
            P[str2powers(raw_power)] = coeff

        return P


class SymPolyMatJSONdecoder(json.JSONDecoder):
    def decode(self, s):
        return diabtools.SymPolyMat.eye(2,1)


