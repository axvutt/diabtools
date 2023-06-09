import json
from . import diabtools

def str2tuple(s):
    """
    Convert string to original tuple of integers.
    Assuming that the string was obtained with str(t),
    t being the tuple.
    """
    return tuple(map(int,s.strip('()').split(', ')))


class DiabJSONDecoder(json.JSONDecoder):
    def __init__(self):
        super().__init__(object_hook=self.object_hook)

    def parse_NdPoly(self, dct):
        P = diabtools.NdPoly.empty(dct["Nd"])
        P.x0 = dct["x0"]
        
        # Get polynomial coefficients
        for raw_power, coeff in dct["coeffs_by_powers"].items():
            P[str2tuple(raw_power)] = coeff

        return P

    def parse_SymPolyMat(self, dct):
        W = diabtools.SymPolyMat(dct["Ns"],dct["Nd"])
        for ij, poly in dct["elements"].items():
            W[str2tuple(ij)] = poly
        
        return W

    def parse_DampedSymPolyMat(self, dct):
        # DW = diabtools.DampedSymPolyMat.from_base(
        #         self.parse_SymPolyMat(dct)
        #         )
        # for ij, damp in dct["damping"].items():
        #     DW.set_damping(str2tuple(ij), self.parse_DampingFunction(damp))
        return DW

    def object_hook(self, dct):
        if "__NdPoly__" in dct:
            return self.parse_NdPoly(dct)

        if "__SymPolyMat__" in dct:
            return self.parse_SymPolyMat(dct)
        
        if "__DampedSymPolyMat__" in dct:
            pass

        if "__Diabatizer__" in dct:
            pass

        if "__Results__" in dct:
            pass

        # Return the unchanged dict otherwise
        return dct


