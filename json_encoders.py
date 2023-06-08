import json
from . import diabtools

class NdPolyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, diabtools.NdPoly):
            poly = {str(p) : c for p,c in obj.items()}
            out = {
                    "__NdPoly__" : True,
                    "Nd" : obj.Nd,
                    "x0" : obj.x0.tolist(),
                    "coeffs_by_powers" : poly,
                    }
            return out
        else:
            return json.JSONEncoder.default(self, obj)


class SymPolyMatJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, diabtools.SymPolyMat):
            Wij = dict()
            poly_encoder = NdPolyJSONEncoder(indent=self.indent)
            for i in range(obj.Ns):
                for j in range(i+1):
                    Wij[f"({i}, {j})"] = poly_encoder.default(obj[i,j])
            out = {
                    "__SymPolyMat__" : True,
                    "Nd" : obj.Nd,
                    "Ns" : obj.Ns,
                    "elements" : Wij
                    }
            return out
        else:
            return json.JSONEncoder.default(self, obj)


