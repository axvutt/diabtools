import json
from . import diabtools

class NdPolyJSONencoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, diabtools.NdPoly):
            poly = {str(p) : c for p,c in obj.items()}
            out = {
                    "Nd" : obj.Nd,
                    "x0" : obj.x0.tolist(),
                    "coeffs_by_powers" : poly,
                    }
            return out
        else:
            return json.JSONEncoder.default(self, obj)


class SymPolyMatJSONencoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, diabtools.SymPolyMat):
            return []
        else:
            return json.JSONEncoder.default(self, obj)


