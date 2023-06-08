import json
from . import diabtools

class DiabJSONEncoder(json.JSONEncoder):
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

        if isinstance(obj, diabtools.SymPolyMat):
            Wij = dict()
            for i in range(obj.Ns):
                for j in range(i+1):
                    Wij[f"({i}, {j})"] = self.default(obj[i,j])
            out = {
                    "__SymPolyMat__" : True,
                    "Nd" : obj.Nd,
                    "Ns" : obj.Ns,
                    "elements" : Wij
                    }
            return out

        # If not a diabtools type, use default parser
        return json.JSONEncoder.default(self, obj)


