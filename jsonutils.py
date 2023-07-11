import json
# from .ndpoly import NdPoly
# from .sympolymat import SymPolyMat
# from .dampedsympolymat import DampedSymPolyMat
# from .damping import DampingFunction, One, Gaussian, Lorentzian
# from .diabatizer import Diabatizer
# from .results import Results
# from .ndpoly import NdPoly

def _str2tuple(s):
    """
    Convert string to original tuple of integers.
    Assuming that the string was obtained with str(t),
    t being the tuple.
    """
    return tuple(map(int,s.strip('()').split(', ')))

# In C++, use template
def save_to_JSON(obj, fname):
    with open(fname, "w") as f:
        json.dump(obj.to_JSON_dict(), f)

# In C++, maybe use variant?
def load_from_JSON(fname):
    with open(fname, "r") as f:
        dct = json.load(f)

    if "__NdPoly__" in dct:
        return NdPoly.from_JSON_dict(dct)

    if "__SymPolyMat__" in dct:
        if "__DampedSymPolyMat__" in dct:
            return DampedSymPolyMat.from_JSON_dict(dct)
        return SymPolyMat.from_JSON_dict(dct)

    if "__DampingFunction__" in dct:
        if "__One__" in dct:
            return One.from_JSON_dict(dct)
        if "__Gaussian__" in dct:
            return Gaussian.from_JSON_dict(dct)
        if "__Lorentzian__" in dct:
            return Lorenzian.from_JSON_dict(dct)
        raise Warning("Serialized abstract DampingFunction instance.")

    if "__Results__" in dct:
        return Results.from_JSON_dict(dct)
    
    if "__Diabatizer__" in dct:
        return Diabatizer.from_JSON_dict(dct)
