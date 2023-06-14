import os
import numpy as np
import pytest
import json
from ..diabtools import *
from ..json_encoders import *
from ..json_decoders import *

@pytest.fixture
def poly_ex():
    """ Sample NdPoly """
    return NdPoly({(1,2,3): 1, (0,1,2): 2, (3,0,2): 3, (0,4,0): 4, (2,2,2): 5, (6,6,6): 0})

@pytest.fixture
def spm_ex(poly_ex):
    """ Sample SymPolyMat """
    M = SymPolyMat(3,3)
    for i in range(3):
        for j in range(i+1):
            M[i,j] = poly_ex
    return M

@pytest.fixture
def damp_ex():
    """ Sample DampingFunction """
    return Gaussian(0,1)

@pytest.fixture
def dspm_ex(spm_ex, damp_ex):
    """ Sample DampedSymPolyMat """
    D = DampedSymPolyMat.from_SymPolyMat(spm_ex)
    for i in range(spm_ex.Ns):
        for j in range(i):
            D.set_damping((i,j), damp_ex)
    return D

@pytest.fixture(params=["spm_ex", "damp_ex"])
def diab_ex(request):
    W = request.param
    diabatizer = Diabatizer(3,3,2, [spm_ex, spm_ex])
    return diabatizer

@pytest.mark.usefixtures('poly_ex','spm_ex','diab_ex')
class TestJSON:
    def _test_ndpoly_codec(self, poly_ex):
        # Encode to JSON string
        s = DiabJSONEncoder(indent=4).encode(poly_ex)

        # Decode from JSON file
        P = DiabJSONDecoder().decode(s)
        assert P == poly_ex

    def _test_sympolymat_codec(self, spm_ex):
        # Encode to JSON string
        s = DiabJSONEncoder(indent=4).encode(spm_ex)

        # Decode from JSON file
        W = DiabJSONDecoder().decode(s)
        assert W == spm_ex

    def _test_diabatizer_codec(self,diab_ex):
        s = DiabJSONEncoder(indent=4).encode(diab_ex)
        with open("diabtest.json", "w") as f:
            f.write(s)

    def test_ndpoly(self, poly_ex):
        s = json.dumps(poly_ex.to_JSON_dict())
        P = NdPoly.from_JSON_dict(json.loads(s))
        assert P == poly_ex

    def test_sympolymat(self, spm_ex):
        s = json.dumps(spm_ex.to_JSON_dict())
        M = SymPolyMat.from_JSON_dict(json.loads(s))
        assert M == spm_ex
