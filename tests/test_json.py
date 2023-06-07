import os
import numpy as np
import pytest
from ..diabtools import *
from ..json_encoders import *
from ..json_decoders import *

@pytest.fixture
def poly_ex():
    return NdPoly({(1,2,3): 1, (0,1,2): 2, (3,0,2): 3, (0,4,0): 4, (2,2,2): 5, (6,6,6): 0})

@pytest.fixture
def spm_ex(poly_ex):
    M = SymPolyMat(3,3)
    for i in range(3):
        for j in range(i+1):
            M[i,j] = poly_ex
    return M

@pytest.mark.usefixtures('poly_ex','spm_ex')
class TestJSON:
    def test_ndpoly_codec(self, poly_ex):
        # Encode to JSON string
        s = NdPolyJSONEncoder(indent=4).encode(poly_ex)

        # Decode from JSON file
        P = NdPolyJSONDecoder().decode(s)
        assert P == poly_ex

    def test_sympolymat_codec(self, spm_ex):
        # Encode to JSON string
        s = SymPolyMatJSONEncoder(indent=4).encode(spm_ex)

        # Decode from JSON file
        W = SymPolyMatJSONDecoder().decode(s)
        assert W == spm_ex

        
