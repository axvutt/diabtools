import os
import numpy as np
import pytest
from ..diabtools import *
from ..json_encoders import *
from ..json_decoders import *

@pytest.fixture
def poly_ex():
    return NdPoly({(1,2,3): 1, (0,1,2): 2, (3,0,2): 3, (0,4,0): 4, (2,2,2): 5, (6,6,6): 0})


@pytest.mark.usefixtures('poly_ex')
class TestJSON:
    def test_ndpoly_codec(self, poly_ex):

        # Encode to JSON string
        s = NdPolyJSONencoder(indent=4).encode(poly_ex)

        # Decode from JSON file
        P = NdPolyJSONdecoder().decode(s)

        assert P == poly_ex

        
