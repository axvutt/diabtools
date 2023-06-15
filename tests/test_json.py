import numpy as np
import pytest
import json
from ..ndpoly import NdPoly
from ..sympolymat import SymPolyMat
from ..dampedsympolymat import DampedSymPolyMat
from ..diabatizer import Diabatizer

class TestJSON:
    """ """
    #################### FIXTURES #################### 
    @pytest.fixture(scope="class")
    def poly_ex(self):
        """ Sample NdPoly """
        return NdPoly({(1,2,3): 1, (0,1,2): 2, (3,0,2): 3, (0,4,0): 4, (2,2,2): 5, (6,6,6): 0})

    @pytest.fixture(scope="class")
    def spm_ex(self, poly_ex):
        """ Sample SymPolyMat """
        M = SymPolyMat(3,3)
        for i in range(3):
            for j in range(i+1):
                M[i,j] = poly_ex
        return M

    @pytest.fixture(scope="class")
    def damp_ex(self):
        """ Sample DampingFunction """
        return Gaussian(0,1)

    @pytest.fixture(scope="class")
    def dspm_ex(self, spm_ex, damp_ex):
        """ Sample DampedSymPolyMat """
        D = DampedSymPolyMat.from_SymPolyMat(spm_ex)
        for i in range(spm_ex.Ns):
            for j in range(i):
                D.set_damping((i,j), damp_ex)
        return D

    @pytest.fixture(scope="class", params=["spm_ex", "dspm_ex"])
    def diab_ex(self, request):
        W = request.getfixturevalue(request.param)
        diabatizer = Diabatizer(3,3,2, [W, W])
        return diabatizer

    #################### FIXTURES #################### 
    def test_ndpoly(self, poly_ex):
        s = json.dumps(poly_ex.to_JSON_dict())
        P = NdPoly.from_JSON_dict(json.loads(s))
        assert P == poly_ex

    def test_sympolymat(self, spm_ex):
        s = json.dumps(spm_ex.to_JSON_dict())
        M = SymPolyMat.from_JSON_dict(json.loads(s))
        assert M == spm_ex

    def test_dampedsympolymat(self, dspm_ex):
        s = json.dumps(dspm_ex.to_JSON_dict())
        M = DampedSymPolyMat.from_JSON_dict(json.loads(s))
        assert M == dspm_ex

    def test_diabatizer(self, diab_ex):
        s = json.dumps(diab_ex.to_JSON_dict(), indent=4)
        with open("diabfile.json", "w") as f:
            f.write(s)
        diab = Diabatizer.from_JSON_dict(json.loads(s))
        # assert diab == diab_ex
