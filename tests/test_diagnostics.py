import numpy as np
import pytest
from ..diagnostics import RMSE, wRMSE, MAE, wMAE

class TestDiagnostics:
    """ Diagnostics tests """
    @pytest.fixture(scope="class", params=[2,3,10,99])
    def residuals(self, request):
        return np.linspace(-1,1,request.param)

    @pytest.fixture(scope="class", params=[0,25,50])
    def delta_residuals(self, request):
        res = np.zeros(51)
        k = request.param
        res[k] = 1
        return res, k

    @pytest.fixture(scope="class", params=[0.1,1,2])
    def constant_weights(self, residuals, request):
        return np.ones(len(residuals))*request.param

    @pytest.fixture(scope="class")
    def cos_weights(self, delta_residuals):
        res, _ = delta_residuals
        return np.cos(np.linspace(-0.5*np.pi, 0.5*np.pi, len(res)))

    def test_rmse(self,residuals):
        n = len(residuals)
        assert RMSE(residuals) == pytest.approx(np.sqrt(2/3*(2*n-1)/(n-1)-1))

    def test_mae(self,residuals):
        n = len(residuals)
        p = n//2
        if n % 2 == 0:
            assert MAE(residuals) == pytest.approx(p/(2*p-1))
        else:
            assert MAE(residuals) == pytest.approx((p+1)/(2*p+1))

    def test_wrmse_as_rmse(self, residuals, constant_weights):
        assert wRMSE(residuals, constant_weights) == pytest.approx(RMSE(residuals))

    def test_wmae_as_mae(self, residuals, constant_weights):
        assert wMAE(residuals, constant_weights) == pytest.approx(MAE(residuals))

    def test_wrmse_deltares(self, delta_residuals, cos_weights):
        res, k = delta_residuals
        p = len(res) // 2
        assert (wRMSE(res, cos_weights)
            == pytest.approx(np.sqrt(np.cos(0.5*np.pi*(k/p-1))*np.tan(0.25*np.pi/p)))
        )

    def test_wmae_deltares(self, delta_residuals, cos_weights):
        res, k = delta_residuals
        p = len(res) // 2
        assert (wMAE(res, cos_weights)
            == pytest.approx(np.cos(0.5*np.pi*(k/p-1))*np.tan(0.25*np.pi/p))
        )





