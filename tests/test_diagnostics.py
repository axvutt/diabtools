import numpy as np
import pytest
from ..diagnostics import RMSE, wRMSE, MAE, wMAE

class TestDiagnostics:
    """ Diagnostics tests """
    @pytest.fixture(scope="class", params=[2,3,10,99])
    def residuals(self, request):
        return np.linspace(-1,1,request.param)

    @pytest.fixture(scope="class", params=[0.1,1,2])
    def constant_weights(self, residuals, request):
        return np.ones(len(residuals))*request.param

    @pytest.fixture(scope="class")
    def heaviside_weights(self, residuals):
        return np.heaviside(residuals,1)

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

    @pytest.mark.skip
    def test_wrmse_heaviside(self, residuals, heaviside_weights):
        n = len(residuals)
        p = n//2
        # if n % 2 == 0:
        #     assert (wRMSE(residuals, heaviside_weights)
        #         == pytest.approx(np.sqrt(2/3*(4*p-1)/(2*p-1)-1))
        #     )
        # else:
        #     assert (wRMSE(residuals, heaviside_weights)
        #         == pytest.approx(np.sqrt( 0.5*n/(p+1) * ((2/3*(4*p-1)/(2*p-1) - 1) )))
        #     )

    @pytest.mark.skip
    def test_wmae_heaviside(self, residuals, heaviside_weights):
        n = len(residuals)
        p = n//2
        # if n % 2 == 0:
        #     assert (wMAE(residuals, heaviside_weights)
        #         == pytest.approx(p/(2*p-1))
        #     )
        # else:
        #     assert (wMAE(residuals, heaviside_weights)
        #         == pytest.approx(0.5*n/(p+1) * (p+1)/(2*p+1))
        #     )





