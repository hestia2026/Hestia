import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar


class OptimalTempInitializer:
    SIGMA = np.sqrt(np.pi / 2)

    @staticmethod
    def _gaussian_pdf(z):
        sigma = OptimalTempInitializer.SIGMA
        return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (z / sigma)**2)

    @staticmethod
    def _soft_quant_func(z, T):
        val = 2 * z / T
        numerator = np.sinh(val)
        denominator = np.cosh(val) + 0.5 * np.exp(1/T)
        return numerator / denominator

    @staticmethod
    def _integrand(z, T):
        error = OptimalTempInitializer._soft_quant_func(z, T) - z
        return (error**2) * OptimalTempInitializer._gaussian_pdf(z)

    @staticmethod
    def _calculate_expected_mse(T):
        result, error = quad(OptimalTempInitializer._integrand, -10, 10, args=(T,))
        return result

    @classmethod
    def calculate(cls, bounds=(0.1, 2.0)):
        res = minimize_scalar(cls._calculate_expected_mse, bounds=bounds, method='bounded')
        
        if res.success:
            return res.x
        else:
            raise RuntimeError("Failed to optimize the temperature.")
