import numpy as np
from typing import Union


class WeightFunction:
    @staticmethod
    def calc(*args, **kwargs):
        raise NotImplementedError('WeightFunction class should not be used directly')

    @staticmethod
    def parse_params(params):
        """A function for seamless integration with ensembles"""
        if isinstance(params, dict) and list(params.keys()) == ['scaling']:
            return params['scaling']
        else:
            return params

    def __init__(self, scaling: Union[float, dict]):
        self.scaling = scaling

    def __call__(self, estimation_coords: np.array, pivot_coords: np.array):
        if len(estimation_coords.shape) == 1:
            estimation_coords_intern = np.array([estimation_coords])
        else:
            estimation_coords_intern = estimation_coords

        if len(pivot_coords.shape) == 1:
            pivot_coords_intern = np.array([pivot_coords])
        else:
            pivot_coords_intern = pivot_coords

        answer = np.ones(estimation_coords_intern.shape[0])
        for p in pivot_coords_intern:
            answer = answer * self.calc(np.linalg.norm(estimation_coords_intern - p, axis=1), self.scaling)

        return answer


class Exponential(WeightFunction):
    @staticmethod
    def calc(distances: np.array, scaling: float):
        return 1 - np.exp(-distances / scaling)


class Spherical(WeightFunction):
    @staticmethod
    def calc(distances: np.array, scaling: float):
        answer = np.ones(len(distances))
        transients_mask = distances < scaling
        transient_points = distances[transients_mask] / scaling
        answer[transients_mask] = 1.5 * transient_points - (transient_points**3)/2
        return answer


class Polynomial(WeightFunction):
    @staticmethod
    def calc(distances: np.array, scaling: float):
        answer = np.ones(len(distances))
        transients_mask = distances < scaling
        x = distances[transients_mask] / scaling
        answer[transients_mask] = 7*x**2-35/4*x**3+7/2*x**5-3/4*x**7
        return answer


class Cubic(WeightFunction):
    @staticmethod
    def calc(distances: np.array, params: dict):
        answer = np.zeros(len(distances))
        left_mask = distances <= params['offset']
        right_mask = distances >= params['offset'] + params['scaling']
        transients_mask = ~(left_mask | right_mask)
        x = (distances[transients_mask] - params['offset']) / params['scaling']
        answer[transients_mask] = -2 * x ** 3 + 3 * x ** 2
        answer[right_mask] = 1
        return answer

    @staticmethod
    def parse_params(params):
        """A function for seamless integration with ensembles"""
        zero_region_absolute = params['scaling'] * params['zero_region_relative']
        transient_region_absolute = params['scaling'] * params['transient_region_relative']
        return {'offset': zero_region_absolute, 'scaling': transient_region_absolute}
