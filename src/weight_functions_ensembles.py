import numpy as np
from sklearn.neighbors import KDTree
from typing import Optional
from enum import Enum


class EnsembleScalingMode(Enum):
    universal = 'universal'
    individual = 'individual'


class WeightFunctionsEnsemble:
    def __init__(
            self, fun_class, pivot_coords: Optional[np.array] = None,
            grid_coords: Optional[np.array] = None,
            scaling_mode: Optional[EnsembleScalingMode] = EnsembleScalingMode.individual,
            params=dict()
    ):
        self._scaling_mode = scaling_mode
        self._ensemble_params = params
        self._fun_class = fun_class
        self._grid_coords = grid_coords
        self.pivot_coords = pivot_coords

    def __call__(self, estimation_coords: np.array):
        if len(estimation_coords.shape) == 1:
            estimation_coords_intern = np.array([estimation_coords])
        else:
            estimation_coords_intern = estimation_coords

        answer = np.ones(estimation_coords_intern.shape[0])
        for fun, p in zip(self.warehouse, self.pivot_coords):
            answer = answer * fun.calc(np.linalg.norm(estimation_coords_intern - p, axis=1), fun.scaling)

        return answer

    @property
    def pivot_coords(self):
        return self._pivot_coords

    @pivot_coords.setter
    def pivot_coords(self, pivot_coords: Optional[np.array]):
        if self.grid_coords is None and self._scaling_mode is EnsembleScalingMode.universal:
            raise AttributeError('in universal scaling mode grid_coords attribute should be set')

        self.warehouse = []
        if pivot_coords is not None and not pivot_coords.shape == 0:
            self._pivot_coords = pivot_coords
            neighbors_processor = KDTree(pivot_coords, leaf_size=2)
            if self._scaling_mode is EnsembleScalingMode.individual:
                dists, _ = neighbors_processor.query(pivot_coords, k=2)
                dists = dists[:, 1] / 2
            else:
                dists, _ = neighbors_processor.query(self.grid_coords, k=1)
                dists = np.ones(self._pivot_coords.shape[0]) * dists.flatten().max()

            params_ = self._ensemble_params.copy()
            for coord, scaling_dist in zip(pivot_coords, dists):
                params_.update({'scaling': scaling_dist})
                self.warehouse.append(self._fun_class(self._fun_class.parse_params(params_)))
        else:
            self._pivot_coords = None

    @property
    def grid_coords(self):
        return self._grid_coords

    @grid_coords.setter
    def grid_coords(self, grid_coords: Optional[np.array]):
        self._grid_coords = grid_coords

    @property
    def fun_class(self):
        return self._fun_class

    @fun_class.setter
    def fun_class(self, cl):
        raise Exception('Changing class of weight function is forbidden. One needs to recreate ensemble instance.')

    @property
    def ensemble_params(self):
        return self._ensemble_params

    @ensemble_params.setter
    def ensemble_params(self, params):
        raise Exception('Changing ensemble parameters is forbidden. One needs to recreate ensemble instance.')

    @property
    def scaling_mode(self):
        return self._scaling_mode
