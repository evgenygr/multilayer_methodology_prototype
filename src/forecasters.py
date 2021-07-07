from scipy.spatial import Delaunay, KDTree
from itertools import product
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from scipy.interpolate import LinearNDInterpolator
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from typing import Union, Optional
from enum import Enum
from scipy.linalg import norm
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from itertools import combinations
from src.random_variable_transformers import IdentityTransform
# from pykrige.uk import UniversalKriging
# import pyKriging

numeric_dtypes = [np.dtype(name+bits) for name, bits in product(['int', 'float'], ['16', '32', '64'])]


class Taxonomy(Enum):
    deterministic = 'deterministic'  # forecaster that pins its forecast of a value in reference point to the value
    # that was prescribed to reference point in learning sample
    probabilistic = 'probabilistic'  # forecaster that doesn't pin its forecast of a value in reference point to the
    # value that was prescribed to reference point in learning sample. Moreover, it is natural for this kind of
    # forecaster to have multiple values from the same reference point


def knn_weights_fun(distances: np.array):
    """
    If there are several equally close neighbors 1nn-regressor alternates between them in different calls.
    This weight fun is destined to mitigate the effect of false peak of variance of 1nn regression
    """
    weights = np.zeros(distances.shape)
    weights[distances == distances.min(axis=1).reshape([distances.shape[0], 1])] = 1
    return weights


class BaselineForecaster:
    def __init__(self, train_data: Optional[pd.DataFrame] = None, grid_builder: Optional[callable] = None):
        self._field_name = None
        if train_data is not None:
            self.train_data = train_data
        else:
            self._train_data = None
        self.grid_builder = grid_builder

    taxon = Taxonomy.deterministic

    @property
    def field_name(self):
        return self._field_name

    @property
    def train_data(self):
        return self._train_data

    @property
    def tri(self):
        return self._tri

    @tri.setter
    def tri(self, val):
        raise AttributeError('Value assignment is prohibited for tri-property')

    @train_data.setter
    def train_data(self, data: pd.DataFrame):
        if data.shape[1] < 3:
            raise AttributeError('train_data should comprise 3 columns including X and Y')
        columns = list(data.columns)
        if 'X' not in columns and 'Y' not in columns:
            raise AttributeError('train_data should contain columns X and Y')
        for string in columns:
            if string not in ['X', 'Y'] and data[string].dtype in numeric_dtypes:
                self._field_name = string
        self._train_data = data.copy()
        self._tri = Delaunay(self._train_data[['X', 'Y']].values)
        self._ip = LinearNDInterpolator(self._tri, self._train_data[self._field_name])

    def forecast(self, objects_to_forecast: Union[pd.DataFrame, np.array], whole: bool = False):
        if self._train_data is None:
            raise AttributeError('Baseline forecaster is incomplete. Set train_data before invoking forecast')

        points = np.array(self._train_data[['X', 'Y', self._field_name]])

        if isinstance(objects_to_forecast, pd.DataFrame):
            objects_to_forecast_standardized = objects_to_forecast[['X', 'Y']].values
        elif isinstance(objects_to_forecast, np.ndarray):
            objects_to_forecast_standardized = objects_to_forecast
        else:
            raise TypeError(
                'Unexpected type of objects_to_forecast: {}. Only pandas.DataFrame and numpy.array are allowed.'.
                format(type(objects_to_forecast))
            )

        ntg_forecast = pd.DataFrame(
            {'X': objects_to_forecast_standardized[:, 0],
             'Y': objects_to_forecast_standardized[:, 1],
             self._field_name: self._ip(objects_to_forecast_standardized)
             }
        )

        if whole is True:
            triangles = [points[self._tri.simplices[
                int(self._tri.find_simplex(el))]] if self._tri.find_simplex(el) != -1 else np.array([None, None, None])
                         for el in objects_to_forecast_standardized]
            ntg_forecast['Distance'] = None
            ntg_forecast['Vertex1'] = [el[0] for el in triangles]
            ntg_forecast['Vertex2'] = [el[1] for el in triangles]
            ntg_forecast['Vertex3'] = [el[2] for el in triangles]

        failed_interpolation_mask = ntg_forecast[self._field_name].isnull().values
        insiders = ntg_forecast[~failed_interpolation_mask]

        if failed_interpolation_mask.any():
            outsiders = ntg_forecast[failed_interpolation_mask]
            nn_pivot_points = insiders.append(self._train_data, sort=False)
            knn = KNeighborsRegressor(n_neighbors=3, weights=knn_weights_fun).fit(
                X=nn_pivot_points[['X', 'Y']].values,
                y=nn_pivot_points[self._field_name].values
            )

            nn_forecasted_partial_ntg_map = pd.DataFrame(
                {'X': outsiders['X'].values,
                 'Y': outsiders['Y'].values,
                 self._field_name: knn.predict(outsiders[['X', 'Y']].values)
                 }
            )

            if whole is True:
                distance_ = knn.kneighbors(X=outsiders[['X', 'Y']].values, n_neighbors=1, return_distance=True)
                distance = distance_[0].ravel()
                corresponding_pivots = nn_pivot_points.reset_index(drop=True).loc[
                    distance_[1].ravel(),
                    ['X', 'Y']
                ].values
                corresponding_simplexes = self._tri.find_simplex(corresponding_pivots)
                if (corresponding_simplexes == -1).any():
                    raise RuntimeError('Failed to determine correct simplex for a point inside a convex hull')
                points_indexes = self._tri.simplices[corresponding_simplexes, :]
                nn_forecasted_partial_ntg_map['Distance'] = distance
                nn_forecasted_partial_ntg_map['Vertex1'] = [points[i, :] for i in points_indexes[:, 0]]
                nn_forecasted_partial_ntg_map['Vertex2'] = [points[i, :] for i in points_indexes[:, 1]]
                nn_forecasted_partial_ntg_map['Vertex3'] = [points[i, :] for i in points_indexes[:, 2]]

            ntg_forecast = insiders.append(nn_forecasted_partial_ntg_map, ignore_index=True, sort=False)

        return ntg_forecast

    def __call__(self, train_data: Optional[pd.DataFrame] = None, whole=False):
        """This is a convenience function meant to be invoked by MAP-action of multiprocess pool"""
        if self.grid_builder is None:
            raise AttributeError('Baseline forecaster is incomplete. Reinitialize it with grid_builder set.')

        if train_data is None:
            if self.train_data is None:
                raise AttributeError(
                    'train_data is not provided as argument. Baseline forecaster doesn`t incapsulate train_data either.'
                )
        else:
            self.train_data = train_data

        objects_to_forecast: np.array = self.grid_builder(train_data)
        return self.forecast(objects_to_forecast, whole)


class BaselineForecasterWithTriFilter(BaselineForecaster):
    def __init__(self,
                 h_cutoff: float,
                 train_data: Optional[pd.DataFrame] = None,
                 grid_builder: Optional[callable] = None):
        super().__init__(train_data, grid_builder)
        self.h_cutoff = h_cutoff

    def forecast(self, objects_to_forecast: Union[pd.DataFrame, np.array], whole: bool = False):
        if self._train_data is None:
            raise AttributeError('Baseline forecaster is incomplete. Set train_data before invoking forecast')

        points = np.array(self._train_data[['X', 'Y', 'NTG']])

        if isinstance(objects_to_forecast, pd.DataFrame):
            objects_to_forecast_standardized = objects_to_forecast[['X', 'Y']].values
        elif isinstance(objects_to_forecast, np.ndarray):
            objects_to_forecast_standardized = objects_to_forecast
        else:
            raise TypeError(
                'Unexpected type of objects_to_forecast: {}.Only pandas.DataFrame and numpy.array are allowed.'.
                format(type(objects_to_forecast))
            )

        simplices_inds = self._tri.find_simplex(objects_to_forecast_standardized)
        corrected_tri_mesh = filter_triangles(points[:, [0, 1]], self._tri.simplices, self.h_cutoff)
        retained_simplices_mask = np.array([np.any([np.all(tri_full == tri_trunc) for tri_trunc in corrected_tri_mesh])
                                            for tri_full in self._tri.simplices])
        retained_simplices_inds = np.arange(self._tri.simplices.shape[0])[retained_simplices_mask]
        failed_interpolation_mask = ~np.isin(simplices_inds, retained_simplices_inds)
        ntg_forecast = pd.DataFrame(
            {'X': objects_to_forecast_standardized[:, 0],
             'Y': objects_to_forecast_standardized[:, 1],
             'NTG': self._ip(objects_to_forecast_standardized)
             }
        )
        ntg_forecast.loc[failed_interpolation_mask, 'NTG'] = np.nan

        if whole is True:
            triangles = [points[self._tri.simplices[i]]
                         if i in retained_simplices_inds else np.array([None, None, None])
                         for i in simplices_inds]
            ntg_forecast['Distance'] = None
            ntg_forecast['Vertex1'] = [el[0] for el in triangles]
            ntg_forecast['Vertex2'] = [el[1] for el in triangles]
            ntg_forecast['Vertex3'] = [el[2] for el in triangles]

        failed_interpolation_mask = ntg_forecast['NTG'].isnull().values
        insiders = ntg_forecast[~failed_interpolation_mask]

        if failed_interpolation_mask.any():
            outsiders = ntg_forecast[failed_interpolation_mask]
            nn_pivot_points = insiders.append(self._train_data, sort=False)
            knn = KNeighborsRegressor(n_neighbors=5, weights=knn_weights_fun).fit(
                X=nn_pivot_points[['X', 'Y']].values,
                y=nn_pivot_points[self._field_name].values
            )
            distance1 = knn.kneighbors(X=outsiders[['X', 'Y']].values, n_neighbors=1, return_distance=True)
            distance = [el[0] for el in distance1[0]]

            nn_forecasted_partial_ntg_map = pd.DataFrame(
                {'X': outsiders['X'].values,
                 'Y': outsiders['Y'].values,
                 self._field_name: knn.predict(outsiders[['X', 'Y']].values)
                 }
            )

            if whole is True:
                nn_forecasted_partial_ntg_map['Distance'] = distance
                nn_forecasted_partial_ntg_map['Vertex1'] = None
                nn_forecasted_partial_ntg_map['Vertex2'] = None
                nn_forecasted_partial_ntg_map['Vertex3'] = None
            ntg_forecast = insiders.append(nn_forecasted_partial_ntg_map, ignore_index=True, sort=False)

        return ntg_forecast


def filter_triangles_1_run(xy: np.array, triangles: np.array, h_cutoff: float):
    sides_3 = np.zeros([triangles.shape[0], 3, 2], dtype=int)
    for i, t in enumerate(triangles):
        sides_3[i, :, :] = np.sort(np.array(list(combinations(t, 2))), axis=1)

    sides_2 = pd.DataFrame(sides_3.reshape([sides_3.shape[0] * sides_3.shape[1], 2]), columns=['p1', 'p2'])
    sides_2['triangle'] = np.repeat(np.arange(0, sides_3.shape[0]), 3)
    sides_2['ext_side'] = np.repeat(np.array([[0, 1, 2]]), sides_3.shape[0], axis=0). \
        reshape([sides_3.shape[0] * sides_3.shape[1]])
    sides_2 = sides_2.sort_values(['p1', 'p2'])

    pair_mask = (sides_2[['p1', 'p2']].values[1:, :] == sides_2[['p1', 'p2']].values[:-1, :]).all(axis=1)
    s1_mask = np.insert(pair_mask, 0, False)
    s2_mask = np.append(pair_mask, False)
    internal_mask = s1_mask | s2_mask

    hull_info = sides_2.loc[~internal_mask, :]
    sides_list = [[0, 1, 2] for i in range(len(hull_info))]
    [sides_list[i].remove(s) for i, s in enumerate(hull_info['ext_side'].values)]
    sides_list = np.array(sides_list)
    hull_info.insert(4, 'in_1', sides_list[:, 0])
    hull_info.insert(5, 'in_2', sides_list[:, 1])

    # if there are triangles with 2 external sides then they should be retained in triangulation, i.e. excluded
    # from consideration by algorithm
    hull_info = hull_info.sort_values(['triangle'])
    adjacent_comparison_mask = (hull_info['triangle'].values[1:] == hull_info['triangle'].values[:-1])
    s1_mask = np.insert(adjacent_comparison_mask, 0, False)
    s2_mask = np.append(adjacent_comparison_mask, False)
    two_times_mask = s1_mask | s2_mask
    hull_info = hull_info.loc[~two_times_mask, :]

    in_1 = sides_3[hull_info['triangle'], hull_info['in_1'], :]
    in_2 = sides_3[hull_info['triangle'], hull_info['in_2'], :]
    common_points_mask = np.array([np.isin(i1, i2) for i1, i2 in zip(in_1, in_2)])
    common_points = in_1[common_points_mask]

    # place common point on the 1st place
    in_1[in_1[:, 0] != common_points, :] = in_1[in_1[:, 0] != common_points, ::-1]
    in_2[in_2[:, 0] != common_points, :] = in_2[in_2[:, 0] != common_points, ::-1]

    vec_1 = xy[in_1, :][:, 1, :] - xy[in_1, :][:, 0, :]
    vec_2 = xy[in_2, :][:, 1, :] - xy[in_2, :][:, 0, :]

    bad_triangles_mask = np.arccos(
        np.sum(np.multiply(vec_1, vec_2), axis=1) / (norm(vec_1, axis=1) * norm(vec_2, axis=1))) >= \
        np.pi - 2 * np.arctan(h_cutoff / 0.5)
    retained_triangles_inds = np.delete(np.arange(triangles.shape[0]),
                                        hull_info.loc[bad_triangles_mask, 'triangle'])

    # if there are "bad" triangles with adjacent sides, then user should be signalled to
    stacked_sides = np.concatenate(
        (in_1[bad_triangles_mask, :],
         in_2[bad_triangles_mask, :]),
        axis=0
    )
    sides_adjacency_mask = np.repeat(False, stacked_sides.shape[0])
    for i, side in enumerate(stacked_sides):
        sides_adjacency_mask[i] = np.any(np.all(
            np.array([np.isin(side, s) for s in np.delete(stacked_sides, i, axis=0)]),
            axis=1
        ))
    triangles_adjacency_mask = \
        sides_adjacency_mask[:int(sides_adjacency_mask.size / 2)] | \
        sides_adjacency_mask[int(sides_adjacency_mask.size / 2):]
    if np.any(triangles_adjacency_mask):
        print("WARNING: grid contains adjacent narrow triangles. It will spoil rerformance.\n")

    return triangles[retained_triangles_inds, :]


def filter_triangles(xy: np.array, triangles: np.array, h_cutoff: float):
    retained_triangles_1 = triangles
    retained_triangles_2 = filter_triangles_1_run(xy, retained_triangles_1, h_cutoff)
    while retained_triangles_2.shape[0] != retained_triangles_1.shape[0]:
        retained_triangles_1 = retained_triangles_2
        retained_triangles_2 = filter_triangles_1_run(xy, retained_triangles_1, h_cutoff)
    return retained_triangles_2


class KrigingForecaster:
    def __init__(
            self,
            ntg_transformer=IdentityTransform,
            train_data: Optional[pd.DataFrame] = None,
            grid_builder: Optional[callable] = None
    ):
        if train_data is not None:
            self.train_data = train_data
        else:
            self._train_data = None
        self.grid_builder = grid_builder
        self.ntg_transformer = ntg_transformer

    taxon = Taxonomy.deterministic

    @property
    def train_data(self):
        return self._train_data

    @train_data.setter
    def train_data(self, data: pd.DataFrame):
        self._train_data = data.copy()
        sample_mean_train_ntg = np.mean(self._train_data['NTG'].values)
        upper_bound_amplitude = ((0.5 - abs(0.5 - sample_mean_train_ntg)) / 3.7) ** 2
        # for grounding see:
        # from scipy.stats import norm
        # norm.cdf(-3.7) * 800
        kernel = ConstantKernel(constant_value=upper_bound_amplitude / 4,
                                constant_value_bounds=(1e-05, upper_bound_amplitude)) *\
            RBF(length_scale=20, length_scale_bounds=(2, 40)) +\
            WhiteKernel(noise_level=16e-4, noise_level_bounds=(1e-05, upper_bound_amplitude / 5))
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=50
        )
        self.gp.fit(self._train_data[['X', 'Y']].values, self.ntg_transformer.direct(self._train_data['NTG'].values))

    def forecast(self, objects_to_forecast: Union[pd.DataFrame, np.array], whole: bool = False):
        if self._train_data is None:
            raise AttributeError('Kriging forecaster is incomplete. Set train_data before invoking forecast')

        points = np.array(self._train_data[['X', 'Y', 'NTG']])

        if isinstance(objects_to_forecast, pd.DataFrame):
            objects_to_forecast_standardized = objects_to_forecast[['X', 'Y']].values
        elif isinstance(objects_to_forecast, np.ndarray):
            objects_to_forecast_standardized = objects_to_forecast
        else:
            raise TypeError(
                'Unexpected type of objects_to_forecast: {}.Only pandas.DataFrame and numpy.array are allowed.'.
                format(type(objects_to_forecast))
            )

        if whole:
            y_pred, y_std = self.gp.predict(objects_to_forecast_standardized, return_std=True)
            ntg_forecast = pd.DataFrame(
                {'X': objects_to_forecast_standardized[:, 0],
                 'Y': objects_to_forecast_standardized[:, 1],
                 'NTG': y_pred,
                 'Std': y_std
                 }
            )
        else:
            y_pred = self.gp.predict(objects_to_forecast_standardized)
            ntg_forecast = pd.DataFrame(
                {'X': objects_to_forecast_standardized[:, 0],
                 'Y': objects_to_forecast_standardized[:, 1],
                 'NTG': y_pred
                 }
            )

        return ntg_forecast

    def __call__(self, train_data: Optional[pd.DataFrame] = None, whole: bool = False):
        """This is a convenience function meant to be invoked by MAP-action of multiprocess pool"""
        if self.grid_builder is None:
            raise AttributeError('Kriging forecaster is incomplete. Reinitialize it with grid_builder set.')

        if train_data is None:
            if self.train_data is None:
                raise AttributeError(
                    'train_data is not provided as argument. Baseline forecaster doesn`t incapsulate train_data either.'
                )
        else:
            self.train_data = train_data

        objects_to_forecast: np.array = self.grid_builder(train_data)
        return self.forecast(objects_to_forecast, whole)


class KrigingForecasterPredefHyperparameters(KrigingForecaster):
    kernel = RBF(length_scale=4)

    @property
    def train_data(self):
        return self._train_data

    @train_data.setter
    def train_data(self, data: pd.DataFrame):
        self._train_data = data.copy()
        # sigma2 = np.var(self.ntg_transformer.direct(self._train_data['NTG'].values)) * \
        #     self._train_data.shape[0] / (self._train_data.shape[0] - 1)

        self.gp = GaussianProcessRegressor(
            kernel=ConstantKernel(0.0049) * self.kernel + WhiteKernel(noise_level=16e-4),
            normalize_y=True,
            optimizer=None
        )
        self.gp.fit(self._train_data[['X', 'Y']].values, self.ntg_transformer.reverse(self._train_data['NTG'].values))


class UniversalKrigingForecaster(KrigingForecaster):
    @property
    def train_data(self):
        return self._train_data

    @train_data.setter
    def train_data(self, data: pd.DataFrame):
        self._train_data = data.copy()
        self.kriger = UniversalKriging(
            self._train_data['X'].values,
            self._train_data['Y'].values,
            self.ntg_transformer.direct(self._train_data['NTG'].values),
            variogram_model='gaussian',
            drift_terms=['regional_linear']
        )

    def forecast(self, objects_to_forecast: Union[pd.DataFrame, np.array], whole: bool = False):
        if self._train_data is None:
            raise AttributeError('Kriging forecaster is incomplete. Set train_data before invoking forecast')

        points = np.array(self._train_data[['X', 'Y', 'NTG']])

        if isinstance(objects_to_forecast, pd.DataFrame):
            objects_to_forecast_standardized = objects_to_forecast[['X', 'Y']].values
        elif isinstance(objects_to_forecast, np.ndarray):
            objects_to_forecast_standardized = objects_to_forecast
        else:
            raise TypeError(
                'Unexpected type of objects_to_forecast: {}.Only pandas.DataFrame and numpy.array are allowed.'.
                format(type(objects_to_forecast))
            )

        y_pred, y_std = self.kriger.execute(
            'points',
            objects_to_forecast_standardized[:, 0].astype(float),
            objects_to_forecast_standardized[:, 1].astype(float)
        )
        if y_pred.mask.any() or y_std.mask.any():
            raise ValueError('Invalid values exist')

        if whole:
            ntg_forecast = pd.DataFrame(
                {'X': objects_to_forecast_standardized[:, 0],
                 'Y': objects_to_forecast_standardized[:, 1],
                 'NTG': y_pred.data,
                 'Std': y_std.data
                 }
            )
        else:
            ntg_forecast = pd.DataFrame(
                {'X': objects_to_forecast_standardized[:, 0],
                 'Y': objects_to_forecast_standardized[:, 1],
                 'NTG': y_pred.data
                 }
            )

        return ntg_forecast


class UniversalPyKrigingForecaster(KrigingForecaster):
    @property
    def train_data(self):
        return self._train_data

    @train_data.setter
    def train_data(self, data: pd.DataFrame):
        self._train_data = data.copy()
        self.kriger = pyKriging.kriging(
            self._train_data[['X', 'Y']].values,
            self.ntg_transformer.direct(self._train_data['NTG'].values)
        )
        self.kriger.train()

    def forecast(self, objects_to_forecast: Union[pd.DataFrame, np.array], *args, **kwargs):
        if self._train_data is None:
            raise AttributeError('Kriging forecaster is incomplete. Set train_data before invoking forecast')

        points = np.array(self._train_data[['X', 'Y', 'NTG']])

        if isinstance(objects_to_forecast, pd.DataFrame):
            objects_to_forecast_standardized = objects_to_forecast[['X', 'Y']].values
        elif isinstance(objects_to_forecast, np.ndarray):
            objects_to_forecast_standardized = objects_to_forecast
        else:
            raise TypeError(
                'Unexpected type of objects_to_forecast: {}.Only pandas.DataFrame and numpy.array are allowed.'.
                format(type(objects_to_forecast))
            )

        y_pred = np.array([self.kriger.predict([x, y]) for x, y in zip(
            np.ravel(objects_to_forecast_standardized[:, 0]),
            np.ravel(objects_to_forecast_standardized[:, 1])
        )])

        ntg_forecast = pd.DataFrame(
            {'X': objects_to_forecast_standardized[:, 0],
             'Y': objects_to_forecast_standardized[:, 1],
             'NTG': y_pred
             }
        )

        return ntg_forecast


class PolynomialRegression:
    def __init__(
            self,
            train_data: Optional[pd.DataFrame] = None,
            grid_builder: Optional[callable] = None
    ):
        if train_data is not None:
            self.train_data = train_data
        else:
            self._train_data = None
        self.grid_builder = grid_builder

    taxon = Taxonomy.probabilistic  # does not try to mirror learning sample in a forecast

    @property
    def train_data(self):
        return self._train_data

    @train_data.setter
    def train_data(self, data: pd.DataFrame):
        self._train_data = data.copy()
        if data.shape[0] < 6:
            self.polynomial_features_transformer = PolynomialFeatures(degree=1)
        elif data.shape[0] < 12:
            self.polynomial_features_transformer = PolynomialFeatures(degree=2)
        else:
            self.polynomial_features_transformer = PolynomialFeatures(degree=3)

        self.regressor = LinearRegression()
        self.regressor.fit(
            self.polynomial_features_transformer.fit_transform(self._train_data[['X', 'Y']].values),
            self._train_data['NTG'].values
        )

    def forecast(self, objects_to_forecast: Union[pd.DataFrame, np.array], whole: bool = False):
        if self._train_data is None:
            raise AttributeError('Polynomial forecaster is incomplete. Set train_data before invoking forecast')

        points = np.array(self._train_data[['X', 'Y', 'NTG']])

        if isinstance(objects_to_forecast, pd.DataFrame):
            objects_to_forecast_standardized = objects_to_forecast[['X', 'Y']].values
        elif isinstance(objects_to_forecast, np.ndarray):
            objects_to_forecast_standardized = objects_to_forecast
        else:
            raise TypeError(
                'Unexpected type of objects_to_forecast: {}.Only pandas.DataFrame and numpy.array are allowed.'.
                format(type(objects_to_forecast))
            )

        y_orig = self.regressor.predict(
            self.polynomial_features_transformer.transform(objects_to_forecast_standardized)
        )
        y_pred = y_orig
        y_pred[y_pred > 1] = 1
        y_pred[y_pred < 0] = 0

        if whole:
            ntg_forecast = pd.DataFrame(
                {'X': objects_to_forecast_standardized[:, 0],
                 'Y': objects_to_forecast_standardized[:, 1],
                 'NTG': y_pred,
                 'NTG_orig': y_orig
                 }
            )
        else:
            ntg_forecast = pd.DataFrame(
                {'X': objects_to_forecast_standardized[:, 0],
                 'Y': objects_to_forecast_standardized[:, 1],
                 'NTG': y_pred
                 }
            )

        return ntg_forecast

    def __call__(self, train_data: Optional[pd.DataFrame] = None, whole: bool = False):
        """This is a convenience function meant to be invoked by MAP-action of multiprocess pool"""
        if self.grid_builder is None:
            raise AttributeError('Polynomial forecaster is incomplete. Reinitialize it with grid_builder set.')

        if train_data is None:
            if self.train_data is None:
                raise AttributeError(
                    'train_data is not provided as argument. Polynomial forecaster doesn`t incapsulate train_data either.'
                )
        else:
            self.train_data = train_data

        objects_to_forecast: np.array = self.grid_builder(train_data)
        return self.forecast(objects_to_forecast, whole)
