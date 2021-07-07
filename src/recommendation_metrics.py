from typing import Optional
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import pandas as pd
from itertools import combinations
from multiprocessing import Pool
from functools import partial
from src.weight_functions_ensembles import WeightFunctionsEnsemble as WFE, EnsembleScalingMode
from src.forecasters import KrigingForecaster
from src.constants import recommend_metrics_name


class Proximity:
    def __init__(self, interpolator, partial_field_map: Optional[pd.DataFrame],
                 weight_class=None, params=None, ensemble_mode=EnsembleScalingMode.individual, logger=None,
                 **kwargs):
        self.interpolator = interpolator
        self._partial_field_map = partial_field_map
        self.logger = logger

        if weight_class is not None:
            grid_coords = self.interpolator.grid_builder(partial_field_map)
            if partial_field_map is not None and not partial_field_map.shape[0] == 0:
                pivot_coords = partial_field_map[['X', 'Y']].values
            else:
                pivot_coords = None

            if params is None:
                params = dict()
            self.wfe = WFE(
                weight_class, pivot_coords=pivot_coords, grid_coords=grid_coords,
                scaling_mode=ensemble_mode, params=params)
        else:
            self.wfe = None

    def __call__(self):
        variance_report = self.interpolator(self._partial_field_map, whole=True)
        # whole_results.rename(columns={"NTG": "NTG_all"})

        # build a set of weighted delta-NTG samples
        if self.wfe is not None:
            variance_report[recommend_metrics_name] = self.wfe(variance_report[['X', 'Y']].values)
        else:
            variance_report[recommend_metrics_name] = np.ones(variance_report.shape[0])

        return variance_report

    @property
    def partial_field_map(self):
        return self._partial_field_map

    @partial_field_map.setter
    def partial_field_map(self, partial_field_map: Optional[pd.DataFrame]):
        if partial_field_map is not None and not partial_field_map.shape[0] == 0:
            pivot_coords = partial_field_map[['X', 'Y']].values
            self._partial_field_map = partial_field_map
        else:
            pivot_coords = None
            self._partial_field_map = None

        if self.wfe is not None:
            self.wfe.grid_coords = self.interpolator.grid_builder(self._partial_field_map)
            self.wfe.pivot_coords = pivot_coords

    @property
    def train_data(self):
        return self._partial_field_map

    @train_data.setter
    def train_data(self, train_data: Optional[pd.DataFrame]):
        self.partial_field_map = train_data


class WeightedMaxStd(Proximity):
    def __init__(self, interpolator, decision_relevant_val, partial_field_map: Optional[pd.DataFrame],
                 weight_class=None, params=None, ensemble_mode=EnsembleScalingMode.individual, logger=None):
        super().__init__(
            interpolator, partial_field_map=partial_field_map, weight_class=weight_class,
            params=params, ensemble_mode=ensemble_mode, logger=logger)
        self.decision_relevant_val = decision_relevant_val

    def __call__(self):
        whole_results = super().__call__()
        whole_results.rename(columns={recommend_metrics_name: recommend_metrics_name+'_superclass'}, inplace=True)

        well_names_train = self._partial_field_map.apply(lambda x: '{}_{}'.format(x['X'], x['Y']), axis=1)
        cross_valid_ntg_maps_sequence = []
        for max_train_wells in range(len(well_names_train) - 1, len(well_names_train)):
            cross_valid_ntg_maps_sequence += [
                self._partial_field_map[well_names_train.isin(well_names_comb)].reset_index(drop=True)
                for well_names_comb in combinations(list(well_names_train), max_train_wells)
            ]

        if self.decision_relevant_val != 'NTG':
            forecaster_ = partial(self.interpolator, whole=True)
        else:
            forecaster_ = self.interpolator

        with Pool(5) as p:
            cross_valid_results = p.map(forecaster_, cross_valid_ntg_maps_sequence)
        # cross_valid_results = forecaster(cross_valid_ntg_maps_sequence[0])

        variance_report = pd.DataFrame()

        lno_relevant_columns = [f'{self.decision_relevant_val}_{j}' for j in range(len(cross_valid_results))]
        for j, df_new_input in enumerate(cross_valid_results):
            df_new_input_ = df_new_input[['X', 'Y', self.decision_relevant_val]].copy()
            df_new_input_.columns = ['X', 'Y', lno_relevant_columns[j]]
            if j == 0:
                variance_report = df_new_input_
            else:
                variance_report = variance_report.merge(df_new_input_, on=['X', 'Y'], how='inner')

        variance_report = variance_report.merge(whole_results, on=['X', 'Y'], how='inner')
        variance_report[recommend_metrics_name] = variance_report[lno_relevant_columns].var(axis=1)

        # build a set of weighted delta-NTG samples
        if self.wfe is not None:
            variance_report[recommend_metrics_name] = \
                variance_report[recommend_metrics_name] * variance_report[recommend_metrics_name+'_superclass']

        return variance_report


class WeightedLinValidity(Proximity):
    def __init__(self, interpolator, decision_relevant_val, partial_field_map: Optional[pd.DataFrame],
                 weight_class=None, params=None, ensemble_mode=EnsembleScalingMode.individual, logger=None):
        super().__init__(
            interpolator, partial_field_map=partial_field_map, weight_class=weight_class,
            params=params, ensemble_mode=ensemble_mode, logger=logger)
        self.decision_relevant_val = decision_relevant_val

    def semidecision(self):
        lin_valid_report = deepcopy(self._partial_field_map)
        linearization_validity_map = []

        for s in lin_valid_report.index:
            self.interpolator.train_data = lin_valid_report.loc[lin_valid_report.index != s, :]
            loo_interpolated = self.interpolator.forecast(lin_valid_report.loc[[s]])

            linearization_validity_map.append(
                loo_interpolated.loc[0, self.interpolator.field_name]
            )

        lin_valid_report[recommend_metrics_name] = \
            np.abs(np.array(linearization_validity_map) - lin_valid_report[self.interpolator.field_name].values)

        return lin_valid_report

    def __call__(self):
        whole_results = super().__call__()
        whole_results.rename(columns={recommend_metrics_name: recommend_metrics_name+'_superclass'}, inplace=True)

        lin_valid_report = self.semidecision()
        lin_valid_interpolated = self.interpolator(lin_valid_report[['X', 'Y', recommend_metrics_name]], whole=False)

        variance_report = lin_valid_interpolated.merge(whole_results, on=['X', 'Y'], how='inner')

        # build a set of weighted delta-NTG samples
        if self.wfe is not None:
            variance_report[recommend_metrics_name] = \
                variance_report[recommend_metrics_name] * variance_report[recommend_metrics_name+'_superclass']
        variance_report.drop(labels=recommend_metrics_name+'_superclass', axis='columns', inplace=True)

        return variance_report


class GPSpecific:
    def __init__(
        self, interpolator: KrigingForecaster, decision_relevant_val, partial_field_map: Optional[pd.DataFrame],
        logger=None, **kwargs
    ):
        self.interpolator = interpolator
        self.decision_relevant_val = decision_relevant_val
        self._partial_field_map = partial_field_map
        self.logger = logger

    def __call__(self):
        results = self.interpolator(self._partial_field_map, whole=False)
        kernel_squared_amplitude = self.interpolator.gp.kernel_.k1.k1.get_params()['constant_value']
        kernel_length_scale = self.interpolator.gp.kernel_.k1.k2.get_params()['length_scale']
        noise_level = self.interpolator.gp.kernel_.k2.get_params()['noise_level']
        if self.logger is not None:
            self.logger.info('Inferred parameters of GP kernel:\n'
                  f'\tLength scale: {kernel_length_scale},\n'
                  f'\tSquared amplitude: {kernel_squared_amplitude},\n'
                  f'\tNoise variance: {noise_level}')

        expected_rmse = np.zeros(results.shape[0])
        if self.logger is not None:
            self.logger.info('Well placement variants assessment in progress.')
            wrapped_iterator = tqdm(range(results.shape[0]))
        else:
            wrapped_iterator = range(results.shape[0])

        for i in wrapped_iterator:
            step_ahead_train_grid = np.concatenate(
                (self._partial_field_map[['X', 'Y']].values,
                 results.loc[i, ['X', 'Y']].values.reshape(1, 2)),
                axis=0
            )
            step_ahead_appraisal_grid = np.delete(results[['X', 'Y']].values, i, axis=0)

            train_cov_matr = self.interpolator.gp.kernel_(step_ahead_train_grid)  # T x T
            cross_sets_cov = self.interpolator.gp.kernel_(step_ahead_appraisal_grid, step_ahead_train_grid)  # D x T
            m = cross_sets_cov @ np.linalg.inv(train_cov_matr)  # D x T
            conditioning_correction = (
                                              m @ cross_sets_cov.T
                                      ).trace() / step_ahead_appraisal_grid.shape[0]
            expected_rmse[i] = np.sqrt(
                kernel_squared_amplitude + noise_level - conditioning_correction +
                ((np.sum((m.T @ m).ravel()) + 2 * np.sum(m.ravel())) / step_ahead_appraisal_grid.shape[0] + 1) *
                np.sum(train_cov_matr.ravel()) / train_cov_matr.shape[0] ** 2
            )

        results[recommend_metrics_name] = -expected_rmse
        return results

    @property
    def partial_field_map(self):
        return self._partial_field_map

    @partial_field_map.setter
    def partial_field_map(self, partial_field_map: Optional[pd.DataFrame]):
        if partial_field_map is not None and not partial_field_map.shape[0] == 0:
            self._partial_field_map = partial_field_map
        else:
            self._partial_field_map = None

    @property
    def train_data(self):
        return self._partial_field_map

    @train_data.setter
    def train_data(self, train_data: Optional[pd.DataFrame]):
        self.partial_field_map = train_data
