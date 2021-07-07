from enum import Enum
from typing import List, Dict
import pandas as pd
from src.weight_functions_ensembles import EnsembleScalingMode
from src.constants import recommend_metrics_name, porosity, thickness, q2m, q2f, value_of_quantity, column_levels,\
    top_surf, bot_surf
from src.generalized_train_data_setter import GeneralizedTrainDataSetter, DestinationOfGeneralizedFunction

set_train_data = GeneralizedTrainDataSetter(DestinationOfGeneralizedFunction.composite_metrics)


class CompositeMetricsTargets(Enum):
    ntg = 'ntg'
    grv = 'grv'


def _reduce_to_ntg(df: pd.DataFrame):
    prod_layers = [ll for ll in df.columns.get_level_values(column_levels[0]).unique().to_list() if isinstance(ll, str)]
    if df.columns.names == column_levels:
        df_prod = df.loc[:, (prod_layers, slice(None), slice(None))]
        df_thickness_val = df_prod.xs((thickness, value_of_quantity), level=column_levels[1:], axis=1)
        df_thickness_metrics = df_prod.xs((thickness, recommend_metrics_name), level=column_levels[1:], axis=1)
        df_porosity_val = df_prod.xs((porosity, value_of_quantity), level=column_levels[1:], axis=1)
        df_porosity_metrics = df_prod.xs((porosity, recommend_metrics_name), level=column_levels[1:], axis=1)
        df[('', CompositeMetricsTargets.ntg.value, value_of_quantity)] = \
            (df_thickness_val * df_porosity_val).apply(sum, axis=1)
        df[('', CompositeMetricsTargets.ntg.value, recommend_metrics_name)] = \
            (df_thickness_val * df_porosity_metrics + df_thickness_metrics * df_porosity_val).apply(sum, axis=1)

    elif df.columns.names == column_levels[:2]:
        df_prod = df.loc[:, (prod_layers, slice(None))]
        if thickness not in df_prod.columns.get_level_values(column_levels[1]):
            df_top_val = df_prod.xs(top_surf, level=column_levels[1], axis=1)
            df_bot_val = df_prod.xs(bot_surf, level=column_levels[1], axis=1)
            df_thickness_val = df_top_val - df_bot_val
        else:
            df_thickness_val = df_prod.xs(thickness, level=column_levels[1], axis=1)
        df_porosity_val = df_prod.xs(porosity, level=column_levels[1], axis=1)
        df[('', CompositeMetricsTargets.ntg.value)] = (df_thickness_val * df_porosity_val).apply(sum, axis=1)

    else:
        raise ValueError(f'incorrect names for columns levels: {df.columns.names}')


def _reduce_to_grv(df: pd.DataFrame):
    prod_layers = [ll for ll in df.columns.get_level_values(column_levels[0]).unique().to_list() if isinstance(ll, str)]
    if df.columns.names == column_levels:
        score_col_indexer = [(layer, thickness, recommend_metrics_name) for layer in prod_layers]
        var_col_indexer = [(layer, thickness, value_of_quantity) for layer in prod_layers]
        df[('', CompositeMetricsTargets.grv.value, value_of_quantity)] = df[var_col_indexer].sum(axis=1)
        df[('', CompositeMetricsTargets.grv.value, recommend_metrics_name)] = df[score_col_indexer].sum(axis=1)
    elif df.columns.names == column_levels[:2]:
        var_col_indexer = [(layer, thickness) for layer in prod_layers]
        df[('', CompositeMetricsTargets.grv.value)] = df[var_col_indexer].sum(axis=1)

    else:
        raise ValueError(f'incorrect names for columns levels: {df.columns.names}')


class CompositeMetrics:
    def __init__(
            self, forecasters_ensemble, layers_to_metrics: List[Dict],
            target_value: CompositeMetricsTargets = CompositeMetricsTargets.ntg):
        self.forecasters_ensemble = forecasters_ensemble
        self._train_data = forecasters_ensemble.train_data
        self._layers_to_metrics = layers_to_metrics
        self.target_value = target_value
        layers_to_forecasters = forecasters_ensemble.layers_to_forecasters
        for layer_rep in self._layers_to_metrics:
            for q in layer_rep[q2m].keys():
                forecaster = [l[q2f][q] for l in layers_to_forecasters if l['name'] == layer_rep['name']][0]
                # self._layers_to_metrics and layers_to_forecasters may differ by comprising nonprod layers
                layer_rep[q2m][q]['metrics'] = layer_rep[q2m][q]['metrics'](
                    interpolator=forecaster,
                    decision_relevant_val=q,
                    partial_field_map=forecaster.train_data,
                    weight_class=layer_rep[q2m][q]['weight_class'],
                    params=layer_rep[q2m][q]['params'],
                    ensemble_mode=layer_rep[q2m][q].get('wf_ensemble_mode', EnsembleScalingMode.universal)
                )

    @property
    def target_value(self):
        return self._target_value

    @target_value.setter
    def target_value(self, target_value: CompositeMetricsTargets):
        self._target_value = target_value
        if target_value is CompositeMetricsTargets.ntg:
            self._relevant_values = [thickness, porosity]
            self.reduce_to_target = _reduce_to_ntg
        else:
            self._relevant_values = [thickness]
            self.reduce_to_target = _reduce_to_grv

    @property
    def train_data(self):
        return self._train_data

    @train_data.setter
    def train_data(self, train_data: pd.DataFrame):
        if train_data is None:
            raise NotImplementedError('option to drop train data is not implemented')
        else:
            self._train_data = train_data
            set_train_data(train_data, self._layers_to_metrics)

    def __call__(self):
        summarised_data = None
        for il, layer_rep in enumerate(self._layers_to_metrics):
            for q in layer_rep[q2m].keys():
                if q in self._relevant_values:
                    focused_data = layer_rep[q2m][q]['metrics']()
                    focused_data = self._reformat_df(focused_data, layer_rep['name'], q, q)
                    if summarised_data is None:
                        summarised_data = focused_data
                    else:
                        summarised_data = summarised_data.merge(
                            focused_data, left_index=True, right_index=True, sort=True
                        )
        self.reduce_to_target(summarised_data)
        return summarised_data

    @staticmethod
    def _reformat_df(df: pd.DataFrame, layer_name: str, quantity_name: str, decision_relelvant_val: str):
        df = df[['X', 'Y', decision_relelvant_val, recommend_metrics_name]]
        df.set_index(keys=['X', 'Y'], drop=True, inplace=True)
        df.columns = pd.MultiIndex.from_tuples(
            [(layer_name, quantity_name, value_of_quantity), (layer_name, quantity_name, recommend_metrics_name)],
            names=column_levels
        )
        return df
