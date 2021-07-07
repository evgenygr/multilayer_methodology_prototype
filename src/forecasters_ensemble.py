from typing import List, Dict, Optional
import pandas as pd
from src.constants import porosity, thickness, top_surf, bot_surf
from src.generalized_train_data_setter import GeneralizedTrainDataSetter, DestinationOfGeneralizedFunction

set_train_data = GeneralizedTrainDataSetter(DestinationOfGeneralizedFunction.forecasters_ensemble)


class ForecastersEnsemble:
    def __init__(
        self, layers_to_forecasters: List[Dict],
        train_data: Optional[pd.DataFrame] = None,
        grid_builder: Optional[callable] = None
    ):
        self._layers_to_forecasters = layers_to_forecasters
        self.train_data = train_data  # see the setter of train_data below
        self.grid_builder = grid_builder

    @property
    def layers_to_forecasters(self):
        return self._layers_to_forecasters

    @property
    def train_data(self):
        return self._train_data

    @train_data.setter
    def train_data(self, train_data: pd.DataFrame):
        if train_data is None:
            raise NotImplementedError('option to drop train data is not implemented')
        else:
            self._train_data = train_data
            set_train_data(train_data, self._layers_to_forecasters)

    @property
    def grid_builder(self):
        # just the very first forecaster in a warehouse
        f = list(self._layers_to_forecasters[0]['q2f'].values())[0]
        return f.grid_builder

    @grid_builder.setter
    def grid_builder(self, grid_builder: Optional[callable]):
        for il, layer_rep in enumerate(self._layers_to_forecasters):
            for f in layer_rep['q2f'].values():
                f.grid_builder = grid_builder

    @staticmethod
    def _reformat_df(df: pd.DataFrame, layer_name: str, quantity_name: str):
        df.set_index(keys=['X', 'Y'], drop=True, inplace=True)
        df.columns = pd.MultiIndex.from_tuples(
            [(layer_name, quantity_name)],
            names=['layer', 'quantity']
        )

    def __call__(self, train_data: Optional[pd.DataFrame] = None):
        if train_data is None:
            if self.train_data is None:
                raise AttributeError(
                    'train_data is not provided as argument.\
 Forecasters ensemble doesn`t incapsulate train_data either.'
                )
        else:
            self.train_data = train_data

        interpolated_data = self._layers_to_forecasters[0]['q2f'][top_surf]()
        self._reformat_df(interpolated_data, self._layers_to_forecasters[0]['name'], top_surf)

        for il, layer_rep in enumerate(self._layers_to_forecasters):
            if isinstance(layer_rep['name'], tuple):
                layer_data = layer_rep['q2f'][thickness]()
                self._reformat_df(layer_data, layer_rep['name'][1], top_surf)
                interpolated_data = interpolated_data.merge(layer_data, left_index=True, right_index=True, sort=True)
                interpolated_data[(layer_rep['name'][1], top_surf)] =\
                    interpolated_data[(layer_rep['name'][0], bot_surf)] -\
                    interpolated_data[(layer_rep['name'][1], top_surf)]
            else:
                # thickness data incorporation
                layer_data = layer_rep['q2f'][thickness]()
                self._reformat_df(layer_data, layer_rep['name'], bot_surf)
                interpolated_data = interpolated_data.merge(layer_data, left_index=True, right_index=True, sort=True)
                interpolated_data[(layer_rep['name'], bot_surf)] =\
                    interpolated_data[(layer_rep['name'], top_surf)] -\
                    interpolated_data[(layer_rep['name'], bot_surf)]

                # porosity data incorporation
                layer_data = layer_rep['q2f'][porosity]()
                self._reformat_df(layer_data, layer_rep['name'], porosity)
                interpolated_data = interpolated_data.merge(layer_data, left_index=True, right_index=True, sort=True)
        return interpolated_data
