import pandas as pd
from src.constants import porosity, top_surf, thickness, bot_surf, q2f, q2m
from typing import List, Dict
from enum import Enum


class DestinationOfGeneralizedFunction(Enum):
    forecasters_ensemble = 'forecasters_ensemble'
    composite_metrics = 'composite_metrics'


def _extract_individ_train_data(train_data, lname, quantity):
    reduced_data = train_data.loc[:, (lname, quantity)]
    reduced_data = reduced_data.reset_index()
    reduced_data.columns = ['X', 'Y', quantity]
    return reduced_data


def _get_atomic_data_consumer_fe(layer_data_holder: Dict, quantity: str):  # fe stands for forecasters ensemble
    return layer_data_holder[q2f][quantity]


def _get_atomic_data_consumer_cm(layer_data_holder: Dict, quantity: str):  # cm stands for composite metrics
    return layer_data_holder[q2m][quantity]['metrics']


def _set_atomic_data_consumer_fe(  # fe stands for forecasters ensemble
        layer_data_holder: Dict, quantity: str, settable
):
    layer_data_holder[q2f][quantity] = settable


def _set_atomic_data_consumer_cm(  # cm stands for composite metrics
        layer_data_holder: Dict, quantity: str, settable
):
    layer_data_holder[q2m][quantity]['metrics'] = settable


class GeneralizedTrainDataSetter:
    def __init__(self, destination: DestinationOfGeneralizedFunction):
        if destination is DestinationOfGeneralizedFunction.forecasters_ensemble:
            self._get_atomic_data_consumer = _get_atomic_data_consumer_fe
            self._set_atomic_data_consumer = _set_atomic_data_consumer_fe
            self.q2e = q2f
        else:
            self._get_atomic_data_consumer = _get_atomic_data_consumer_cm
            self._set_atomic_data_consumer = _set_atomic_data_consumer_cm
            self.q2e = q2m

    def __call__(self, train_data: pd.DataFrame, data_holders: List[Dict]):
        for il, layer_rep in enumerate(data_holders):
            for q in layer_rep[self.q2e].keys():
                if (q == porosity and not isinstance(layer_rep['name'], (tuple, list))) or \
                   (q == top_surf and il == 0):
                    focused_data = _extract_individ_train_data(train_data, layer_rep['name'], q)
                elif q == thickness:
                    if isinstance(layer_rep['name'], (tuple, list)):
                        top_data = _extract_individ_train_data(train_data, layer_rep['name'][0], bot_surf)
                        bot_data = _extract_individ_train_data(train_data, layer_rep['name'][1], top_surf)
                        focused_data = top_data[['X', 'Y']]
                        focused_data[thickness] = top_data[bot_surf] - bot_data[top_surf]
                    else:  # thickness of productive layer
                        top_data = _extract_individ_train_data(train_data, layer_rep['name'], top_surf)
                        bot_data = _extract_individ_train_data(train_data, layer_rep['name'], bot_surf)
                        focused_data = top_data[['X', 'Y']]
                        focused_data[thickness] = top_data[top_surf] - bot_data[bot_surf]
                else:
                    raise Exception('incorrect entry in layers_to_forecasters')
                atomic_data_consumer = self._get_atomic_data_consumer(layer_rep, q)
                if isinstance(atomic_data_consumer, type):
                    self._set_atomic_data_consumer(layer_rep, q, atomic_data_consumer(train_data=focused_data))
                else:
                    atomic_data_consumer.train_data = focused_data
