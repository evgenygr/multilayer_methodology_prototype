import numpy as np
import pandas as pd
from typing import Union


class DataProvider:
    def __init__(self, path: str):
        self._df_data_source = pd.read_csv(path, sep=' ', header=[0, 1], index_col=[0, 1])
        reservoir_layers = list(self._df_data_source.columns.get_level_values('layer').unique().values)
        npl = len(reservoir_layers)  # the number of productive (reservoir) layers
        if npl == 1:
            self._nl = 1
            self._layers = reservoir_layers
        else:
            self._nl = npl * 2 - 1  # the number of all layers (productive + separating)
            self._layers = [None] * self._nl
            self._layers[0::2] = reservoir_layers
            for i in range(1, self._nl, 2):
                self._layers[i] = (self._layers[i - 1], self._layers[i + 1])

    def get_data(self, x: Union[int, float, np.ndarray, pd.MultiIndex], y=None):
        if y is None:
            if isinstance(x, np.ndarray):  # x is an Nx2 nd.array of indices
                indexer = pd.MultiIndex.from_tuples([tuple(row) for row in x], names=['X', 'Y'])
            else:
                indexer = x
        else:
            indexer = (x, y)
        return self._df_data_source.loc[indexer, :]

    @property
    def productive_layers(self):
        return self._layers[::2]

    @property
    def nonproductive_layers(self):
        return self._layers[1::2]

    @property
    def all_layers(self):
        return self._layers

    @property
    def data_source(self):
        return self._df_data_source.copy()
