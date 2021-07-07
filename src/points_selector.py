import random
from collections import namedtuple
from src.constants import recommend_metrics_name
import pandas as pd

Point = namedtuple('Point', ['X', 'Y'])


class PointsSelector:
    def __init__(self, metrics_name, logger=None):
        self.logger = logger
        self.metrics_name = metrics_name

    def __call__(self, variance_report, log_mark=''):
        max_row = variance_report[
            variance_report[self.metrics_name] == variance_report[self.metrics_name].max()
        ]
        # random.seed(a=123)
        if max_row.shape[0] > 1:
            # it is undesirable to always choose the 1st element because at unshuffled datasets
            # it may bias decision to a particular side of a field
            max_std_row = max_row.iloc[random.randint(0, (max_row.shape[0] - 1))]
            if self.logger is not None:
                self.logger.info(f'{max_row.shape[0]} highs of recommendation metrics found')
        else:
            max_std_row = max_row.iloc[0]
        if isinstance(variance_report.index, pd.MultiIndex):
            x, y = max_std_row.name
        else:
            x = max_std_row['X'].item()
            y = max_std_row['Y'].item()
        metrics_val = max_std_row[self.metrics_name].item()

        if self.logger is not None:
            self.logger.info(f'Point selection: {log_mark} : {(x, y)} : metrics_val : {metrics_val}')

        return Point(X=x, Y=y)
