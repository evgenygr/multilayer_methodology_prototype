import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt


def compare_maps_by_rmse(df1: pd.DataFrame, df2: pd.DataFrame):
    df_ntg_comparison = df1[['X', 'Y', 'NTG']].merge(df2[['X', 'Y', 'NTG']], on=['X', 'Y'], how='inner')
    df_ntg_comparison.columns = ['X', 'Y', 'NTG1', 'NTG2']
    return sqrt(mean_squared_error(df_ntg_comparison['NTG1'], df_ntg_comparison['NTG2']))


def generate_padded_str_tmplt(n: int):
    padding = 1  # number of signs in formatted string based on the template
    while n >= 10 ** padding:
        padding += 1
    return '{:0>' + str(padding) + '}'


def calculate_distortion_coeff(x1, x2, der1, der2):
    m = np.array([
        [3 * x1 ** 2, 2 * x1, 1, 0],
        [3 * x2 ** 2, 2 * x2, 1, 0],
        [x1 ** 3, x1 ** 2, x1, 1],
        [x2 ** 3, x2 ** 2, x2, 1]
    ])
    left = np.array([
        [der1],
        [der2],
        [x1],
        [x2]
    ])
    return np.linalg.inv(m) @ left


def poly_distortion(x, c_vec):
    x_intern = x.flatten()
    x_matr = np.vstack((x_intern ** 3, x_intern ** 2, x_intern, np.ones(x_intern.shape)))
    return (x_matr.T @ c_vec).flatten()
