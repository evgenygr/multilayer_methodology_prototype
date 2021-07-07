from os import getcwd, makedirs
from os.path import exists, join
from src.constants import project_root, column_levels, value_of_quantity
from sklearn.metrics import mean_squared_error
from math import sqrt
from enum import Enum
from typing import List, Optional
import numpy as np
import pandas as pd


class DataDump:
    def __init__(self, overall_results_folder, res_dir_template, variance_report_template, mode=None, active=True):
        self.active = active
        if self.active:
            if isinstance(mode, Enum):
                mode = mode.value
            if res_dir_template is not None and not isinstance(res_dir_template, str) or\
               not isinstance(overall_results_folder, str):
                raise ValueError('invalid combination of overall_results_folder and res_dir_template')

            if isinstance(res_dir_template, str) and isinstance(overall_results_folder, str):
                if mode is None or mode == '':
                    overall_results_folder = join(overall_results_folder, res_dir_template.format(''))
                else:
                    overall_results_folder = join(overall_results_folder, res_dir_template.format(f'_{mode}'))
                if not getcwd().endswith(project_root):
                    overall_results_folder = join('..', overall_results_folder)

            if not exists(overall_results_folder):
                makedirs(overall_results_folder)
            self.overall_results_folder = overall_results_folder
            self.variance_report_template = variance_report_template

    def __call__(self, table: pd.DataFrame, file_mark):
        if self.active:
            if isinstance(file_mark, tuple) and len(file_mark) == 2:
                variance_report_file = self.variance_report_template.format(format(file_mark[1], '03'), file_mark[0]+'_')
            else:
                variance_report_file = self.variance_report_template.format(format(file_mark, '03'), '')
            if isinstance(table.index, pd.MultiIndex):
                table.to_csv(join(self.overall_results_folder, variance_report_file), sep=' ')
            else:
                table.to_csv(join(self.overall_results_folder, variance_report_file), index=False)

    def __enter__(self):
        return self

    def __exit__(self, type_, value, traceback):
        pass


class ConciseDataDump:
    def __init__(
            self, res_file_path: str, pnum_seq: List[int],
            profile_names: List[str], reference_df: pd.DataFrame,
            current_name: Optional[str] = None, target_quantity: str = 'NTG', active=True
    ):
        profile_length = len(pnum_seq)
        self.active = active
        if active:
            self._target_quantity = target_quantity
            self._reference_series = self.digest_df(reference_df)
            self._profile_length = profile_length
            self.res_file_path = res_file_path
            self.particular_profile = pd.DataFrame(
                np.eye(profile_length, len(profile_names)) * np.nan, columns=profile_names
            )
            self.particular_profile['pnum'] = pnum_seq
            self.current_name = current_name
            self._current_ind = {n: 0 for n in profile_names}

    def __call__(self, var_rep: pd.DataFrame):
        if self.active:
            if self.current_name is None:
                raise AttributeError('current_name should be set before calling an instance')
            if self._current_ind[self.current_name] > self._profile_length:
                raise IndexError('profile has exhausted its preallocated capacity')

            self.particular_profile.loc[
                self._current_ind[self.current_name],
                self.current_name
            ] = sqrt(mean_squared_error(self.digest_df(var_rep), self._reference_series[var_rep.index]))
            self._current_ind[self.current_name] += 1

    def __enter__(self):
        return self

    def __exit__(self, type_, value, traceback):
        if self.active and not self.particular_profile.isna().any().any():
            self.particular_profile.to_csv(self.res_file_path, index=False)

    def digest_df(self, orig_df: pd.DataFrame):
        df = orig_df.copy()
        if not isinstance(df.index, pd.MultiIndex):
            df.set_index(keys=['X', 'Y'], drop=True, inplace=True)
        if not isinstance(df.columns, pd.MultiIndex):
            return df[self._target_quantity]
        elif df.columns.names == column_levels[:2]:
            return df[('', self._target_quantity)]
        elif df.columns.names == column_levels:
            return df[('', self._target_quantity, value_of_quantity)]
        else:
            raise ValueError(f'incorrect names for columns levels: {df.columns.names}')
