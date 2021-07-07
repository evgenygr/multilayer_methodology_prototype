import random
from enum import Enum
from itertools import combinations
from multiprocessing import Pool
from os.path import join, exists
from collections import namedtuple
from os import makedirs
from copy import deepcopy
import itertools
from src.constants import MaxStdModes
from functools import partial
import numpy as np
from sklearn.neighbors import KDTree

import pandas as pd
from tqdm import tqdm

from src.constants import max_std_res_dir_template, expect_rmse_res_dir_template,\
    rnd_res_dir_template, variance_report_template
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# np.seterr(all='raise')

Point = namedtuple('Point', ['X', 'Y'])


class RandomDecisionReportingModes(Enum):
    BASIC: str = 'basic'
    LOO: str = 'leave_one_out'

"""
def max_std_based_decision_univers_weight(forecaster, partial_ntg_map: pd.DataFrame, results_folder, file_mark=None,
                                          weight_fun=None, nargout=1) -> Point:
    if file_mark is not None:
        max_std_res_path = join(results_folder, max_std_res_dir_template.format(f'_{MaxStdModes.basic.value}'))
        if not exists(max_std_res_path):
            makedirs(max_std_res_path)

    whole_results = forecaster(partial_ntg_map, whole=True)
    whole_results.rename(columns={"NTG": "NTG_all"})

    well_names_train = partial_ntg_map.Well
    cross_valid_ntg_maps_sequence = []
    for max_train_wells in range(len(well_names_train) - 1, len(well_names_train)):
        cross_valid_ntg_maps_sequence += [
            partial_ntg_map[partial_ntg_map['Well'].isin(well_names_comb)].reset_index(drop=True)
            for well_names_comb in combinations(list(well_names_train), max_train_wells)
        ]

    with Pool(5) as p:
        cross_valid_results = p.map(forecaster, cross_valid_ntg_maps_sequence)
    # cross_valid_results = forecaster(cross_valid_ntg_maps_sequence[0])

    variance_report = pd.DataFrame()

    lno_ntg_columns = ['NTG_' + str(j) for j in range(len(cross_valid_results))]
    for j, df_new_input in enumerate(cross_valid_results):
        df_new_input.columns = ['X', 'Y', lno_ntg_columns[j]]
        if j == 0:
            variance_report = df_new_input
        else:
            variance_report = variance_report.merge(df_new_input, on=['X', 'Y'], how='inner')

    variance_report = variance_report.merge(whole_results, on=['X', 'Y'], how='inner')
    variance_report['Var_score'] = variance_report[lno_ntg_columns].var(axis=1)

    # build a set of weighted delta-NTG samples
    if weight_fun is not None:
        weights_map = weight_fun(variance_report[['X', 'Y']].values, partial_ntg_map[['X', 'Y']].values)
        variance_report['Var_score'] = variance_report['Var_score'] * weights_map

    if file_mark is not None:
        variance_report_file = variance_report_template.format(file_mark)
        variance_report.to_csv(join(max_std_res_path, variance_report_file), index=False)

    df1 = deepcopy(variance_report)
    max_row = df1[df1['Var_score'] == df1['Var_score'].max()]
    # random.seed(a=123)
    if max_row.shape[0] > 1:
        # it is undesirable to always choose the 1st element because at unshuffled datasets it may bias decision to a
        # particular side of a field
        max_std_row = max_row.iloc[random.randint(0, (max_row.shape[0]-1))]
    else:
        max_std_row = max_row.iloc[0]

    if file_mark is not None:
        print(f'Max std based decision run mark: {file_mark}\n{(max_std_row["X"],max_std_row["Y"])}')

    if nargout == 1:
        return Point(X=max_std_row['X'].item(), Y=max_std_row['Y'].item())
    else:
        return Point(X=max_std_row['X'].item(), Y=max_std_row['Y'].item()), variance_report
"""

def max_std_based_decision_grid_adapted_univers_weight(
        forecaster, partial_ntg_map: pd.DataFrame, results_folder, file_mark=None,
        weight_class=None, decision_relevant_val='NTG', nargout=1
):
    neighbors_processor = KDTree(partial_ntg_map[['X', 'Y']].values, leaf_size=2)

    if file_mark is not None:
        max_std_res_path = join(results_folder, max_std_res_dir_template.format(f'_{MaxStdModes.basic.value}'))
        #max_std_res_path = results_folder
        if not exists(max_std_res_path):
            makedirs(max_std_res_path)

    whole_results = forecaster(partial_ntg_map, whole=True)
    # whole_results.rename(columns={"NTG": "NTG_all"})

    well_names_train = partial_ntg_map.Well
    cross_valid_ntg_maps_sequence = []
    for max_train_wells in range(len(well_names_train) - 1, len(well_names_train)):
        cross_valid_ntg_maps_sequence += [
            partial_ntg_map[partial_ntg_map['Well'].isin(well_names_comb)].reset_index(drop=True)
            for well_names_comb in combinations(list(well_names_train), max_train_wells)
        ]

    if decision_relevant_val != 'NTG':
        forecaster_ = partial(forecaster, whole=True)
    else:
        forecaster_ = forecaster

    with Pool(5) as p:
        cross_valid_results = p.map(forecaster_, cross_valid_ntg_maps_sequence)
    # cross_valid_results = forecaster(cross_valid_ntg_maps_sequence[0])

    variance_report = pd.DataFrame()

    lno_ntg_columns = [f'{decision_relevant_val}_{j}' for j in range(len(cross_valid_results))]
    for j, df_new_input in enumerate(cross_valid_results):
        df_new_input_ = df_new_input[['X', 'Y', decision_relevant_val]].copy()
        df_new_input_.columns = ['X', 'Y', lno_ntg_columns[j]]
        if j == 0:
            variance_report = df_new_input_
        else:
            variance_report = variance_report.merge(df_new_input_, on=['X', 'Y'], how='inner')

    variance_report = variance_report.merge(whole_results, on=['X', 'Y'], how='inner')
    variance_report['Var_score'] = variance_report[lno_ntg_columns].var(axis=1)

    # After metric is calculated, restrict it to license area polygon
    #!Note: with universal weighting, restriction should be done before weighting to avoid dissapearing metric problems
    variance_report = forecaster.grid_builder.drillable(variance_report)

    # build a set of weighted delta-NTG samples
    if weight_class is not None:
        dists, _ = neighbors_processor.query(variance_report[['X', 'Y']].values, k=1)
        characteristic_dist = dists.flatten().max() * 2/5
        weight_fun = weight_class({'offset': characteristic_dist, 'scaling': characteristic_dist})
        weights_map = weight_fun(variance_report[['X', 'Y']].values, partial_ntg_map[['X', 'Y']].values)
        variance_report['Var_score'] = variance_report['Var_score'] * weights_map

    #variance_report = forecaster.grid_builder.drillable(variance_report)

    if file_mark is not None:
        if isinstance(file_mark, str) or isinstance(file_mark, int):
            variance_report_file = variance_report_template.format('', file_mark)
        elif isinstance(file_mark, tuple) and len(file_mark) == 2:
            variance_report_file = variance_report_template.format(file_mark[0]+'_', file_mark[1])
        variance_report.to_csv(join(max_std_res_path, variance_report_file), index=False)

    df1 = deepcopy(variance_report)
    max_row = df1[df1['Var_score'] == df1['Var_score'].max()]
    # random.seed(a=123)
    if max_row.shape[0] > 1:
        # it is undesirable to always choose the 1st element because at unshuffled datasets it may bias decision to a
        # particular side of a field
        max_std_row = max_row.iloc[random.randint(0, (max_row.shape[0]-1))]
    else:
        max_std_row = max_row.iloc[0]

    if file_mark is not None:
        print(f'Max std based decision run mark: {file_mark}\n{(max_std_row["X"],max_std_row["Y"])}')

    if nargout == 1:
        return Point(X=max_std_row['X'].item(), Y=max_std_row['Y'].item())
    else:
        return Point(X=max_std_row['X'].item(), Y=max_std_row['Y'].item()), variance_report


#Same as max_std_based_decision_grid_adapted_univers_weight, except for weighting procedures
def max_std_based_decision_grid_adapted_individual_weight(
        forecaster, partial_ntg_map: pd.DataFrame, results_folder, file_mark=None,
        weight_class=None, decision_relevant_val='NTG', nargout=1
):
    neighbors_processor = KDTree(partial_ntg_map[['X', 'Y']].values, leaf_size=2)

    if file_mark is not None:
        max_std_res_path = join(results_folder, max_std_res_dir_template.format(f'_{MaxStdModes.basic.value}'))
        #max_std_res_path = results_folder
        if not exists(max_std_res_path):
            makedirs(max_std_res_path)

    whole_results = forecaster(partial_ntg_map, whole=True)
    # whole_results.rename(columns={"NTG": "NTG_all"})

    well_names_train = partial_ntg_map.Well
    cross_valid_ntg_maps_sequence = []
    for max_train_wells in range(len(well_names_train) - 1, len(well_names_train)):
        cross_valid_ntg_maps_sequence += [
            partial_ntg_map[partial_ntg_map['Well'].isin(well_names_comb)].reset_index(drop=True)
            for well_names_comb in combinations(list(well_names_train), max_train_wells)
        ]

    if decision_relevant_val != 'NTG':
        forecaster_ = partial(forecaster, whole=True)
    else:
        forecaster_ = forecaster

    with Pool(5) as p:
        cross_valid_results = p.map(forecaster_, cross_valid_ntg_maps_sequence)
    # cross_valid_results = forecaster(cross_valid_ntg_maps_sequence[0])

    variance_report = pd.DataFrame()

    lno_ntg_columns = [f'{decision_relevant_val}_{j}' for j in range(len(cross_valid_results))]
    for j, df_new_input in enumerate(cross_valid_results):
        df_new_input_ = df_new_input[['X', 'Y', decision_relevant_val]].copy()
        df_new_input_.columns = ['X', 'Y', lno_ntg_columns[j]]
        if j == 0:
            variance_report = df_new_input_
        else:
            variance_report = variance_report.merge(df_new_input_, on=['X', 'Y'], how='inner')

    variance_report = variance_report.merge(whole_results, on=['X', 'Y'], how='inner')
    variance_report['Var_score'] = variance_report[lno_ntg_columns].var(axis=1)

    # Build a set of weighted delta-NTG samples (individual: Voronoi domain maximum)
    # if weight_class is not None:
    #     dists_nearest, indices_nearest = neighbors_processor.query(variance_report[['X', 'Y']].values, k=1)
    #     relative_offset, relative_scaling = 0.4, 0.4 #change offset, scaling here
    #     dist_max_indiv = np.zeros(partial_ntg_map.shape[0])
    #     weights_map = np.ones(variance_report.shape[0])
    #     for iwell, well_coord in enumerate(partial_ntg_map[['X', 'Y']].values):
    #         # compute maximum distance for the set of points nearest to each well
    #         try: dist_max_indiv[iwell] = dists_nearest[indices_nearest == iwell].max()
    #         except: ValueError # empty set exception: no points nearest to a well (high well density) - result zero
    #         #print(iwell, variance_report[['X', 'Y']].values[iwell], dist_max_indiv[iwell])
    #         characteristic_dist = dist_max_indiv[iwell]
    #         weight_fun_well = weight_class({'offset': characteristic_dist * relative_offset,
    #                                         'scaling': characteristic_dist * relative_offset})
    #         weights_map *= weight_fun_well(variance_report[['X', 'Y']].values, well_coord)
    #     variance_report['Var_score'] = variance_report['Var_score'] * weights_map

    # Build a set of weighted delta-NTG samples (individual: n-th nearest well)
    if weight_class is not None:
        nth_nearest = 3 #4 and higher not possible (for 4 starting wells)
        dists_well, _ = neighbors_processor.query(variance_report[['X', 'Y']].values, k=nth_nearest+1)
        relative_offset, relative_scaling = 0.4, 0.4 #change offset, scaling here
        weights_map = np.ones(variance_report.shape[0])
        dist_well_indiv = dists_well[:,nth_nearest] #distance from each well to the n-th nearest well
        for iwell, well_coord in enumerate(partial_ntg_map[['X', 'Y']].values):
            #print(iwell, variance_report[['X', 'Y']].values[iwell], dist_well_indiv[iwell])
            characteristic_dist = dist_well_indiv[iwell]
            weight_fun_well = weight_class({'offset': characteristic_dist * relative_offset,
                                            'scaling': characteristic_dist * relative_scaling})
            weights_map *= weight_fun_well(variance_report[['X', 'Y']].values, well_coord)
        variance_report['Var_score'] = variance_report['Var_score'] * weights_map

    # After metric is calculated, restrict it to license area polygon
    variance_report = forecaster.grid_builder.drillable(variance_report)

    if file_mark is not None:
        if isinstance(file_mark, str) or isinstance(file_mark, int):
            variance_report_file = variance_report_template.format('', file_mark)
        elif isinstance(file_mark, tuple) and len(file_mark) == 2:
            variance_report_file = variance_report_template.format(file_mark[0]+'_', file_mark[1])
        variance_report.to_csv(join(max_std_res_path, variance_report_file), index=False)

    df1 = deepcopy(variance_report)
    max_row = df1[df1['Var_score'] == df1['Var_score'].max()]
    # random.seed(a=123)
    if max_row.shape[0] > 1:
        # it is undesirable to always choose the 1st element because at unshuffled datasets it may bias decision to a
        # particular side of a field
        max_std_row = max_row.iloc[random.randint(0, (max_row.shape[0]-1))]
    else:
        max_std_row = max_row.iloc[0]

    if file_mark is not None:
        print(f'Max std based decision run mark: {file_mark}\n{(max_std_row["X"],max_std_row["Y"])}')

    if nargout == 1:
        return Point(X=max_std_row['X'].item(), Y=max_std_row['Y'].item())
    else:
        return Point(X=max_std_row['X'].item(), Y=max_std_row['Y'].item()), variance_report


def linearization_validity_semidecision(forecaster, partial_ntg_map: pd.DataFrame) -> Point:
    lin_valid_report = deepcopy(partial_ntg_map)
    linearizaton_validity_map = []

    for s in lin_valid_report.index:
        loo_interpolated = forecaster(partial_ntg_map.loc[partial_ntg_map.index != s, :])

        linearizaton_validity_map.append(
            loo_interpolated.loc[
                (loo_interpolated["X"] == partial_ntg_map.loc[s, 'X']) &
                (loo_interpolated["Y"] == partial_ntg_map.loc[s, 'Y']),
                'NTG'
            ].values[0])

    lin_valid_report['Discrep'] = linearizaton_validity_map
    lin_valid_report['Discrep'] = lin_valid_report['Discrep'] - lin_valid_report['NTG']

    return lin_valid_report


#Uses linearization_validity_semidecision to make decision with universal weighting
def lin_val_based_decision_grid_adapted_univers_weight(
        forecaster, partial_ntg_map: pd.DataFrame, results_folder, file_mark=None,
        weight_class=None, decision_relevant_val='NTG', nargout=1
):
    neighbors_processor = KDTree(partial_ntg_map[['X', 'Y']].values, leaf_size=2)

    if file_mark is not None:
        max_std_res_path = join(results_folder, max_std_res_dir_template.format(f'_{MaxStdModes.basic.value}'))
        #max_std_res_path = results_folder
        if not exists(max_std_res_path):
            makedirs(max_std_res_path)

    # Use semidecision to compute linear validity at wells (as Discrep)
    lin_valid_report = linearization_validity_semidecision(forecaster, partial_ntg_map)
    #forecaster interpolates over NTG, so rename Discrep->NTG so that it interpolates over Discrep
    lin_valid_report = lin_valid_report.rename(columns={'NTG': 'NTG_real', 'Discrep': 'NTG'})
    lin_valid_interpolated = forecaster(lin_valid_report, whole=True)
    lin_valid_interpolated['Var_score'] = np.abs(lin_valid_interpolated['NTG']) #column with linear validity

    whole_results = forecaster(partial_ntg_map, whole=True) #real NTG interpolation
    #avoid merging real ntg ('NTG' in whole results) with fake ntg ('NTG' in lin_valid, really signed lin.validity)
    variance_report = whole_results.merge(lin_valid_interpolated[['X','Y','Var_score']], on=['X','Y'], how='inner')

    # After metric is calculated, restrict it to license area polygon
    #!Note: with universal weighting, restriction should be done before weighting to avoid dissapearing metric problems
    variance_report = forecaster.grid_builder.drillable(variance_report)

    # Build a set of weighted delta-NTG samples (universal)
    if weight_class is not None:
        dists, _ = neighbors_processor.query(variance_report[['X', 'Y']].values, k=1)
        characteristic_dist = dists.max() #flatten() taken out
        relative_offset, relative_scaling = 0.0, 0.8 #change offset, scaling here
        weight_fun = weight_class({'offset': characteristic_dist * relative_offset,
                                   'scaling': characteristic_dist * relative_scaling})
        weights_map = weight_fun(variance_report[['X', 'Y']].values, partial_ntg_map[['X', 'Y']].values)
        variance_report['Var_score'] = variance_report['Var_score'] * weights_map

    #variance_report = forecaster.grid_builder.drillable(variance_report)

    if file_mark is not None:
        if isinstance(file_mark, str) or isinstance(file_mark, int):
            variance_report_file = variance_report_template.format('', file_mark)
        elif isinstance(file_mark, tuple) and len(file_mark) == 2:
            variance_report_file = variance_report_template.format(file_mark[0]+'_', file_mark[1])
        variance_report.to_csv(join(max_std_res_path, variance_report_file), index=False)

    df1 = deepcopy(variance_report)
    max_row = df1[df1['Var_score'] == df1['Var_score'].max()]
    # random.seed(a=123)
    if max_row.shape[0] > 1:
        # it is undesirable to always choose the 1st element because at unshuffled datasets it may bias decision to a
        # particular side of a field
        max_std_row = max_row.iloc[random.randint(0, (max_row.shape[0]-1))]
    else:
        max_std_row = max_row.iloc[0]

    if file_mark is not None:
        print(f'Lin.valid. based decision run mark: {file_mark}\n{(max_std_row["X"],max_std_row["Y"])}')

    if nargout == 1:
        return Point(X=max_std_row['X'].item(), Y=max_std_row['Y'].item())
    else:
        return Point(X=max_std_row['X'].item(), Y=max_std_row['Y'].item()), variance_report


#Uses linearization_validity_semidecision to make decision with individual weighting
def lin_val_based_decision_grid_adapted_individual_weight(
        forecaster, partial_ntg_map: pd.DataFrame, results_folder, file_mark=None,
        weight_class=None, decision_relevant_val='NTG', nargout=1
):
    neighbors_processor = KDTree(partial_ntg_map[['X', 'Y']].values, leaf_size=2)

    if file_mark is not None:
        max_std_res_path = join(results_folder, max_std_res_dir_template.format(f'_{MaxStdModes.basic.value}'))
        #max_std_res_path = results_folder
        if not exists(max_std_res_path):
            makedirs(max_std_res_path)

    # Use semidecision to compute linear validity at wells (as Discrep)
    lin_valid_report = linearization_validity_semidecision(forecaster, partial_ntg_map)
    #forecaster interpolates over NTG, so rename Discrep->NTG so that it interpolates over Discrep
    lin_valid_report = lin_valid_report.rename(columns={'NTG': 'NTG_real', 'Discrep': 'NTG'})
    lin_valid_interpolated = forecaster(lin_valid_report, whole=True)
    lin_valid_interpolated['Var_score'] = np.abs(lin_valid_interpolated['NTG']) #column with linear validity

    whole_results = forecaster(partial_ntg_map, whole=True) #real NTG interpolation
    # Avoid merging real ntg ('NTG' in whole results) with fake ntg ('NTG' in lin_valid, really signed lin.validity)
    variance_report = whole_results.merge(lin_valid_interpolated[['X','Y','Var_score']], on=['X','Y'], how='inner')

    # Build a set of weighted delta-NTG samples (individual: n-th nearest well)
    if weight_class is not None:
        nth_nearest = 3 #4 and higher not possible (for 4 starting wells)
        dists_well, indices_well = neighbors_processor.query(variance_report[['X', 'Y']].values, k=nth_nearest+1)
        relative_offset, relative_scaling = 0.0, 0.8 #change offset, scaling here
        weights_map = np.ones(variance_report.shape[0])
        dist_well_indiv = dists_well[:,nth_nearest] #distance from each well to the n-th nearest well
        for iwell, well_coord in enumerate(partial_ntg_map[['X', 'Y']].values):
            #print(iwell, variance_report[['X', 'Y']].values[iwell], dist_well_indiv[iwell])
            characteristic_dist = dist_well_indiv[iwell]
            weight_fun_well = weight_class({'offset': characteristic_dist * relative_offset,
                                            'scaling': characteristic_dist * relative_scaling})
            weights_map *= weight_fun_well(variance_report[['X', 'Y']].values, well_coord)
        variance_report['Var_score'] = variance_report['Var_score'] * weights_map

    # After metric is calculated, restrict it to license area polygon
    variance_report = forecaster.grid_builder.drillable(variance_report)

    if file_mark is not None:
        if isinstance(file_mark, str) or isinstance(file_mark, int):
            variance_report_file = variance_report_template.format('', file_mark)
        elif isinstance(file_mark, tuple) and len(file_mark) == 2:
            variance_report_file = variance_report_template.format(file_mark[0]+'_', file_mark[1])
        variance_report.to_csv(join(max_std_res_path, variance_report_file), index=False)

    df1 = deepcopy(variance_report)
    max_row = df1[df1['Var_score'] == df1['Var_score'].max()]
    # random.seed(a=123)
    if max_row.shape[0] > 1:
        # it is undesirable to always choose the 1st element because at unshuffled datasets it may bias decision to a
        # particular side of a field
        max_std_row = max_row.iloc[random.randint(0, (max_row.shape[0]-1))]
    else:
        max_std_row = max_row.iloc[0]

    if file_mark is not None:
        print(f'Lin.valid. based decision run mark: {file_mark}\n{(max_std_row["X"],max_std_row["Y"])}')

    if nargout == 1:
        return Point(X=max_std_row['X'].item(), Y=max_std_row['Y'].item())
    else:
        return Point(X=max_std_row['X'].item(), Y=max_std_row['Y'].item()), variance_report


def is_equal_triangle(pd_series):
    primary_triangle = [t.tolist()[:2] for t in pd_series[['Vertex1', 'Vertex2', 'Vertex3']].tolist()]
    secondary_triangle = [t.tolist()[:2] for t in pd_series[['Vertex1_lno', 'Vertex2_lno', 'Vertex3_lno']].tolist()]
    return all([v in primary_triangle for v in secondary_triangle])


def lin_validity_decision_univers_weight(
        forecaster, partial_ntg_map: pd.DataFrame, results_folder, file_mark=None,
        weight_class=None, nargout=1) -> Point:
    neighbors_processor = KDTree(partial_ntg_map[['X', 'Y']].values, leaf_size=2)

    if file_mark is not None:
        max_std_res_path = join(results_folder, max_std_res_dir_template.format(f'_{MaxStdModes.basic.value}'))
        if not exists(max_std_res_path):
            makedirs(max_std_res_path)

    whole_results = forecaster(partial_ntg_map, whole=True)
    whole_results.rename(columns={"NTG": "NTG_all"})

    well_names_train = partial_ntg_map.Well
    cross_valid_ntg_maps_sequence = []
    for max_train_wells in range(len(well_names_train) - 1, len(well_names_train)):
        cross_valid_ntg_maps_sequence += [
            partial_ntg_map[partial_ntg_map['Well'].isin(well_names_comb)].reset_index(drop=True)
            for well_names_comb in combinations(list(well_names_train), max_train_wells)
        ]

    forecaster_ = partial(forecaster, whole=True)
    with Pool(5) as p:
        cross_valid_results = p.map(forecaster_, cross_valid_ntg_maps_sequence)
    # cross_valid_results = forecaster(cross_valid_ntg_maps_sequence[0])

    variance_report = whole_results

    if file_mark is not None:
        wrapped_enumerator = tqdm(enumerate(cross_valid_results), total=len(cross_valid_results))
    else:
        wrapped_enumerator = enumerate(cross_valid_results)

    lno_ntg_columns = ['NTG_' + str(j) for j in range(len(cross_valid_results))]
    for j, df_new_input in wrapped_enumerator:
        df_new_input = df_new_input.rename(
            columns={
                'NTG': lno_ntg_columns[j],
                'Vertex1': 'Vertex1_lno',
                'Vertex2': 'Vertex2_lno',
                'Vertex3': 'Vertex3_lno',
                'Distance': 'Distance_lno'
            }
        )
        variance_report = variance_report.merge(df_new_input, on=['X', 'Y'], how='inner')
        config_intactness = variance_report.apply(is_equal_triangle, axis=1).values
        variance_report.drop(['Vertex1_lno', 'Vertex2_lno', 'Vertex3_lno', 'Distance_lno'], axis=1, inplace=True)
        variance_report.loc[config_intactness, lno_ntg_columns[j]] = np.nan

    distinct_configs = (~variance_report[lno_ntg_columns].isna()).sum(axis=1)
    variance_report['Var_score'] = variance_report[lno_ntg_columns].\
        apply(lambda x: (x - variance_report['NTG'].values) ** 2).apply(np.sum, axis=1) / distinct_configs

    # build a set of weighted delta-NTG samples
    if weight_class is not None:
        dists, _ = neighbors_processor.query(variance_report[['X', 'Y']].values, k=1)
        characteristic_dist = dists.flatten().max() * 2 / 5
        weight_fun = weight_class({'offset': characteristic_dist, 'scaling': characteristic_dist})
        weights_map = weight_fun(variance_report[['X', 'Y']].values, partial_ntg_map[['X', 'Y']].values)
        variance_report['Var_score'] = variance_report['Var_score'] * weights_map

    # if weight_class is not None:
    #     dists, _ = neighbors_processor.query(partial_ntg_map[['X', 'Y']].values, k=2)
    #     dists = dists[:, 1].flatten()
    #     weights_map = np.ones(variance_report.shape[0])
    #     for p, d in zip(partial_ntg_map[['X', 'Y']].values, dists):
    #         characteristic_dist = d * 2 / 5
    #         weight_fun = weight_class({'offset': characteristic_dist, 'scaling': characteristic_dist})
    #         weights_map = weights_map * weight_fun(variance_report[['X', 'Y']].values, p)
    #     variance_report['Var_score'] = variance_report['Var_score'] * weights_map

    if file_mark is not None:
        max_std_res_path = join(results_folder, max_std_res_dir_template.format(f'_{MaxStdModes.unique_triangles.value}'))
        if not exists(max_std_res_path):
            makedirs(max_std_res_path)
        variance_report_file = variance_report_template.format(file_mark)
        variance_report.to_csv(join(max_std_res_path, variance_report_file), index=False)

    df1 = deepcopy(variance_report)
    max_row = df1[df1['Var_score'] == df1['Var_score'].max()]
    # random.seed(a=123)
    if max_row.shape[0] > 1:
        # it is undesirable to always choose the 1st element because at unshuffled datasets it may bias decision to a
        # particular side of a field
        max_std_row = max_row.iloc[random.randint(0, (max_row.shape[0]-1))]
    else:
        max_std_row = max_row.iloc[0]

    if file_mark is not None:
        print(f'Max std based decision run mark: {file_mark}\n{(max_std_row["X"],max_std_row["Y"])}')

    if nargout == 1:
        return Point(X=max_std_row['X'].item(), Y=max_std_row['Y'].item())
    else:
        return Point(X=max_std_row['X'].item(), Y=max_std_row['Y'].item()), variance_report


def max_std_based_decision_samspec_weight(
        forecaster, partial_ntg_map: pd.DataFrame, results_folder, file_mark=None,
        weight_fun=None, nargout=1) -> Point:
    whole_results = forecaster(partial_ntg_map, whole=True)
    whole_results.rename(columns={"NTG": "NTG_all"})

    well_names_train = partial_ntg_map.Well
    cross_valid_ntg_maps_sequence = []
    for max_train_wells in range(len(well_names_train) - 1, len(well_names_train)):
        cross_valid_ntg_maps_sequence += [
            partial_ntg_map[partial_ntg_map['Well'].isin(well_names_comb)].reset_index(drop=True)
            for well_names_comb in combinations(list(well_names_train), max_train_wells)
        ]

    with Pool(5) as p:
        cross_valid_results = p.map(forecaster, cross_valid_ntg_maps_sequence)
    # cross_valid_results = forecaster(cross_valid_ntg_maps_sequence[0])

    variance_report = pd.DataFrame()

    lno_ntg_columns = ['NTG_' + str(j) for j in range(len(cross_valid_results))]
    for j, df_new_input in enumerate(cross_valid_results):
        df_new_input.columns = ['X', 'Y', lno_ntg_columns[j]]
        if j == 0:
            variance_report = df_new_input
        else:
            variance_report = variance_report.merge(df_new_input, on=['X', 'Y'], how='inner')

    variance_report = variance_report.merge(whole_results, on=['X', 'Y'], how='inner')

    # build a set of weighted delta-NTG samples
    if weight_fun is not None:
        instability_samples = np.zeros([variance_report.shape[0], len(lno_ntg_columns)])
        all_wells = set(well_names_train)
        hold_out_well_names_combs = [tuple(all_wells - set(p['Well'].values)) for p in cross_valid_ntg_maps_sequence]
        hold_out_coord_combs = [partial_ntg_map.loc[partial_ntg_map['Well'].isin(c), ['X', 'Y']].values
                                for c in hold_out_well_names_combs]

        for i, c in enumerate(lno_ntg_columns):
            instability_samples[:, i] = (variance_report[c].values - variance_report['NTG'].values) * \
                                        weight_fun(variance_report[['X', 'Y']].values, hold_out_coord_combs[i])

        variance_report['Var_score'] = np.var(instability_samples, axis=1)
    else:
        variance_report['Var_score'] = variance_report[lno_ntg_columns].var(axis=1)

    if file_mark is not None:
        max_std_res_path = join(results_folder, max_std_res_dir_template.format(f'_{MaxStdModes.basic.value}'))
        if not exists(max_std_res_path):
            makedirs(max_std_res_path)
        variance_report_file = variance_report_template.format(file_mark)
        variance_report.to_csv(join(max_std_res_path, variance_report_file), index=False)

    df1 = deepcopy(variance_report)
    max_row = df1[df1['Var_score'] == df1['Var_score'].max()]
    # random.seed(a=123)
    if max_row.shape[0] > 1:
        # it is undesirable to always choose the 1st element because at unshuffled datasets it may bias decision to a
        # particular side of a field
        max_std_row = max_row.iloc[random.randint(0, (max_row.shape[0]-1))]
    else:
        max_std_row = max_row.iloc[0]

    if file_mark is not None:
        print(f'Max std based decision run mark: {file_mark}\n{(max_std_row["X"],max_std_row["Y"])}')

    if nargout == 1:
        return Point(X=max_std_row['X'].item(), Y=max_std_row['Y'].item())
    else:
        return Point(X=max_std_row['X'].item(), Y=max_std_row['Y'].item()), variance_report


def random_decision(
        forecaster,
        partial_ntg_map: pd.DataFrame,
        results_folder,
        file_mark,
        weight_class=None,
        reporting_mode=RandomDecisionReportingModes.BASIC
) -> Point:
    rnd_res_path = join(results_folder, rnd_res_dir_template.format('_' + reporting_mode.value))
    if not exists(rnd_res_path):
        makedirs(rnd_res_path)

    well_names_train = partial_ntg_map.Well

    if reporting_mode is RandomDecisionReportingModes.LOO:
        cross_valid_ntg_maps_sequence = []
        for max_train_wells in range(len(well_names_train) - 1, len(well_names_train)):
            cross_valid_ntg_maps_sequence += [
                partial_ntg_map[partial_ntg_map['Well'].isin(well_names_comb)].reset_index(drop=True)
                for well_names_comb in combinations(list(well_names_train), max_train_wells)
            ]
        with Pool(5) as p:
            cross_valid_results = p.map(forecaster, cross_valid_ntg_maps_sequence)
        # cross_valid_results = forecaster(cross_valid_ntg_maps_sequence[0])

        variance_report = pd.DataFrame()

        for j, df_new_input in enumerate(cross_valid_results):
            df_new_input.columns = ['X', 'Y', 'NTG_' + str(j)]
            if j == 0:
                variance_report = df_new_input
            else:
                variance_report = variance_report.merge(df_new_input, on=['X', 'Y'], how='outer')
        variance_report.dropna(inplace=True)
        variance_report_file = variance_report_template.format(file_mark)
        variance_report.to_csv(join(rnd_res_path, variance_report_file), index=False)

        decision_variants = variance_report[['X', 'Y']]

    elif reporting_mode is RandomDecisionReportingModes.BASIC:
        xy_arr = forecaster.grid_builder(partial_ntg_map)
        decision_variants = pd.DataFrame(xy_arr, columns=['X', 'Y'])

    else:
        raise ValueError('Incorrect value for reporting_mode parameter')

    if weight_class:
        neighbors_processor = KDTree(partial_ntg_map[['X', 'Y']].values, leaf_size=2)
        dists, _ = neighbors_processor.query(decision_variants.values, k=1)
        characteristic_dist = dists.flatten().max() * 2 / 5
        weight_fun = weight_class({'offset': characteristic_dist, 'scaling': characteristic_dist})
        weights_map = weight_fun(decision_variants.values, partial_ntg_map[['X', 'Y']].values)

        max_row = decision_variants.loc[weights_map == weights_map.max()]
        # random.seed(a=123)
        if max_row.shape[0] > 1:
            # it is undesirable to always choose the 1st element because at unshuffled datasets it may bias
            # decision to a particular side of a field
            rand_row = max_row.loc[max_row.index[random.randint(0, (max_row.shape[0] - 1))]]
        else:
            rand_row = max_row.loc[max_row.index[0]]
    else:
        rand_row = decision_variants.loc[random.randrange(0, decision_variants.shape[0])]

    return Point(X=rand_row.loc['X'], Y=rand_row.loc['Y'])


def max_triangle_based_decision(forecaster: callable, partial_ntg_map: pd.DataFrame, results_folder,
                                file_mark, nargout=1) -> Point:
    max_std_res_path = join(results_folder, max_std_res_dir_template.format(f'_{MaxStdModes.tri_centers.value}'))
    if not exists(max_std_res_path):
        makedirs(max_std_res_path)

    whole_results = forecaster(partial_ntg_map, whole=True)
    whole_results.rename(columns={"NTG": "NTG_all"})
    well_names_train = partial_ntg_map.Well
    cross_valid_ntg_maps_sequence = []
    for max_train_wells in range(len(well_names_train) - 1, len(well_names_train)):
        cross_valid_ntg_maps_sequence += [
            partial_ntg_map[partial_ntg_map['Well'].isin(well_names_comb)]
            for well_names_comb in combinations(list(well_names_train), max_train_wells)
        ]

    with Pool(5) as p:
        cross_valid_results = p.map(forecaster, cross_valid_ntg_maps_sequence)
    # cross_valid_results = forecaster(cross_valid_ntg_maps_sequence[0])

    variance_report = pd.DataFrame()

    for j, df_new_input in enumerate(cross_valid_results):
        df_new_input.columns = ['X', 'Y', 'NTG_' + str(j)]
        if j == 0:
            variance_report = df_new_input
        else:
            variance_report = variance_report.merge(df_new_input, on=['X', 'Y'], how='outer')
    variance_report.dropna(inplace=True)

    variance_report['Var'] = variance_report.drop(['X', 'Y'], axis=1).var(axis=1)
    variance_report = variance_report.merge(whole_results, on=['X', 'Y'], how='outer')
    variance_report_file = variance_report_template.format(file_mark)
    variance_report.to_csv(join(max_std_res_path, variance_report_file), index=False)

    df1 = deepcopy(variance_report)
    max_row = df1[df1['Var'] == df1['Var'].max()]
    # random.seed(a=123)
    if max_row.shape[0] > 1:
        max_std_row = max_row.iloc[random.randint(0, (max_row.shape[0]-1))]
    else:
        max_std_row = max_row.iloc[0]
    if max_std_row['Vertex1'] is not None:
        point_to_drill = [int(round(el, 0)) for el in
                          sum([max_std_row['Vertex1'], max_std_row['Vertex2'], max_std_row['Vertex3']]) / 3]
    elif max_std_row['Distance'] is not None:
        max_table = variance_report[variance_report['Var'] == variance_report['Var'].max()]
        point_to_drill = max_table.loc[
            max_table['Distance'].idxmax(), ['X', 'Y']
        ].values
    else:
        raise Exception('Something is wrong with output DataFrame!')

    print(f'Center of a triangle based decision run mark: {file_mark}\n{(point_to_drill[0], point_to_drill[1])}')

    if nargout == 1:
        return Point(X=point_to_drill[0], Y=point_to_drill[1])
    else:
        return Point(X=point_to_drill[0], Y=point_to_drill[1]), variance_report


def drill_apart_based_decision(forecaster: callable, partial_ntg_map: pd.DataFrame, results_folder,
                               file_mark, nargout=1) -> Point:
    max_std_res_path = join(results_folder, max_std_res_dir_template.format(f'_{MaxStdModes.drill_apart.value}'))
    if not exists(max_std_res_path):
        makedirs(max_std_res_path)

    whole_results = forecaster(partial_ntg_map, whole=True)
    whole_results.rename(columns={"NTG": "NTG_all"})
    well_names_train = partial_ntg_map.Well
    cross_valid_ntg_maps_sequence = []
    for max_train_wells in range(len(well_names_train) - 1, len(well_names_train)):
        cross_valid_ntg_maps_sequence += [
            partial_ntg_map[partial_ntg_map['Well'].isin(well_names_comb)]
            for well_names_comb in combinations(list(well_names_train), max_train_wells)
        ]

    with Pool(5) as p:
        cross_valid_results = p.map(forecaster, cross_valid_ntg_maps_sequence)

    variance_report = pd.DataFrame()

    for j, df_new_input in enumerate(cross_valid_results):
        df_new_input.columns = ['X', 'Y', 'NTG_' + str(j)]
        if j == 0:
            variance_report = df_new_input
        else:
            variance_report = variance_report.merge(df_new_input, on=['X', 'Y'], how='outer')
    variance_report.dropna(inplace=True)

    variance_report['Var'] = variance_report.drop(['X', 'Y'], axis=1).var(axis=1)
    variance_report = variance_report.merge(whole_results, on=['X', 'Y'], how='outer')
    variance_report_file = variance_report_template.format(file_mark)
    variance_report.to_csv(join(max_std_res_path, variance_report_file), index=False)

    df1 = deepcopy(variance_report)
    flag = 0
    forbidden_list = []
    for pp in zip(partial_ntg_map['X'], partial_ntg_map['Y']):
        listik = list(
            itertools.product([pp[0], pp[0] + 1, pp[0] - 1], [pp[1], pp[1] + 1, pp[1] - 1]))
        forbidden_list.extend(listik)
    forbidden_list = list(set(forbidden_list))
    while flag == 0:
        max_row = df1[df1['Var'] == df1['Var'].max()]
        df1 = df1[df1['Var'] != df1['Var'].max()]
        for jjj, row in max_row.iterrows():
            if (row['X'], row['Y']) in forbidden_list:
                continue
            else:
                flag = 1
                point_to_drill = row[['X', 'Y']]
                break

    print(f'Drill apart based decision run mark: {file_mark}\n{point_to_drill}')

    if nargout == 1:
        return Point(X=point_to_drill["X"], Y=point_to_drill["Y"])
    else:
        return Point(X=point_to_drill["X"], Y=point_to_drill["Y"]), variance_report


def forecaster_biased_decision(forecaster: callable, partial_ntg_map: pd.DataFrame, results_folder,
                               file_mark=None, weight_fun=None, nargout=1) -> Point:
    whole_results = forecaster(partial_ntg_map, whole=True)

    well_names_train = partial_ntg_map.Well
    cross_valid_ntg_maps_sequence = []
    for max_train_wells in range(len(well_names_train) - 1, len(well_names_train)):
        cross_valid_ntg_maps_sequence += [
            partial_ntg_map[partial_ntg_map['Well'].isin(well_names_comb)].reset_index(drop=True)
            for well_names_comb in combinations(list(well_names_train), max_train_wells)
        ]

    with Pool(5) as p:
        cross_valid_results = p.map(forecaster, cross_valid_ntg_maps_sequence)
    # cross_valid_results = forecaster(cross_valid_ntg_maps_sequence[0])

    variance_report = pd.DataFrame()

    lno_ntg_columns = ['NTG_' + str(j) for j in range(len(cross_valid_results))]
    for j, df_new_input in enumerate(cross_valid_results):
        df_new_input.columns = ['X', 'Y', lno_ntg_columns[j]]
        if j == 0:
            variance_report = df_new_input
        else:
            variance_report = variance_report.merge(df_new_input, on=['X', 'Y'], how='inner')

    variance_report = variance_report.merge(whole_results, on=['X', 'Y'], how='inner')
    variance_report['Var_score'] = variance_report[lno_ntg_columns].var(axis=1)

    # build a set of weighted delta-NTG samples
    variance_report['Var_score'] = variance_report['Var_score'] * whole_results['Std']
    if weight_fun is not None:
        weights_map = weight_fun(variance_report[['X', 'Y']].values, partial_ntg_map[['X', 'Y']].values)
        variance_report['Var_score'] = variance_report['Var_score'] * weights_map

    if file_mark is not None:
        max_std_res_path = join(results_folder, max_std_res_dir_template.format('_kriging'))
        if not exists(max_std_res_path):
            makedirs(max_std_res_path)
        variance_report_file = variance_report_template.format(file_mark)
        variance_report.to_csv(join(max_std_res_path, variance_report_file), index=False)

    df1 = deepcopy(variance_report)
    max_row = df1[df1['Var_score'] == df1['Var_score'].max()]
    # random.seed(a=123)
    if max_row.shape[0] > 1:
        # it is undesirable to always choose the 1st element because at unshuffled datasets it may bias decision to a
        # particular side of a field
        max_std_row = max_row.iloc[random.randint(0, (max_row.shape[0]-1))]
    else:
        max_std_row = max_row.iloc[0]

    if file_mark is not None:
        print(f'Max std based decision run mark: {file_mark}\n{(max_std_row["X"],max_std_row["Y"])}')

    if nargout == 1:
        return Point(X=max_std_row['X'].item(), Y=max_std_row['Y'].item())
    else:
        return Point(X=max_std_row['X'].item(), Y=max_std_row['Y'].item()), variance_report


def kriging_expect_rmse_decision(forecaster: callable, partial_ntg_map: pd.DataFrame, results_folder,
                                 file_mark=None, weight_fun=None, nargout=1) -> Point:
    results = forecaster(partial_ntg_map, whole=False)
    kernel_squared_amplitude = forecaster.gp.kernel_.k1.k1.get_params()['constant_value']
    kernel_length_scale = forecaster.gp.kernel_.k1.k2.get_params()['length_scale']
    noise_level = forecaster.gp.kernel_.k2.get_params()['noise_level']

    expected_rmse = np.zeros(results.shape[0])
    if file_mark is not None:
        wrapped_iterator = tqdm(range(results.shape[0]))
        print(f'{file_mark}: Well placement variants assessment in progress ...\n')
    else:
        wrapped_iterator = range(results.shape[0])

    for i in wrapped_iterator:
        step_ahead_train_grid = np.concatenate(
            (partial_ntg_map[['X', 'Y']].values,
             results.loc[i, ['X', 'Y']].values.reshape(1, 2)),
            axis=0
        )
        step_ahead_appraisal_grid = np.delete(results[['X', 'Y']].values, i, axis=0)

        train_cov_matr = forecaster.gp.kernel_(step_ahead_train_grid)  # T x T
        cross_sets_cov = forecaster.gp.kernel_(step_ahead_appraisal_grid, step_ahead_train_grid)  # D x T
        m = cross_sets_cov @ np.linalg.inv(train_cov_matr)  # D x T
        conditioning_correction = (
            m @ cross_sets_cov.T
        ).trace() / step_ahead_appraisal_grid.shape[0]
        expected_rmse[i] = np.sqrt(
            kernel_squared_amplitude + noise_level - conditioning_correction +
            ((np.sum((m.T @ m).ravel()) + 2 * np.sum(m.ravel()))/step_ahead_appraisal_grid.shape[0] + 1) *
            np.sum(train_cov_matr.ravel()) / train_cov_matr.shape[0] ** 2
        )

    results['Var_score'] = expected_rmse
    if weight_fun is not None:
        weights_map = weight_fun(results[['X', 'Y']].values, partial_ntg_map[['X', 'Y']].values)
        results['Var_score'] = results['Var_score'] * weights_map

    if file_mark is not None:
        if isinstance(file_mark, str) or isinstance(file_mark, int):
            variance_report_file = variance_report_template.format('', file_mark)
        elif isinstance(file_mark, tuple) and len(file_mark) == 2:
            variance_report_file = variance_report_template.format(file_mark[0]+'_', file_mark[1])
        results.to_csv(join(results_folder, variance_report_file), index=False)

    df1 = deepcopy(results)
    best_expect_rmse_forecast = df1['Var_score'].min()
    selected_rows = df1[df1['Var_score'] == best_expect_rmse_forecast]
    # random.seed(a=123)
    if selected_rows.shape[0] > 1:
        # it is undesirable to always choose the 1st element because at unshuffled datasets it may bias decision to a
        # particular side of a field
        selected_row = selected_rows.iloc[random.randint(0, (selected_rows.shape[0] - 1))]
    else:
        selected_row = selected_rows.iloc[0]

    if file_mark is not None:
        print(f'Max std based decision run mark: {file_mark}\n'
              f'Inferred parameters of GP kernel:\n'
              f'\tLength scale: {kernel_length_scale},\n'
              f'\tSquared amplitude: {kernel_squared_amplitude},\n'
              f'\tNoise variance: {noise_level}\n'
              f'Expect_RMSE forecast: {best_expect_rmse_forecast}\n')

    if nargout == 1:
        return Point(X=selected_row['X'].item(), Y=selected_row['Y'].item())
    else:
        return Point(X=selected_row['X'].item(), Y=selected_row['Y'].item()), results


def gamer_decision(
        forecaster,
        partial_ntg_map: pd.DataFrame,
        results_folder,
        file_mark
) -> Point:
    # gamer_decision_res_path = join(results_folder, rnd_res_dir_template.format('_' + reporting_mode.value))
    # well_names_train = partial_ntg_map.Well
    whole_results = forecaster(partial_ntg_map, whole=True)
    whole_results.rename(columns={"NTG": "NTG_all"})
    # xy_arr = forecaster.grid_builder(partial_ntg_map)
    whole_results.to_csv(join(results_folder, (f'game_'+variance_report_template.format(file_mark))), index=False)

    # decision_variants = variance_report[['X', 'Y']]
    #
    #
    # rand_row = decision_variants.loc[random.randrange(0, decision_variants.shape[0])]

def universal_kriging_expect_rmse_decision(
        forecaster: callable, partial_ntg_map: pd.DataFrame, results_folder,
        file_mark=None, weight_fun=None
) -> Point:
    results = forecaster(partial_ntg_map, whole=False)
    noise_level = forecaster.kriger.variogram_model_parameters[2]  # nugget
    kernel_squared_amplitude = forecaster.kriger.variogram_model_parameters[0]  # 'psill
    kernel_length_scale = forecaster.kriger.variogram_model_parameters[1]

    kernel = kernel_squared_amplitude * RBF(length_scale=kernel_length_scale) + WhiteKernel(noise_level=noise_level)

    expected_rmse = np.zeros(results.shape[0])
    if file_mark is not None:
        wrapped_iterator = tqdm(range(results.shape[0]))
        print(f'{file_mark}: Well placement variants assessment in progress ...\n')
    else:
        wrapped_iterator = range(results.shape[0])

    for i in wrapped_iterator:
        step_ahead_train_grid = np.concatenate(
            (partial_ntg_map[['X', 'Y']].values,
             results.loc[i, ['X', 'Y']].values.reshape(1, 2)),
            axis=0
        )
        step_ahead_appraisal_grid = np.delete(results[['X', 'Y']].values, i, axis=0)

        uncond_c = kernel_squared_amplitude + noise_level

        train_cov_matr = kernel(step_ahead_train_grid)
        # T x T
        cross_sets_cov = kernel(step_ahead_appraisal_grid, step_ahead_train_grid)
        # D x T
        train_drift_terms = np.hstack(
            (np.ones((step_ahead_train_grid.shape[0], 1)),
             step_ahead_train_grid)
        )
        # T x 3
        appraisal_drift_terms = np.hstack(
            (np.ones((step_ahead_appraisal_grid.shape[0], 1)),
             step_ahead_appraisal_grid)
        )
        # D x 3

        inv_train_cov_matr = np.linalg.inv(train_cov_matr)
        conditioning_correction = (
            cross_sets_cov @ inv_train_cov_matr @ cross_sets_cov.T
        ).trace() / step_ahead_appraisal_grid.shape[0]

        side_m = appraisal_drift_terms.T - train_drift_terms.T @ inv_train_cov_matr @ cross_sets_cov.T
        # 3 x D

        middle_m = np.linalg.inv(train_drift_terms.T @ inv_train_cov_matr @ train_drift_terms)
        # 3 x 3

        unknown_mean_correction = (
            side_m.T @ middle_m @ side_m
        ).trace() / step_ahead_appraisal_grid.shape[0]

        squared_expect_rmse = uncond_c - conditioning_correction + unknown_mean_correction
        if squared_expect_rmse > 0:
            expected_rmse[i] = np.sqrt(uncond_c - conditioning_correction + unknown_mean_correction)
        else:
            expected_rmse[i] = 0

    results['Var_score'] = expected_rmse
    if weight_fun is not None:
        weights_map = weight_fun(results[['X', 'Y']].values, partial_ntg_map[['X', 'Y']].values)
        results['Var_score'] = results['Var_score'] * weights_map

    if file_mark is not None:
        res_path = join(results_folder, expect_rmse_res_dir_template.format('_kriging'))
        if not exists(res_path):
            makedirs(res_path)
        results_file = variance_report_template.format(file_mark)
        results.to_csv(join(res_path, results_file), index=False)

    df1 = deepcopy(results)
    best_expect_rmse_forecast = df1['Var_score'].min()
    selected_rows = df1[df1['Var_score'] == best_expect_rmse_forecast]
    # random.seed(a=123)
    if selected_rows.shape[0] > 1:
        # it is undesirable to always choose the 1st element because at unshuffled datasets it may bias decision to a
        # particular side of a field
        selected_row = selected_rows.iloc[random.randint(0, (selected_rows.shape[0] - 1))]
    else:
        selected_row = selected_rows.iloc[0]

    if file_mark is not None:
        print(f'Max std based decision run mark: {file_mark}\n'
              f'Inferred parameters of GP kernel:\n'
              f'\tLength scale: {kernel_length_scale},\n'
              f'\tSquared amplitude: {kernel_squared_amplitude},\n'
              f'\tNoise variance: {noise_level}\n'
              f'Expect_RMSE forecast: {best_expect_rmse_forecast}\n')

    return Point(X=selected_row['X'], Y=selected_row['Y'])


def universal_pykriging_decision(
        forecaster: callable, partial_ntg_map: pd.DataFrame, results_folder,
        file_mark=None, weight_fun=None
) -> Point:
    results = forecaster(partial_ntg_map)

    if file_mark is not None:
        wrapped_iterator = tqdm(range(results.shape[0]))
        print(f'{file_mark}: Well placement variants assessment in progress ...\n')
    else:
        wrapped_iterator = range(results.shape[0])

    for i in wrapped_iterator:
        step_ahead_train_grid = np.concatenate(
            (partial_ntg_map[['X', 'Y']].values,
             results.loc[i, ['X', 'Y']].values.reshape(1, 2)),
            axis=0
        )

    results['Var_score'] = 1
    if weight_fun is not None:
        weights_map = weight_fun(results[['X', 'Y']].values, partial_ntg_map[['X', 'Y']].values)
        results['Var_score'] = results['Var_score'] * weights_map

    if file_mark is not None:
        res_path = join(results_folder, expect_rmse_res_dir_template.format('_pyKriging'))
        if not exists(res_path):
            makedirs(res_path)
        results_file = variance_report_template.format(file_mark)
        results.to_csv(join(res_path, results_file), index=False)

    newpoint = forecaster.kriger.infill(1)

    if file_mark is not None:
        print(f'Max std based decision run mark: {file_mark}\n'
              f'{newpoint}\n')

    return Point(X=round(newpoint[0, 0]), Y=round(newpoint[0, 1]))
