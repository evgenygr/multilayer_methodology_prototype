import logging
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('GeoBot')
    logger.setLevel(logging.INFO)
    logger.info('importing modules')

from os.path import join
import json
import argparse
import random
import numpy as np
from functools import partial
import src.forecasters
import src.recommendation_metrics
import src.weight_functions
from src.grid_builder import build_grid
from src.constants import driller_walk_arrows_file, variance_report_template,\
    recommend_metrics_name, q2f, q2m
from src.points_selector import PointsSelector
from src.data_dump import DataDump, ConciseDataDump
from src.weight_functions_ensembles import EnsembleScalingMode
from src.data_provider import DataProvider
from src.composite_metrics import CompositeMetrics
from src.forecasters_ensemble import ForecastersEnsemble

# -----------------Deployment of the methodology----------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, action='store',
                        help='specify path to a file storing multilayer field data')

    parser.add_argument('-r', "--res_path", type=str, action='store',
                        help='specify path to a directory where results will be stored')

    parser.add_argument('-s', '--settings', type=str, action='store',
                        help='specify path to a file with experiment settings')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='specify whether one needs detailed variance reports saved')

    parser.add_argument('-c', '--concise', action='store_true',
                        help='specify whether one needs RMSE evolution profile saved')

    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    res = parser.parse_args()

    # ------------ debug ----------------
    # from os import chdir
    # from dataclasses import dataclass
    # @dataclass
    # class Res:
    #     filename: str = '../data/theOilCase2020/OilCase2020.csv'
    #     res_path: str = 'results_warehouse/OilCase2020_field_run_results'
    #     settings: str = 'settings_multilayer_field_data_experiment.json'
    #     verbose: bool = True
    #     concise: bool = True
    # res = Res()
    # chdir(r'C:\Users\user-pc\PycharmProjects\data-analysis-for-gpn-tasks\dvc_experiments')

    with open(res.settings) as f:
        settings = json.load(f)

    layers2forecasters = settings['layers2forecasters']
    for layer_rep in layers2forecasters:
        for q in layer_rep[q2f].keys():
            layer_rep[q2f][q] = getattr(src.forecasters, layer_rep[q2f][q])

    layers2metrics = settings['layers2metrics']
    for layer_rep in layers2metrics:
        for q in layer_rep[q2m].keys():
            layer_rep[q2m][q]['metrics'] = getattr(src.recommendation_metrics, layer_rep[q2m][q]['metrics'])
            weight_class = layer_rep[q2m][q].get('weight_class')
            if weight_class is not None:
                layer_rep[q2m][q]['weight_class'] = getattr(src.weight_functions, weight_class)
                layer_rep[q2m][q]['wf_ensemble_mode'] = getattr(
                    EnsembleScalingMode,
                    layer_rep[q2m][q].get('wf_ensemble_mode', EnsembleScalingMode.universal.value)
                )

    random.seed(settings['random_seed'])
    data_provider = DataProvider(path=res.filename)
    train = np.array(settings['train'])
    partial_multilayer_field_data = data_provider.get_data(train)
    logger.info(f'initial set of {partial_multilayer_field_data.shape[0]} \
wells with known NTG:\n{partial_multilayer_field_data}')

    build_grid = partial(build_grid, **settings['grid_builder_settings'])
    f_ensemble = ForecastersEnsemble(
        layers_to_forecasters=layers2forecasters,
        train_data=partial_multilayer_field_data,
        grid_builder=build_grid
    )
    target_value_of_composite_metrics = settings.get(
        'target_value_of_composite_metrics',
        src.composite_metrics.CompositeMetricsTargets.ntg.value
    )
    metrics = CompositeMetrics(
        forecasters_ensemble=f_ensemble,
        layers_to_metrics=layers2metrics,
        target_value=getattr(
            src.composite_metrics.CompositeMetricsTargets,
            target_value_of_composite_metrics
        )
    )
    select_points = PointsSelector(('', target_value_of_composite_metrics, recommend_metrics_name), logger=logger)
    dump_data = DataDump(
        overall_results_folder=res.res_path,
        res_dir_template=None,
        variance_report_template=variance_report_template,
        active=res.verbose
    )
    train_len = len(train)
    reference_target_quantity_map = data_provider.data_source
    metrics.reduce_to_target(reference_target_quantity_map)
    concise_data_dump = ConciseDataDump(
        res_file_path=join(res.res_path, 'RMSE_decline_profile.csv'),
        pnum_seq=list(range(train_len, train_len + settings['number_of_repetitions'])),
        profile_names=['RMSE'],
        reference_df=reference_target_quantity_map,
        current_name='RMSE',
        target_quantity=target_value_of_composite_metrics,
        active=res.concise
    )
    # -----------------Run the methodology----------------------

    # driller_walk_arrows = pd.DataFrame(columns=['X', 'Y'])
    with dump_data as verbose_data_logger, concise_data_dump as concise_data_logger:
        for i in range(0, settings['number_of_repetitions']):
            metrics.train_data = partial_multilayer_field_data
            variance_report = metrics()
            new_well_coords = select_points(variance_report, log_mark=i)
            verbose_data_logger(variance_report, file_mark=i)
            concise_data_logger(variance_report)
            train = np.vstack([train, new_well_coords])
            partial_multilayer_field_data = data_provider.get_data(train)

    # out_folder = join(overall_res_dir, settings_specific_res_dir_template.format(f'_{mode.value}'))
    # driller_walk_arrows.to_csv(join(out_folder, driller_walk_arrows_file), index=False)
