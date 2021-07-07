import os
from flask import Flask, Response, json, request
import logging
from logging.handlers import TimedRotatingFileHandler
from functools import partial

import re
import pandas as pd

import json
from src.forecasters import BaselineForecaster as BilinearForecaster
from src.forecasters import KrigingForecaster
from src.grid_builder import build_grid
from src.decision_makers import max_std_based_decision_grid_adapted_univers_weight as determine_new_well_bilinear
from src.decision_makers import kriging_expect_rmse_decision as determine_new_well_kriging
from src.weight_functions import Cubic as WeightClass
from src.constants import server_files_storage, variance_report_template

# logging settings
logger_service = logging.getLogger('driller_walk')
logger_service.setLevel(logging.INFO)
log_format = logging.Formatter('%(asctime)s [%(name)-24.24s] [%(levelname)-5.5s]  %(message)s')
file_handler = TimedRotatingFileHandler(f"log/{'driller_walk'}.log", when='h', interval=1, backupCount=50,
                                        encoding='UTF-8')
file_handler.setFormatter(log_format)
logger_service.addHandler(file_handler)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_format)
logger_service.addHandler(console_handler)

logger_service.info(f"Starting service {'driller walk'}")

app = Flask(__name__)


def success_handle(output, status=200, mimetype='application/json'):
    logger_service.info(f"Running success_handle for new well")
    return Response(json.dumps(output) + '\n', status=status, mimetype=mimetype)


def error_handle(error_message, status=500, mimetype='application/json'):
    logger_service.error(f'error_handle: `{error_message}`')
    return Response(json.dumps({'error': {'message': error_message}}, ensure_ascii=False) + '\n',
                    status=status, mimetype=mimetype)


@app.route('/api/v1/generate_next_point/<mode>/<username>', methods=['POST'])  # ,
def generate_next_point_kriging(mode, username):
    logger = logger_service.getChild("generate_next_point")
    logger.info(f'Request from {request.environ["REMOTE_ADDR"]}, mode: {mode}, username: {username}')

    if mode not in ['ordinary_kriging', 'bilinear']:
        return error_handle(
            "Unknown mode of operation was set in URL. Correct modes are: 'bilinear' and 'ordinary_kriging'"
        )

    message = request.get_json()
    message['partial_ntg_map']['X'] = message['partial_ntg_map'].pop('i')
    message['partial_ntg_map']['Y'] = message['partial_ntg_map'].pop('j')
    logger.debug(message)

    number_pattern = re.compile(
        f"(?<=^{variance_report_template.format(username + '_' + mode + '_', '')[:-4]})[0-9]+(?=\\.csv)")
    current_iteration = 0
    for file in os.listdir(server_files_storage):
        match = number_pattern.search(file)
        if match:
            considered_iteration = (int(match.group()))
            current_iteration = considered_iteration if considered_iteration > current_iteration \
                else current_iteration
    current_iteration += 1

    partial_ntg_map = pd.DataFrame(message['partial_ntg_map'])
    build_grid_runtime = partial(
        build_grid, x_max_grid=message["width"], y_max_grid=message["height"], x_min_grid=1, y_min_grid=1
    )

    logger.info(f'Determined current iteration number: {current_iteration}')

    if mode == 'ordinary_kriging':
        try:
            forecaster = KrigingForecaster(grid_builder=build_grid_runtime)
        except Exception as e:
            return error_handle("Kriging forecaster instantiation failed\n{}".format(str(e)))
        try:
            new_well_coords, variance_report = determine_new_well_kriging(
                forecaster, partial_ntg_map, results_folder=server_files_storage,
                file_mark=(username + '_' + mode, current_iteration),
                nargout=2
            )
        except Exception as e:
            return error_handle("Expect-RMSE decision maker failed\n{}".format(str(e)))
    else:
        try:
            forecaster = BilinearForecaster(grid_builder=build_grid_runtime)
        except Exception as e:
            return error_handle("Weighted bilinear forecaster instantiation failed\n{}".format(str(e)))
        try:
            new_well_coords, variance_report = determine_new_well_bilinear(
                forecaster, partial_ntg_map, results_folder=server_files_storage,
                file_mark=(username + '_' + mode, current_iteration),
                weight_class=WeightClass, nargout=2
            )
        except Exception as e:
            return error_handle("LOO-based decision maker failed\n{}".format(str(e)))

    variance_report = variance_report[['X', 'Y', 'NTG', 'Var_score']]

    return success_handle(
        {'new_well': {'i': new_well_coords.X, 'j': new_well_coords.Y},
         'variance_report': variance_report.to_dict('list')}
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0')
