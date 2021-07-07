from os.path import join
from enum import Enum

recommend_metrics_name = 'Var_score'
project_root = 'data-analysis-for-gpn-tasks'
res_folder = 'res_driller_walk'
logs_folder = join('data', 'input', 'LOGs', 'data', 'train_test')
variance_report_filename_base = 'variance_report'
variance_report_template = '{1}' + variance_report_filename_base + '_{0}.csv'
driller_walk_arrows_file = 'driller_walk_arrows.csv'
max_std_res_dir_template = 'max_std_reports{}'
expect_rmse_res_dir_template = 'expect_rmse_reports{}'
rnd_res_dir_template = 'rand_reports{}'
rnd_rmse_report_template = 'rmse_report{}.csv'
experiments_res_dir_template = 'experiment{}'
NTG_MAP_FILE = 'new_train.csv'

# constants for game_logs
res_folder_game = 'res_driller_walk_game'
logs_folder_game = join('data', 'theGeoCase', 'NTC_teams')

# constants for server
server_files_storage = '/workdir/driller_walk_reports'

# variance_report_template = 'variance_report_{}.csv'
# driller_walk_arrows_file = 'driller_walk_arrows.csv'
# max_std_res_dir_template = 'max_std_reports{}'
# expect_rmse_res_dir_template = 'expect_rmse_reports{}'
# rnd_res_dir_template = 'rand_reports{}'
# rnd_rmse_report_template = 'rmse_report{}.csv'
# experiments_res_dir_template = 'experiment{}'


class MaxStdModes(Enum):
    basic = '' #chooses maxstd_univ
    maxstd_univ = 'maxstd_univ'
    maxstd_indiv = 'maxstd_indiv'
    linval_univ = 'linval_univ'
    linval_indiv = 'linval_indiv'
    drill_apart = 'drill_apart'
    tri_centers = 'tri_centers'
    unique_triangles = 'unique_triangles'
    kriging = 'kriging'


# geologic quantities' names
porosity = 'por'
thickness = 'thickness'
top_surf = 'top'
bot_surf = 'bot'

# layered_model_dict_keys
q2f = 'q2f'  # quantity-to-forecaster
q2m = 'q2m'  # quantity-to-metrics

# multiindex column names
value_of_quantity = 'value'
column_levels = ['layer', 'quantity', 'characteristics']
