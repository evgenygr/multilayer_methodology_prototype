from os import listdir, makedirs
import pandas as pd
from os.path import join, exists
from src.constants import variance_report_filename_base, recommend_metrics_name
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.ioff()

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--directory', type=str, action='store',
                    help='specify path to a directory containing variance reports')

parser.add_argument('-o', "--output", type=str, action='store',
                    help='specify path to a directory that is meant to store png figures')

parser.add_argument('-t', "--target", type=str, action='store', default='NTG',
                    help='specify what geological value was tageted by a methodology')

parser.add_argument('-m', "--multilayer", action='store_true',
                    help='specify if the mode of methodology is multllayer')

parser.add_argument('--version', action='version', version='%(prog)s 1.0')
res = parser.parse_args()

if not exists(res.output):
    makedirs(res.output)

files = [f for f in listdir(res.directory) if variance_report_filename_base in f]
files.sort()
for i, f in tqdm(enumerate(files), total=len(files)):
    if res.multilayer:
        input_file = pd.read_csv(join(res.directory, f), sep=' ', header=[0, 1, 2], index_col=[0, 1])
        ll = input_file.columns.get_level_values('layer').to_list()
        for j, n in enumerate(ll):
            if n.startswith('Unnamed:'):
                ll[j] = ''
        qq = input_file.columns.get_level_values('quantity').to_list()
        cc = input_file.columns.get_level_values('characteristics').to_list()
        input_file.columns = pd.MultiIndex.from_tuples(
            [t for t in zip(ll, qq, cc)], names=['level', 'quantity', 'characteristics']
        )
    else:
        input_file = pd.read_csv(join(res.directory, f), sep=',')

    plt.figure(dpi=200)
    plt.title(f'Карта прогноза таргетируемой величины ({res.target})')
    if res.multilayer:
        fig = plt.scatter(
            input_file.index.get_level_values('X'),
            input_file.index.get_level_values('Y'),
            cmap='jet',
            c=input_file[('', res.target, 'value')], marker='s')
    else:
        fig = plt.scatter(input_file.X, input_file.Y, cmap='jet', c=input_file[res.target], marker='s')
    plt.colorbar()
    fig = fig.get_figure()
    fig.savefig(join(res.output, f'{res.target}_forecast_{i}.png'))
    plt.close(fig)

    plt.figure(dpi=200)
    plt.title('Карта рекомендательной метрики')
    if res.multilayer:
        fig1 = plt.scatter(
            input_file.index.get_level_values('X'),
            input_file.index.get_level_values('Y'),
            cmap='jet',
            c=input_file[('', res.target, recommend_metrics_name)], marker='s')
    else:
        fig1 = plt.scatter(input_file.X, input_file.Y, cmap='jet', c=input_file[recommend_metrics_name], marker='s')
    plt.colorbar()
    fig1 = fig1.get_figure()
    fig1.savefig(join(res.output, f'{recommend_metrics_name}_{i}.png'))
    plt.close(fig1)
