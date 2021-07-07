from os.path import join, isfile, split, exists
from os import listdir, makedirs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse


def extract_results(location: str):
    if isfile(location):
        input_path, single_file = split(location)
        file_list = [single_file]
    else:
        input_path = location
        file_list = listdir(location)
    existence_of_corrupted_files = False
    df_cumul = pd.read_csv(join(input_path, file_list[0]), sep=',', index_col='pnum')
    for f in file_list[1:]:
        dfi = pd.read_csv(join(input_path, f), sep=',', index_col='pnum')
        if dfi.isnull().values.any():
            if not existence_of_corrupted_files:
                print('Corrupted files found:')
                existence_of_corrupted_files = True
            print(f)
        df_cumul = df_cumul + dfi
    df_cumul = df_cumul / len(file_list)
    return len(file_list), df_cumul


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, action='append',
                        help='specify path to a directory of a file containing RMSE decline profile(s).\
One may set multiple -d paths to display many plots on one figure')

    parser.add_argument('-o', "--output", type=str, action='store',
                        help='specify path to a png figure. Filename is supposed to present in a path')

    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    res = parser.parse_args()

    figsize_rel = (np.ceil(len(res.data)/2), int(len(res.data) >= 2) + 1)
    fig = plt.figure(figsize=(figsize_rel[1]*9, figsize_rel[0]*6), dpi=200)
    for i, e in enumerate(res.data):
        file_quantity, df_cumul = extract_results(e)
        plt.subplot(figsize_rel[0], figsize_rel[1], i + 1)
        for c in df_cumul:
            plt.plot(df_cumul.index, df_cumul[c], label=c)
            plt.ylim((df_cumul.values.min(), df_cumul.values.max()))
        plt.legend(loc='best')
        plt.grid(True)
        plt.xlabel('количество скважин')
        plt.ylabel('Выборочное среднее RMSE прогноза NTG\nдля всего поля')
        plt.title('{} (симуляций: {})'.format(e[-60:], file_quantity))
        plt.margins(0, 0)

    output_path, _ = split(res.output)
    if not exists(output_path):
        makedirs(output_path)
    fig.savefig(res.output)
    plt.close(fig)
