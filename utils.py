"""
This file will include utility functions we add in order to fit our surgical data and task
"""
import pandas as pd
import numpy as np
import paths


def create_actions_dict():
    file_ptr = open(paths.mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])
    return actions_dict


def gt_converter(path):
    actions_dict = create_actions_dict()
    df = pd.read_csv(path, header=None, sep=' ', names=['start', 'end', 'label'])
    gt_array = np.zeros(df.iloc[-1, 1])
    for _, row in df.iterrows():
        gt_array[int(row[0]):int(row[1]) + 1] = actions_dict[row[2]]
    return gt_array
