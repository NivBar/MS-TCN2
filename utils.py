"""
This file will include utility functions we add in order to fit our surgical data and task
"""
import pandas as pd
import numpy as np
import paths

available_folds = 2
start_idx = 0

def get_folds_paths():
    vid_list_file_folds = [paths.vid_list_file + f"valid {i}.txt" for i in range(available_folds)]
    vid_list_file_tst_folds = [paths.vid_list_file_tst + f"valid {i}.txt" for i in range(available_folds)]
    features_path_folds = [paths.features_path + f"{i}/" for i in range(available_folds)]
    return vid_list_file_folds, vid_list_file_tst_folds, features_path_folds


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
