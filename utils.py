"""
This file will include utility functions we add in order to fit our surgical data and task
"""
import pandas as pd
import numpy as np
import paths
import torch


def chosen_epochs(path="new_final_training_results_15_complete.csv", write_=False):
    df = pd.read_csv(path)
    chosen = dict()
    chosen_df = df[(df.Type == "Validation")]
    for split in set(chosen_df.Split):
        max_epoch, max_acc = 0, 0
        for ind, row in chosen_df[chosen_df.Split == split].iterrows():
            if row.Accuracy > max_acc:
                max_epoch = row.Epoch
                max_acc = row.Accuracy
        chosen[split] = max_epoch
    if write_:
        with open("chosen_epochs_new.txt", "w") as f:
            f.write(str(chosen))
    return chosen


if torch.cuda.is_available():  # on machine with GPU
    available_folds = 5
    num_epochs = 5
    model_dict = chosen_epochs()
else:  # local debugging
    available_folds = 2
    num_epochs = 3
    model_dict = chosen_epochs()

start_idx = 0  # in case we want to skip training on some indexes
kin_lambda = 0.2  # (36/1280)*2 # 36/1280
len_df = pd.read_csv("length_table.csv").set_index("vid_name")
kin_features_dim = 36
clearml_flag = False


def get_folds_paths():
    vid_list_file_folds = [paths.vid_list_file + f"valid {i}.txt" for i in range(available_folds)]
    vid_list_file_tst_folds = [paths.vid_list_file_tst + f"test {i}.txt" for i in range(available_folds)]
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
