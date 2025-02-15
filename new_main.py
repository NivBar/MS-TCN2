#!/usr/bin/python2.7

import torch
from new_model import Trainer
from batch_gen import BatchGenerator
import os
import argparse
import random
import paths
import utils
import pandas as pd
from clearml import Task
import eval

# clearml block
if utils.clearml_flag:
    task = Task.init(project_name='CVOR_PROJ', task_name='TEST-TRAIN')
    task.set_user_properties({"name": "backbone", "description": "network type", "value": "mstcn++"})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='predict')
parser.add_argument('--dataset', default="gtea")
parser.add_argument('--split', default='1')
parser.add_argument('--features_dim', default='1280', type=int)
parser.add_argument('--kin_features_dim', default=utils.kin_features_dim, type=int)
parser.add_argument('--bz', default='1', type=int)
parser.add_argument('--lr', default='0.0005', type=float)
parser.add_argument('--num_f_maps', default='64', type=int)
parser.add_argument('--num_epochs', default=utils.num_epochs, type=int)
parser.add_argument('--num_layers_PG', default=11, type=int)
parser.add_argument('--num_layers_R', default=10, type=int)
parser.add_argument('--num_R', default=3, type=int)

args = parser.parse_args()

num_epochs = args.num_epochs
features_dim = args.features_dim
kin_features_dim = args.kin_features_dim
bz = args.bz
lr = args.lr

num_layers_PG = args.num_layers_PG
num_layers_R = args.num_layers_R
num_R = args.num_R
num_f_maps = args.num_f_maps
sample_rate = 1

# utils.create_new_data_division()
vid_list_file_folds, vid_list_file_tst_folds, features_path_folds = utils.get_folds_paths()
kinematics_path = paths.kinematics_path
gt_path = paths.gt_path
mapping_file = paths.mapping_file
model_dir = paths.model_dir
results_dir = paths.results_dir

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])

num_classes = len(actions_dict)

if args.action == "train":
    train_df = pd.DataFrame()
    chosen = open(fr"chosen_epochs_new.txt", "a+")
    for i in range(utils.start_idx, utils.available_folds):
        trainer = Trainer(num_layers_PG=num_layers_PG, num_layers_R=num_layers_R, num_R=num_R, num_f_maps=num_f_maps,
                          features_dim=features_dim, num_classes=num_classes, dataset=f"fold{i}", split=f"{i}",
                          pretrained=True)

        train_feature_paths = valid_feature_paths = [features_path_folds[i]]

        train_files = [fr"./new_data_division/train_{i}.txt"]
        val_files = [fr"./new_data_division/valid_{i}.txt"]

        batch_gen_train = BatchGenerator(num_classes, actions_dict, gt_path, train_feature_paths, sample_rate,
                                         kinematics=True)
        batch_gen_train.read_data(train_files)

        batch_gen_val = BatchGenerator(num_classes, actions_dict, gt_path, valid_feature_paths, sample_rate,
                                       kinematics=True)
        batch_gen_val.read_data(val_files)

        train_df = trainer.train(save_dir=model_dir, batch_gen_train=batch_gen_train, batch_gen_val=batch_gen_val,
                                 train_df=train_df,
                                 num_epochs=num_epochs, batch_size=bz, learning_rate=lr, split=i, device=device)
        train_df.to_csv("temp_training_results.csv", index=False)

        best_epoch = \
            train_df.iloc[train_df[(train_df.Split == i) & (train_df.Type == "Validation")]["Accuracy"].idxmax()][
                "Epoch"]
        chosen.write(f"{i}: {best_epoch}\n")
        print(
            '\033[1m' + f"\n\n### best model for split {i} was chosen from epoch number {best_epoch}\{num_epochs} ###\n\n" + '\033[0m')

    epochs = sorted(set([str(x) for x in train_df["Epoch"]]))
    train_df.to_csv(f"final_training_results_epochs_{'_'.join(epochs)}.csv", index=False)

if args.action == "predict":
    model_dict = utils.model_dict
    for i in range(len(model_dict)):
        test_files = vid_list_file_tst_folds[i]
        test_feature_paths = features_path_folds[i]
        test_kin_path = paths.kinematics_path

        trainer = Trainer(num_layers_PG, num_layers_R, num_R, num_f_maps, features_dim, num_classes, f"fold{i}",
                          f"fold{i}")
        trainer.predict(model_dir, results_dir, test_feature_paths, test_files, model_dict[i], actions_dict, device,
                        sample_rate, split=i)
    eval.main()
