#!/usr/bin/python2.7

import torch
from model import Trainer
from batch_gen import BatchGenerator
import os
import argparse
import random
import paths
import utils
# from os import listdir
# from os.path import isfile, join
import pandas as pd
from clearml import Task

# clearml block
if utils.clearml_flag:
    task = Task.init(project_name='CVOR_PROJ', task_name='TEST-TRAIN')
    # TODO: check if can be changed (Ilanit)
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
parser.add_argument('--bz', default='1', type=int)
parser.add_argument('--lr', default='0.0005', type=float)

parser.add_argument('--num_f_maps', default='64', type=int)

# Need input
parser.add_argument('--num_epochs', default=utils.num_epochs, type=int)
parser.add_argument('--num_layers_PG', default=11, type=int)
parser.add_argument('--num_layers_R', default=10, type=int)
parser.add_argument('--num_R', default=3, type=int)

args = parser.parse_args()

num_epochs = args.num_epochs
features_dim = args.features_dim
bz = args.bz
lr = args.lr

num_layers_PG = args.num_layers_PG
num_layers_R = args.num_layers_R
num_R = args.num_R
num_f_maps = args.num_f_maps

# use the full temporal resolution @ 15fps
sample_rate = 2
# sample input features @ 15fps instead of 30 fps
# for 50salads, and up-sample the output to 30 fps
# if args.dataset == "50salads":
#    sample_rate = 2

# TODO: delete original code

# vid_list_file = "./data/"+args.dataset+"/splits/train.split"+args.split+".bundle"
# vid_list_file_tst = "./data/"+args.dataset+"/splits/test.split"+args.split+".bundle"
# features_path = "./data/"+args.dataset+"/features/"
# gt_path = "./data/"+args.dataset+"/groundTruth/"
# mapping_file = "./data/"+args.dataset+"/mapping.txt"
# model_dir = "./models/"+args.dataset+"/split_"+args.split
# results_dir = "./results/"+args.dataset+"/split_"+args.split

vid_list_file_folds, vid_list_file_tst_folds, features_path_folds = utils.get_folds_paths()
# vid_list_file = paths.vid_list_file
# vid_list_file_tst = paths.vid_list_file_tst
# features_path = paths.features_path
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

# trainer = Trainer(num_layers_PG, num_layers_R, num_R, num_f_maps, features_dim, num_classes, args.dataset, args.split)

if args.action == "train":
    train_df = pd.DataFrame()
    chosen = open("chosen_epochs.txt", "a+")
    for i in range(utils.start_idx, utils.available_folds):
        trainer = Trainer(num_layers_PG, num_layers_R, num_R, num_f_maps, features_dim, num_classes, f"fold{i}",
                          f"fold{i}")
        train_files = [tf for tf in vid_list_file_folds if vid_list_file_folds.index(tf) != i]
        val_files = [tf for tf in vid_list_file_folds if vid_list_file_folds.index(tf) == i]
        train_feature_paths = [tf for tf in features_path_folds if features_path_folds.index(tf) != i]
        valid_feature_paths = [tf for tf in features_path_folds if features_path_folds.index(tf) == i]

        batch_gen_train = BatchGenerator(num_classes, actions_dict, gt_path, train_feature_paths, sample_rate)
        batch_gen_train.read_data(train_files)

        batch_gen_val = BatchGenerator(num_classes, actions_dict, gt_path, valid_feature_paths, sample_rate)
        batch_gen_val.read_data(val_files)
        train_df = trainer.train(model_dir, batch_gen_train, batch_gen_val, train_df, num_epochs=num_epochs,
                                 batch_size=bz, learning_rate=lr, split=i, device=device)
        train_df.to_csv("temp_training_results.csv", index=False)

        # predict on test fold
        # test_files = [tf for tf in vid_list_file_tst_folds if vid_list_file_folds.index(tf) = i]
        # test_feature_paths = [tf for tf in features_path_folds if features_path_folds.index(tf) == i]=

        best_epoch = \
            train_df.iloc[train_df[(train_df.Split == i) & (train_df.Type == "Validation")]["Accuracy"].idxmax()][
                "Epoch"]
        chosen.write(f"{i}: {best_epoch}\n")
        # trainer.predict(model_dir, results_dir, *test_feature_paths, *test_files, best_epoch, actions_dict, device, sample_rate, i)
        print(
            '\033[1m' + f"\n\n### best model for split {i} was chosen from epoch number {best_epoch}\{num_epochs} ###\n\n" + '\033[0m')

    train_df.to_csv("final_training_results.csv", index=False)

if args.action == "predict":
    model_dict = utils.model_dict
    for i in range(len(model_dict)):
        test_files = vid_list_file_tst_folds[i]
        test_feature_paths = features_path_folds[i]

        trainer = Trainer(num_layers_PG, num_layers_R, num_R, num_f_maps, features_dim, num_classes, f"fold{i}",
                          f"fold{i}")
        trainer.predict(model_dir, results_dir, test_feature_paths, test_files, model_dict[i], actions_dict, device,
                        sample_rate, split=i)
