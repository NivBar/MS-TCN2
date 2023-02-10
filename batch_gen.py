#!/usr/bin/python2.7

import torch
import numpy as np
import random

import paths
import utils


class BatchGenerator(object):
    def __init__(self, num_classes, actions_dict, gt_path, features_paths, sample_rate, kinematics=False):
        self.kinematics = kinematics
        self.list_of_examples = list()
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_paths = features_paths
        self.sample_rate = sample_rate

    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False

    def read_data(self, vid_list_file):
        for path in vid_list_file:
            file_ptr = open(path, 'r')
            self.list_of_examples.extend(file_ptr.read().split('\n')[:-1])
            file_ptr.close()
            random.shuffle(self.list_of_examples)

    def next_batch(self, batch_size):
        batch = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size

        batch_input = []
        batch_target = []
        if self.kinematics:
            batch_kin = []
        for vid in batch:
            for feat_path in self.features_paths:
                # size fitting_addition
                features = np.load(feat_path + vid.split('.')[0] + '.npy')
                if self.kinematics:
                    kinematics = np.load(paths.kinematics_path + vid.split('.')[0] + '.npy')
                # TODO: delete original code
                # file_ptr = open(self.gt_path + vid, 'r')
                # content = file_ptr.read().split('\n')[:-1]
                content = utils.gt_converter(self.gt_path + vid.replace("csv", "txt"))
                #  size referencing
                if self.kinematics:
                    cutoff = min(utils.len_df.loc[vid.replace("csv", "npy")].values)
                else:
                    cutoff = min(np.shape(features)[1], len(content))

                features = features[:, :cutoff]
                if self.kinematics:
                    kinematics = kinematics[:, :cutoff]
                content = content[:cutoff]
                batch_input.append(features[:, ::self.sample_rate])
                if self.kinematics:
                    batch_kin.append(kinematics[:, ::self.sample_rate])
                batch_target.append(content[::self.sample_rate])

        length_of_sequences = list(map(len, batch_target))
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences),
                                         dtype=torch.float)
        if self.kinematics:
            batch_kin_tensor = torch.zeros(len(batch_kin), np.shape(batch_kin[0])[0], max(length_of_sequences),
                                             dtype=torch.float)

        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long) * (-100)
        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)

        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
            if self.kinematics:
                batch_kin_tensor[i, :, :np.shape(batch_kin[i])[1]] = torch.from_numpy(batch_kin[i])
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])

        if self.kinematics:
            return batch_input_tensor, batch_target_tensor, mask, batch_kin_tensor
        else:
            return batch_input_tensor, batch_target_tensor, mask
