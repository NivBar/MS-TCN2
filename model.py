#!/usr/bin/python2.7
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np
from loguru import logger
from clearml import Logger
from tqdm import tqdm
import pandas as pd
from os import listdir
import paths
import utils


class MS_TCN2(nn.Module):
    def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes):
        super(MS_TCN2, self).__init__()
        self.PG = Prediction_Generation(num_layers_PG, num_f_maps, dim, num_classes)
        self.Rs = nn.ModuleList(
            [copy.deepcopy(Refinement(num_layers_R, num_f_maps, num_classes, num_classes)) for s in range(num_R)])

    def forward(self, x):
        out = self.PG(x)
        outputs = out.unsqueeze(0)
        for R in self.Rs:
            out = R(F.softmax(out, dim=1))
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)

        return outputs


class Prediction_Generation(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(Prediction_Generation, self).__init__()

        self.num_layers = num_layers

        self.conv_1x1_in = nn.Conv1d(dim, num_f_maps, 1)

        self.conv_dilated_1 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2 ** (num_layers - 1 - i), dilation=2 ** (num_layers - 1 - i))
            for i in range(num_layers)
        ))

        self.conv_dilated_2 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2 ** i, dilation=2 ** i)
            for i in range(num_layers)
        ))

        self.conv_fusion = nn.ModuleList((
            nn.Conv1d(2 * num_f_maps, num_f_maps, 1)
            for i in range(num_layers)

        ))

        self.dropout = nn.Dropout()
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        f = self.conv_1x1_in(x)

        for i in range(self.num_layers):
            f_in = f
            f = self.conv_fusion[i](torch.cat([self.conv_dilated_1[i](f), self.conv_dilated_2[i](f)], 1))
            f = F.relu(f)
            f = self.dropout(f)
            f = f + f_in

        out = self.conv_out(f)

        return out


class Refinement(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(Refinement, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out


class SS_TCN(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SS_TCN, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


class Trainer:
    def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, features_dim, num_classes, dataset, split,
                 pretrained=False):

        self.model = MS_TCN2(num_layers_PG, num_layers_R, num_R, num_f_maps, features_dim, num_classes)
        self.pretrained = pretrained
        self.split = split
        self.latest_epoch = 0
        if self.pretrained and (fr"split-{split}-epoch-1" in os.listdir(paths.model_dir)):
            self.load_previous_model("model")
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes

        logger.add('logs/' + dataset + "_" + split + "_{time}.log")
        logger.add(sys.stdout, colorize=True, format="{message}")

    def load_previous_model(self, type_, optimizer=None):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        map_location = "cuda:0" if torch.cuda.is_available() else 'cpu'
        model_list = [m for m in listdir(paths.model_dir) if (type_ in m) & (self.split in m)]
        max_ind = max([int(model.split(".")[0].split("-")[-1]) for model in model_list])
        self.latest_epoch = max_ind
        if type_ == "model":
            self.model.load_state_dict(
                torch.load(paths.model_dir + f"/split-{self.split}-epoch-" + str(max_ind) + f".{type_}",
                           map_location=map_location))
        elif type_ == "opt":
            optimizer.load_state_dict(
                torch.load(paths.model_dir + f"/split-{self.split}-epoch-" + str(max_ind) + f".{type_}",
                           map_location=map_location))
            if map_location == "cuda:0":
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()

    def train(self, save_dir, batch_gen_train, batch_gen_val, train_df, num_epochs, batch_size, learning_rate, split,
              device):
        print(
            f"\n\n##### running new model #####\nsplit-{split}\nplanned epochs per model - {utils.num_epochs}\navailable folds - {utils.available_folds}\n\n")
        data = []
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        if self.pretrained and (fr"split-{split}-epoch-1" in os.listdir(paths.model_dir)):
            self.load_previous_model("opt", optimizer=optimizer)

        for epoch in range(self.latest_epoch, self.latest_epoch + num_epochs):
            self.model.train()
            self.model.to(device)
            epoch_loss = 0
            correct = 0
            total = 0
            print(f"Train epoch number: {epoch + 1}")
            for _ in tqdm(range(len(batch_gen_train.list_of_examples))):
                batch_input, batch_target, mask = batch_gen_train.next_batch(batch_size)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                optimizer.zero_grad()
                predictions = self.model(batch_input)

                loss = 0
                for p in predictions:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    loss += 0.15 * torch.mean(torch.clamp(
                        self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                        max=16) * mask[:, :, 1:])

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            batch_gen_train.reset()
            torch.save(self.model.state_dict(), save_dir + f"/split-{split}-epoch-" + str(epoch + 1) + ".model")
            torch.save(optimizer.state_dict(), save_dir + f"/split-{split}-epoch-" + str(epoch + 1) + ".opt")
            logger.info("[epoch %d]: epoch train loss = %f,   train acc = %f" %
                        (epoch + 1, epoch_loss / len(batch_gen_train.list_of_examples), float(correct) / total))

            train_loss = epoch_loss / len(batch_gen_train.list_of_examples)
            train_acc = float(correct) / total

            # clearml block
            if utils.clearml_flag:
                Logger.current_logger().report_scalar(title="train_loss", series="loss", iteration=(epoch + 1),
                                                      value=train_loss)
                Logger.current_logger().report_scalar(title="train_acc", series="accuracy", iteration=(epoch + 1),
                                                      value=train_acc)

            data.append(
                {"Split": split, "Type": "Train", "Epoch": epoch + 1, "Loss": train_loss, "Accuracy": train_acc})

            ##### Validation Section #####

            with torch.no_grad():
                print(f"Validation epoch number: {epoch + 1}")
                self.model.eval()
                self.model.to(device)
                epoch_loss = 0
                correct = 0
                total = 0
                for _ in tqdm(range(len(batch_gen_val.list_of_examples))):
                    batch_input, batch_target, mask = batch_gen_val.next_batch(batch_size)
                    batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                    # optimizer.zero_grad()
                    predictions = self.model(batch_input)

                    loss = 0
                    for p in predictions:
                        loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes),
                                        batch_target.view(-1))
                        loss += 0.15 * torch.mean(torch.clamp(
                            self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)),
                            min=0,
                            max=16) * mask[:, :, 1:])

                    epoch_loss += loss.item()
                    # loss.backward()
                    # optimizer.step()

                    _, predicted = torch.max(predictions[-1].data, 1)
                    correct += ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
                    total += torch.sum(mask[:, 0, :]).item()

                batch_gen_val.reset()
                # torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
                # torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
                logger.info("[epoch %d]: epoch valid loss = %f,   valid acc = %f" %
                            (epoch + 1, epoch_loss / len(batch_gen_train.list_of_examples), float(correct) / total))

                valid_loss = epoch_loss / len(batch_gen_val.list_of_examples)
                valid_acc = float(correct) / total

                # clearml block
                if utils.clearml_flag:
                    Logger.current_logger().report_scalar(title="valid_loss", series="loss", iteration=(epoch + 1),
                                                          value=valid_loss)
                    Logger.current_logger().report_scalar(title="valid_acc", series="accuracy", iteration=(epoch + 1),
                                                          value=valid_acc)

                data.append({"Split": split, "Type": "Validation", "Epoch": epoch + 1, "Loss": valid_loss,
                             "Accuracy": valid_acc})
        df = pd.DataFrame(data)
        return pd.concat([train_df, df])

    def predict(self, model_dir, results_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate,
                split):
        print(f"##### prediction - model: split-{split}-epoch-{epoch} #####")
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + f"/split-{split}-epoch-" + str(epoch) + ".model"))
            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            for vid in tqdm(list_of_vids):
                # print vid
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                predictions = self.model(input_x)
                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze()
                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[
                                                                    list(actions_dict.values()).index(
                                                                        predicted[i].item())]] * sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()
