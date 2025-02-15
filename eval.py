#!/usr/bin/python2.7
# adapted from: https://github.com/colincsl/TemporalConvolutionalNetworks/blob/master/code/metrics.py

import numpy as np
# import argparse
import utils
import paths
from clearml import Logger, Task

if utils.clearml_flag:
    task = Task.init(project_name='CVOR_PROJ', task_name='TEST-EVAL')
    # TODO: check if can be changed (Ilanit)
    task.set_user_properties({"name": "backbone", "description": "network type", "value": "mstcn++"})


def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    return content


def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i)
    return labels, starts, ends


def levenstein(p, y, norm=False):
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row + 1, n_col + 1], float)
    for i in range(m_row + 1):
        D[i, 0] = i
    for i in range(n_col + 1):
        D[0, i] = i

    for j in range(1, n_col + 1):
        for i in range(1, m_row + 1):
            if y[j - 1] == p[i - 1]:
                D[i, j] = D[i - 1, j - 1]
            else:
                D[i, j] = min(D[i - 1, j] + 1,
                              D[i, j - 1] + 1,
                              D[i - 1, j - 1] + 1)

    if norm:
        score = (1 - D[-1, -1] / max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score


def edit_score(recognized, ground_truth, norm=True, bg_class=["background"]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)


def f_score(recognized, ground_truth, overlap, bg_class=["background"]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0 * intersection / union) * ([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)


def main(ret_dict=False, name = "new"):
    actions_dict = utils.create_actions_dict()
    ground_truth_path = paths.gt_path
    recog_path = paths.results_dir #+ f"-{name}"

    list_of_videos = []
    _, vid_list_file_tst_folds, _ = utils.get_folds_paths()
    for file_list in vid_list_file_tst_folds:
        list_of_videos.extend(read_file(file_list).split('\n')[:-1])

    overlap = [.1, .25, .5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

    correct = 0
    total = 0
    edit = 0

    for vid in list_of_videos:
        gt_file = ground_truth_path + vid.replace("csv", "txt")
        gt_content = list(utils.gt_converter(gt_file))

        recog_file = recog_path + "/" + vid.split('.')[0]
        recog_content = [float(actions_dict[label]) for label in read_file(recog_file).split('\n')[1].split()]

        for i in range(min(len(gt_content), len(recog_content))):
            total += 1
            if gt_content[i] == recog_content[i]:
                correct += 1

        edit += edit_score(recog_content, gt_content)

        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score(recog_content, gt_content, overlap[s])
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1

    acc = (100 * float(correct) / total)
    edit = ((1.0 * edit) / len(list_of_videos))
    print("Acc: %.4f" % (acc))
    print('Edit: %.4f' % (edit))

    if utils.clearml_flag:
        # clearml block
        Logger.current_logger().report_text(f"Test Results:\nTest Acc: {acc}\nTest Edit: {edit}")

    res_dict = dict()
    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s] + fp[s])
        recall = tp[s] / float(tp[s] + fn[s])

        f1 = 2.0 * (precision * recall) / (precision + recall)

        f1 = np.nan_to_num(f1) * 100
        print('F1@%0.2f: %.4f' % (overlap[s], f1))

        res_dict[f"f1@{overlap[s]}"] = f1
        if utils.clearml_flag:
            # clearml block
            Logger.current_logger().report_text(f"F1@{overlap[s]}: {f1}")
    if ret_dict:
        res_dict["acc"], res_dict["edit"] = acc, edit
        return res_dict


if __name__ == '__main__':
    main()
