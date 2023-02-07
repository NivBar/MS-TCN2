import torch

if torch.cuda.is_available():
    # machine run
    vid_list_file = "../../../datashare/APAS/folds/valid 0.txt"
    vid_list_file_tst = "../../../datashare/APAS/folds/test 0.txt"
    features_path = "../../../datashare/APAS/features/fold0/"
    gt_path = '../../../datashare/APAS/transcriptions_gestures/'
    mapping_file = "../../../datashare/APAS/mapping_gestures.txt"
    model_dir = "./models/test"
    results_dir = "./results/test"
else:
    # local run - test\debug
    vid_list_file = "./datashare/APAS/folds/"
    vid_list_file_tst = "./datashare/APAS/folds/"
    features_path = "./datashare/APAS/features/fold"
    gt_path = './datashare/APAS/transcriptions_gestures/'
    mapping_file = "./datashare/APAS/mapping_gestures.txt"
    model_dir = "./models/test"
    results_dir = "./results/test"
