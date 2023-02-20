import torch

if torch.cuda.is_available():
    # machine run
    vid_list_file = fr"../../../datashare/APAS/folds/"
    vid_list_file_tst = fr"../../../datashare/APAS/folds/"
    features_path = fr"../../../datashare/APAS/features/fold"
    kinematics_path = fr"../../../datashare/APAS/kinematics_npy/"
    gt_path = fr"../../../datashare/APAS/transcriptions_gestures/"
    mapping_file = fr"../../../datashare/APAS/mapping_gestures.txt"
    model_dir = fr"./models/test"
    results_dir = fr"./results/test"
else:
    # local run - test\debug
    vid_list_file = fr"./datashare/APAS/folds/"
    vid_list_file_tst = fr"./datashare/APAS/folds/"
    features_path = fr"./datashare/APAS/features/fold"
    kinematics_path = fr"./datashare/APAS/kinematics_npy/"
    gt_path = fr"./datashare/APAS/transcriptions_gestures/"
    mapping_file = fr"./datashare/APAS/mapping_gestures.txt"
    model_dir = fr"./models/test"
    results_dir = fr"./results/test"
