import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import utils
import paths
from eval import read_file


tasks = {"acc_loss": True, "conf_mat": False}
colors = {0: "orange", 1: "gold", 2: "green", 3: "blue", 4: "purple"}
lable_dict = {0: "G0", 1: "G1", 2: "G2", 3: "G3", 4: "G4", 5: "G5"}
df = pd.read_csv("old_final_training_results_15_complete.csv")
epochs = len(set(df.Epoch))
chosen = utils.chosen_epochs(path="old_final_training_results_15_complete.csv", write_=False)
x=1

if tasks["acc_loss"]:
    for measure in ["Accuracy", "Loss"]:
        for type_ in ["Train", "Validation"]:
            type_df = df[df.Type == type_]
            for i in range(5):
                plt.plot(range(1, epochs+1), type_df[type_df.Split == i][measure], label=f"fold {i}", color=colors[i])
                plt.plot(chosen[i], list(type_df[type_df.Split == i][measure])[chosen[i] - 1],
                         marker="o", markeredgecolor=colors[i], markerfacecolor="red")
            plt.legend()
            plt.ylim(0)
            plt.xlabel('Epoch')
            plt.ylabel(measure)
            plt.title(f"{type_} {measure} by fold")
            plt.xticks(np.arange(1, epochs, 1.0))
            plt.savefig(f"./graphs/old_{epochs}_{type_}_{measure}.jpg")
            plt.show()

if tasks["conf_mat"]:
    list_of_videos = []
    _, vid_list_file_tst_folds, _ = utils.get_folds_paths()
    actions_dict = utils.create_actions_dict()
    for file_list in vid_list_file_tst_folds:
        list_of_videos.extend(read_file(file_list).split('\n')[:-1])

    gt_content, recog_content = [], []
    for vid in list_of_videos:
        gt_content.extend(list(utils.gt_converter(paths.gt_path + vid.replace("csv", "txt"))))
        recog_content.extend([float(actions_dict[label]) for label in
                              read_file(paths.results_dir + "-old/" + vid.split('.')[0]).split('\n')[1].split()])

    cutoff = min(len(gt_content), len(recog_content))
    cm = confusion_matrix(gt_content[:cutoff], recog_content[:cutoff])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(f"./graphs/old_conf_matrix.jpg")
    plt.show()
