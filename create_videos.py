import paths
import utils
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import cv2
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
color_dict = {0: "r", 1: "b", 2: "g", 3: "y", 4: "m", 5: "c"}


def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    return content


def num_label_to_string(num):
    G_map = {0: "G0", 1: "G1", 2: "G2", 3: "G3", 4: "G4", 5: "G5"}
    str_map = {"no gesture": "G0", "needle passing": "G1", "pull the suture": "G2", "Instrument tie": "G3",
               "Lay the knot": "G4", "Cut the suture": "G5"}
    inv_str_map = {v: k for k, v in str_map.items()}
    return inv_str_map[G_map[num]]


def create_ranges_list(vid, col):
    df = pd.read_csv(fr"{vid}_full_res.csv")
    length = min(utils.len_df.loc[fr"{vid}.npy"].values)
    lst = list(df[col])
    ranges_list = []
    curr_label = lst[0]
    curr_start = 0
    for i in range(length):
        curr_index = i
        if lst[i] != curr_label:
            ranges_list.append(
                {"start": curr_start, "end": curr_index - 1, "len": curr_index - 1 - curr_start, "label": curr_label})
            curr_label = lst[i]
            curr_start = curr_index
    return ranges_list


def get_zero_pad(num):
    if num < 10:
        return "0000"
    elif num < 100:
        return "000"
    elif num < 1000:
        return "00"
    elif num < 10000:
        return "0"
    else:
        return ""


##### organize data #####

videos = ["P017_balloon2_side", "P035_balloon2_side", "P040_balloon2_side"]
actions_dict = utils.create_actions_dict()

ground_truth_path = paths.gt_path
orig_res_path = fr"./results/new_model"
new_res_path = fr"./results/old_model"

for vid in videos:
    df = pd.DataFrame()

    gt_file = ground_truth_path + vid.replace("_side", ".txt")
    gt_content = list(utils.gt_converter(gt_file))

    orig_file = orig_res_path + "/" + vid.replace("_side", "")
    orig_content = [float(actions_dict[label]) for label in read_file(orig_file).split('\n')[1].split()]

    new_file = new_res_path + "/" + vid.replace("_side", "")
    new_content = [float(actions_dict[label]) for label in read_file(new_file).split('\n')[1].split()]

    df['GT'] = pd.Series(gt_content)
    df['MSTCN2'] = pd.Series(orig_content)
    df['PMSTCN2'] = pd.Series(new_content)
    df.to_csv(fr"{vid.replace('_side', '_full_res')}.csv")

    ##### create graph images #####

plt.rcParams['figure.figsize'] = [6.4, 4]

patches = []
for k, v in color_dict.items():
    patches.append(mpatches.Patch(color=v, label=num_label_to_string(k)))

for vid in ["P017_balloon2", "P035_balloon2", "P040_balloon2"]:
    print(fr"video: {vid}")
    fig, ax = plt.subplots()
    plt.barh([" "], [0], left=[0], color="w")
    df = pd.read_csv(fr"{vid}_full_res.csv")
    last = 0
    for col in ["GT", "MSTCN2", "PMSTCN2"]:
        r_lst = create_ranges_list(vid, col)
        for r in r_lst:
            plt.barh([col], [r["len"]], left=[r["start"]], color=color_dict[r["label"]], height=0.3, align='center')
        last = int(r_lst[-1]["end"])
    plt.legend(handles=patches, loc='upper center', ncol=2, fancybox=True, shadow=True)
    for i in tqdm(range(last)):
        plt.barh(["progress"], [1], left=[i], color="black")
        if i == 0: ax.invert_yaxis()
        plt.savefig(fr"./datashare/APAS/frames/{vid}_graphs/{i}.jpeg")

##### create videos #####


for vid in ["P017_balloon2", "P035_balloon2", "P040_balloon2"]:
    print(fr"video: {vid}")
    last = int(create_ranges_list(vid, "GT")[-1]["end"])
    frames = []
    print("concatenating images...")
    for i in tqdm(range(last)):
        try:
            zero_pad = get_zero_pad(i + 1)
            image = cv2.imread(fr'./datashare/APAS/frames/{vid}_side/img_{zero_pad}{i + 1}.jpg')
            image = cv2.resize(image, (640, 400), interpolation=cv2.INTER_AREA)
            graph = cv2.imread(fr'./datashare/APAS/frames/{vid}_graphs/{i}.jpeg')
            graph = cv2.resize(graph, (640, 240), interpolation=cv2.INTER_AREA)
            im_v = cv2.vconcat([image, graph])
            im_v_reshaped = cv2.resize(im_v, (640, 640), interpolation=cv2.INTER_AREA)
            # cv2.imshow('img',im_v_reshaped)
            # cv2.waitKey(0)
            frames.append(im_v_reshaped)
        except:
            continue

    print("creating video...")
    writer = cv2.VideoWriter(f"./datashare/APAS/frames/{vid}_video.mp4", fps=30, fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                             frameSize=(640, 640))

    for i in tqdm(range(len(frames))):
        writer.write(frames[i])
    writer.release()
