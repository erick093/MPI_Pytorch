import pandas as pd
from shutil import copy2
import os
from tqdm import tqdm
import json
import utils
from sklearn.model_selection import train_test_split
tqdm.pandas()


def read_json(input_file):
    try:
        with open(input_file, "r", encoding="ISO-8859-1") as file:
            ann_file = json.load(file)
        return ann_file
    except OSError as err:
        print("OS error: {0}".format(err))


def copy_file(src, mode):
    dst_path = "./data/img/{}/".format(mode)+src
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    copy2(utils.TRAIN_DIR+src, dst_path)


def create_dataframe(ann_file):
    _img = pd.DataFrame(ann_file['images'])
    _ann = pd.DataFrame(ann_file['annotations']).drop(columns='image_id')
    df = _img.merge(_ann, on='id')
    return df


if __name__ == '__main__':

    main_input_file = utils.TRAIN_DIR + utils.TRAIN_FILE
    ann_file = read_json(main_input_file)
    main_df = create_dataframe(ann_file)
    sample = main_df.sample(utils.N_IMAGES, random_state=0).reset_index(drop=True)
    train_sample, test_sample = train_test_split(sample, test_size=0.2).copy()
    test_sample.to_csv('./data/test_sample.csv')
    train_sample.to_csv('./data/train_sample.csv')
    print("Created train & test samples")
    df_test = pd.read_csv("./data/test_sample.csv")
    df_train = pd.read_csv("./data/train_sample.csv")
    print("creating test dataset...")
    df_test.file_name.progress_apply(copy_file, mode="test")
    print("creating train dataset...")
    df_train.file_name.progress_apply(copy_file, mode="train")
