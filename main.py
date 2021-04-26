from pycocotools.coco import COCO
from mpi4py import MPI
import json
from datetime import datetime
import pandas as pd
import logging
import numpy as np
import matplotlib.pyplot as plt
import sklearn  # For LabelEncoder and Metrics
from sklearn import preprocessing  # For the 🏷 Label Encoder
import albumentations  # For Image Augmentations
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold  # For Cross Validation
from torchvision import models, transforms
import utils
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from albumentations.pytorch import ToTensorV2  # For Converting to torch.Tensor
from data_loader import GetData
import mpi_tools

comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # process rank
size = comm.Get_size()  # number of workers


# Creating a logger
def init_logger(log_file: str = 'training.log'):
    # Specify the format
    formatter = logging.Formatter('%(levelname)s:%(name)s_R{}:%(message)s'.format(rank))

    # Create a StreamHandler Instance
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    # Create a FileHandler Instance
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Create a logging.Logger Instance
    logger = logging.getLogger('Herbarium')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


LOGGER = init_logger()
if rank == 0:
    LOGGER.info('Logger Initialized')


def read_json(input_file):
    try:
        with open(input_file, "r", encoding="ISO-8859-1") as file:
            ann_file = json.load(file)
        return ann_file
    except OSError as err:
        print("OS error: {0}".format(err))


def chunk_data(lst, n):
    return [lst[i::n] for i in range(n)]


def create_dataframe(ann_file):
    _img = pd.DataFrame(ann_file['images'])
    _ann = pd.DataFrame(ann_file['annotations']).drop(columns='image_id')
    df = _img.merge(_ann, on='id')
    return df


def create_kfolds(df):
    LOGGER.info("__TEST:RECEIVED:{}, type{}".format(df, type(df)))
    train_labels = df['category_id'].values
    LOGGER.info("Creating StratifiedKFold Instance")
    kf = StratifiedKFold(n_splits=2)
    LOGGER.info("Creating Splits")
    for fold, (train_index, val_index) in enumerate(kf.split(df.values, train_labels)):
        LOGGER.info("__TEST:val index fold({}):{}".format(fold, val_index))
        df.loc[val_index, 'fold'] = int(fold)
    df['fold'] = df['fold'].astype(int)
    FOLD = 0
    train_idx = df[df['fold'] != FOLD].index
    val_idx = df[df['fold'] == FOLD].index
    return train_idx, val_idx


def main():
    device = 'cpu'  # don't use cuda, causes an O.S crash
    torch.set_num_threads(1)
    filenames_to_scatter = None
    Transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((utils.WIDTH, utils.HEIGHT)),
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    if rank == 0:
        train_input_file = utils.TRAIN_DIR + utils.TRAIN_FILE
        ann_file = read_json(train_input_file)
        LOGGER.info("JSON Train File Read")
        train_df = create_dataframe(ann_file)
        LOGGER.info("Train DataFrame Created")
        encoder = preprocessing.LabelEncoder()
        LOGGER.info("LabelEncoder Instance created ")
        LOGGER.info("Fitting the LabelEncoder Instance")
        encoder.fit(train_df['category_id'])
        LOGGER.info("Converting Labels to Normalized Encoding")
        train_df['category_id'] = encoder.transform(train_df['category_id'])
        if utils.DEBUG:
            folds = train_df.sample(utils.N_IMAGES, random_state=0).reset_index(drop=True).copy()
        else:
            folds = train_df.copy()
        filenames_to_scatter = np.array_split(folds, size)

    img_filenames = comm.scatter(filenames_to_scatter, root=0)
    img_filenames.reset_index(drop=True, inplace=True)
    LOGGER.info("_Files Received: {}".format(len(img_filenames)))
    train_idx, val_idx = create_kfolds(img_filenames)
    train_set = GetData(Dir=utils.TRAIN_DIR, FNames=img_filenames.loc[train_idx]['file_name'].values,
                        Labels=img_filenames.loc[train_idx]['category_id'].values, Transform=Transform)
    LOGGER.info("_Training Dataset Object Created ")
    val_set = GetData(Dir=utils.TRAIN_DIR, FNames=img_filenames.loc[val_idx]['file_name'].values,
                      Labels=img_filenames.loc[val_idx]['category_id'].values, Transform=Transform)
    LOGGER.info("_Validation Dataset Object Created ")
    train_loader = DataLoader(train_set, batch_size=utils.BATCH_SIZE, shuffle=False)
    LOGGER.info("_Training Loader Created")
    val_loader = DataLoader(val_set, batch_size=utils.BATCH_SIZE, shuffle=False)
    LOGGER.info("_Validation Loader Created")
    model = models.resnet34()
    LOGGER.info("_Model Created")
    model.fc = nn.Linear(512, utils.NUM_CLASSES, bias=True)
    mpi_tools.sync_params(model)
    model.to(device)
    LOGGER.info("_Model loaded")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=utils.LR)
    LOGGER.info("_Optimizer Created")
    for epoch in range(utils.NUM_EPOCHS):
        tr_loss = 0.0
        model = model.train()
        init_time = MPI.Wtime()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images.float())
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            mpi_tools.mpi_avg_grads(model)
            optimizer.step()
            tr_loss += loss.detach().item()
        end_time = MPI.Wtime()

        model.eval()
        val_loss = 0
        preds = np.zeros((len(val_set)))
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                y_preds = model(images)
            preds[i * utils.BATCH_SIZE: (i+1) * utils.BATCH_SIZE] = y_preds.argmax(1).to('cpu').numpy()
        score = sklearn.metrics.accuracy_score(img_filenames.loc[val_idx]['category_id'].values, preds)
        #LOGGER.info("_Epoch: {} | Loss: {} | Time: {}".format(epoch, tr_loss / len(train_loader), end_time - init_time))
        LOGGER.info("_Epoch: {} | Loss: {} | Accuracy: {}".format(epoch, tr_loss / len(train_loader), score))


if __name__ == '__main__':
    main()
