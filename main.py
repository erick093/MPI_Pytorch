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


def main():
    device = 'cpu'  # don't use cuda, causes an O.S crash
    torch.set_num_threads(1)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # LOGGER.info("Device Loaded: {}".format(device))
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
        if utils.DEBUG:
            folds = train_df.sample(utils.N_IMAGES, random_state=0).reset_index(drop=True).copy()
        else:
            folds = train_df.copy()
        filenames_to_scatter = np.array_split(folds, size)

    my_filenames = comm.scatter(filenames_to_scatter, root=0)
    LOGGER.info("_Files Received: {}".format(len(my_filenames)))
    train_set = GetData(Dir=utils.TRAIN_DIR, FNames=my_filenames['file_name'].values,
                        Labels=my_filenames['category_id'].values, Transform=Transform)
    train_loader = DataLoader(train_set, batch_size=utils.BATCH_SIZE, shuffle=True)
    LOGGER.info("_Loader Created")
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
        #model.eval()
        LOGGER.info("_Epoch: {} | Loss: {} | Time: {}".format(epoch, tr_loss/len(train_loader), end_time - init_time))
        #print('Epoch: %d | Loss: %.4f' % (epoch, tr_loss))


if __name__ == '__main__':
    main()
