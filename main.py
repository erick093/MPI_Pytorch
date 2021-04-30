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
import shutil
from sklearn.model_selection import train_test_split
from torchvision import models, transforms
import utils
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from data_loader import GetData
import mpi_tools
from helpers import load_checkpoint, save_checkpoint
from models import initialize_model
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


# def save_checkpoint(state, epoch, checkpoint_dir, best_model_dir, is_best=True):
#     #f_path = checkpoint_dir / 'checkpoint_{}.pt'.format(epoch)
#     f_path = checkpoint_dir + 'checkpoint.pt'
#     torch.save(state, f_path)
#     if is_best:
#         best_fpath = best_model_dir + 'best_model.pt'
#         shutil.copyfile(f_path, best_fpath)
#
#
# def load_checkpoint(checkpoint_fpath, model, optimizer):
#     checkpoint = torch.load(checkpoint_fpath)
#     model.load_state_dict(checkpoint['state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer'])
#     return model, optimizer, checkpoint['epoch']


def create_dataframe(ann_file):
    _img = pd.DataFrame(ann_file['images'])
    _ann = pd.DataFrame(ann_file['annotations']).drop(columns='image_id')
    df = _img.merge(_ann, on='id')
    return df


def main():
    device = 'cpu'  # don't use cuda, causes an O.S crash
    torch.set_num_threads(1)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    filenames_to_scatter = None
    Transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((utils.WIDTH, utils.HEIGHT)),
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    # Transform = transforms.Compose([
    #     transforms.RandomResizedCrop(utils.WIDTH),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #      ])
    if rank == 0:
        train_input_file = utils.TRAIN_DIR + utils.TRAIN_FILE
        ann_file = read_json(train_input_file)
        LOGGER.info("JSON Train File Read")
        train_df = create_dataframe(ann_file)
        LOGGER.info("Train DataFrame Created")

        if utils.DEBUG:
            sample = train_df.sample(utils.N_IMAGES, random_state=0).reset_index(drop=True).copy()
            train_sample, test_sample = train_test_split(sample, test_size=0.2).copy()
            test_sample.to_csv('./data/test_sample.csv')
            train_sample.to_csv('./data/train_sample.csv')
        else:
            train_sample, test_sample = train_test_split(train_df, test_size=0.2, stratify=train_df['category_id']).copy()  # at least n=64500 / 2,257,759
            test_sample.to_csv('./data/test_sample.csv')
        filenames_to_scatter = np.array_split(train_sample, size)

    my_filenames = comm.scatter(filenames_to_scatter, root=0)
    LOGGER.info("_Files Received: {}".format(len(my_filenames)))
    train_set = GetData(Dir=utils.TRAIN_DIR, FNames=my_filenames['file_name'].values,
                        Labels=my_filenames['category_id'].values, Transform=Transform)
    LOGGER.info("_Training Dataset Object Created")
    train_loader = DataLoader(train_set, batch_size=utils.BATCH_SIZE, shuffle=True)
    LOGGER.info("_Training Loader Created")
    if rank == 0 and utils.VALIDATE:
        # uncomment to use the test_sample
        # val_set = GetData(Dir=utils.TRAIN_DIR, FNames=test_sample['file_name'].values,
        #                   Labels=test_sample['category_id'].values, Transform=Transform)
        val_set = GetData(Dir=utils.TRAIN_DIR, FNames=train_sample['file_name'].values,
                          Labels=train_sample['category_id'].values, Transform=Transform)
        LOGGER.info("_Validation Dataset Object Created")
        val_loader = DataLoader(val_set, batch_size=utils.BATCH_SIZE, shuffle=False)
        LOGGER.info("_Validation Loader Created")
    # TODO: Add support for pretrained models from pytorch (uncomment the commented lines to activate this)
    # model, input_size = initialize_model(utils.MODEL_NAME, utils.NUM_CLASSES, utils.FEATURE_EXTRACT, use_pretrained=True)
    model = models.resnet34()
    model.fc = nn.Linear(512, utils.NUM_CLASSES, bias=True)
    # params_to_update = model.parameters()
    LOGGER.info("_Model Created")
    optimizer = torch.optim.Adam(model.parameters(), lr=utils.LR)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    LOGGER.info("_Optimizer Created")
    if utils.FROM_CHECKPOINT:
        LOGGER.info("_Loading Checkpoint")
        model, optimizer, start_epoch = load_checkpoint(utils.CHECKPOINT_DIR + utils.CHECKPOINT_NAME, model, optimizer)
        LOGGER.info("_Checkpoint loaded")
    mpi_tools.sync_params(model)
    model.to(device)
    LOGGER.info("_Model loaded to CPU")
    criterion = nn.CrossEntropyLoss()
    LOGGER.info("_Entering training Loop")
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
        LOGGER.info("_Epoch: {} | Train Loss: {} | Time: {}".format(epoch, tr_loss/len(train_loader), end_time - init_time))
        if rank == 0: # FIXME: Move model evaluation outside the for loop, its lagging the other processes
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': tr_loss/len(train_loader)
            }
            LOGGER.info("_Creating a checkpoint at epoch {}".format(epoch))
            save_checkpoint(checkpoint, epoch, utils.CHECKPOINT_DIR, utils.MODELS_DIR)
            LOGGER.info("_Checkpoint saved")
            LOGGER.info("_Evaluating model")
            running_corrects = 0
            model.eval()
            for i, (images, labels) in enumerate(val_loader):
                images = images.to(device)
                labels = labels.to(device)
                with torch.no_grad():
                    outputs = model(images)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
            epoch_acc = running_corrects.double() / len(val_loader.dataset)
            LOGGER.info("_Epoch: {} | Acc: {}".format(epoch, epoch_acc))
    # TODO: Construct the distributed prediction pipeline


if __name__ == '__main__':
    main()
