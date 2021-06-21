from mpi4py import MPI
import pandas as pd
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import transforms
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


def main():
    device = 'cpu'  # use CPU for training
    torch.set_num_threads(1)  # deactivate torch default parallelism
    filenames_to_scatter = None

    """
    -------------------------------------#
    Define image transformations:
    1. Transform Images to Tensors
    2. Resize images to a given WIDTH and HEIGHT
    3. Normalize a tensor image with mean and standard deviation
    -------------------------------------#
    """
    Transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((utils.WIDTH, utils.HEIGHT)),
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    """
    -------------------------------------#
    LOADING TRAINING & TESTING DATASETS
    -------------------------------------#
    """
    # node 0 will read the train/test csv datasets and split the training dataset
    if rank == 0:
        LOGGER.info("Reading Training & Testing samples")
        if utils.DEBUG:  # If debugging, we will use a sample of the original dataset( defined in create_dataset.py)
            # using test_sample.csv since it has a smaller quantity of data points.
            df_test = pd.read_csv("./project/project_git/MPI_Pytorch/data/test_sample.csv")
            sample = df_test.sample(1000, random_state=0).reset_index(drop=True).copy()
            train_sample, test_sample = train_test_split(sample, test_size=0.2).copy()
        else:
            train_sample = pd.read_csv("./project/project_git/MPI_Pytorch/data/train_sample.csv")
            test_sample = pd.read_csv("./project//project_git/MPI_Pytorch/data/test_sample.csv")
        # split the training dataset into to the total number of nodes
        filenames_to_scatter = np.array_split(train_sample, size)
    # scatter the splits of the dataset from node 0 to all the nodes
    my_filenames = comm.scatter(filenames_to_scatter, root=0)

    """
    -------------------------------------#
    CREATING TRAINING & TESTING DATA LOADERS
    -------------------------------------#
    """
    LOGGER.info("_Files Received: {}".format(len(my_filenames)))
    train_set = GetData(Dir=utils.TRAIN_DIR, FNames=my_filenames['file_name'].values,
                        Labels=my_filenames['category_id'].values, Transform=Transform)
    LOGGER.info("_Training Dataset Object Created")
    train_loader = DataLoader(train_set, batch_size=utils.BATCH_SIZE, shuffle=True)
    LOGGER.info("_Training Loader Created")
    if rank == 0 and utils.VALIDATE:  # node 0 will perform validation step (calculate accuracy of each training step)
        # uncomment to use the test_sample
        # val_set = GetData(Dir=utils.TEST_DIR, FNames=test_sample['file_name'].values,
        #                   Labels=test_sample['category_id'].values, Transform=Transform)
        val_set = GetData(Dir=utils.TRAIN_DIR, FNames=train_sample['file_name'].values,
                          Labels=train_sample['category_id'].values, Transform=Transform)
        LOGGER.info("_Validation Dataset Object Created")
        val_loader = DataLoader(val_set, batch_size=utils.BATCH_SIZE, shuffle=False)
        LOGGER.info("_Validation Loader Created")

    """
    -------------------------------------#
    INITIALIZING TRAINING MODEL
    -------------------------------------#
    """
    # training CNN model can be defined as one of the 7 models defined in models.py
    # resnet18, resnet34, alexnet, vgg, squeezenet, densenet, inception.
    model, input_size = initialize_model(utils.MODEL_NAME, utils.NUM_CLASSES, utils.FEATURE_EXTRACT,
                                         use_pretrained=True)

    LOGGER.info("_Model Created: {}".format(utils.MODEL_NAME))
    optimizer = torch.optim.Adam(model.parameters(), lr=utils.LR)
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

    """
    -------------------------------------#
    TRAINING LOOP
    -------------------------------------#
    """
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
            # each node will sync their gradients and average them, before using them in the optimizer step.
            mpi_tools.mpi_avg_grads(model)
            optimizer.step()
            tr_loss += loss.detach().item()

        end_time = MPI.Wtime()
        LOGGER.info(
            "_Epoch: {} | Train Loss: {} | Time: {}".format(epoch, tr_loss / len(train_loader), end_time - init_time))
        # node 0 will checkpoint each training step, we can use a checkpoint to resume training or to test a model
        if rank == 0:
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': tr_loss / len(train_loader)
            }
            LOGGER.info("_Creating a checkpoint at epoch {}".format(epoch))
            save_checkpoint(checkpoint, epoch, utils.MODEL_NAME, utils.CHECKPOINT_DIR, utils.MODELS_DIR)
            LOGGER.info("_Checkpoint saved")
            # if validate step is activated, node 0 will calculate the accuracy of the training step
            if utils.VALIDATE:
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


if __name__ == '__main__':
    main()
