import pandas as pd
import torch
import os
import sys
from mpi4py import MPI
from PIL import Image
import utils
import torch.nn as nn
from torchvision import models, transforms
from models import initialize_model
import numpy as np
import mpi_tools

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def send_to_predictors(message):
    """
    Sends an input message to all the predictor nodes.
    :param message: input message
    """
    for i in range(size - 3):
        comm.send(message, dest=i + 3)


def read_images(df):
    """
    Read each image from the dataframe column "file_name" and send the vales of the image, the file_name,
    the category_id (label) and the node_predictor to node 1.
    :param df: input dataframe
    """
    directory = utils.TRAIN_DIR
    data = list(zip(df["file_name"], df["category_id"], df["node_predictor"]))

    for (fname, category_id, node_predictor) in data:

        # open each image stated in the dataframe
        pil_image = Image.open(os.path.join(directory, fname))
        print('Node 0 sends', fname)
        sys.stdout.flush()

        # send the image to node 1
        comm.send((pil_image, fname, category_id, node_predictor), dest=1)

    # send None 4-tuple when all images are already sent to node 1
    comm.send((None, None, None, None), dest=1)


def resize_images():
    """
    Resize each image to the width and height specified in utils.py.
    Each resized image is sent to node 2
    """
    while True:

        # receive image from node 0
        pil_image, filename, category_id, node_predictor = comm.recv(source=0)

        # if received image is None, then break (no more images to resize)
        if pil_image is None:
            break

        # resizing the image
        resized_image = pil_image.resize((utils.WIDTH, utils.HEIGHT))
        # transform = transforms.Compose(
        #     [
        #         transforms.Resize((utils.WIDTH, utils.HEIGHT))
        #     ])
        # resized_image = transform(pil_image)

        # send the resized image to node 2
        comm.send((resized_image, filename, category_id, node_predictor), dest=2)
        print('Node 1 resized: ', filename)
        sys.stdout.flush()

    # send None 4-tuple when all resized-images are already sent to node 2
    comm.send((None, None, None, None), dest=2)


def preprocess_image():
    """
    Apply a transformation to each image, this transformation consist in converting the image to a tensor form and
    normalizing the image.
    """
    while True:

        # receive resized-images from node 1
        pil_image, filename, category_id, node_predictor = comm.recv(source=1)

        # if received image is None, then break (no more images to process)
        if pil_image is None:
            break

        # define the transformation for the images:
        # 1. Each image will be transformed to a tensor form
        # 2. Each image will be normalized
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.485, 0.456, 0.406),
                                  (0.229, 0.224, 0.225))])

        # apply the defined transformation to the image
        normalized_img = transform(pil_image)

        # send the image to the corresponding predictor node
        comm.send((normalized_img, filename, category_id), dest=node_predictor)
        print('Node 2 preprocessed: ', filename)
        sys.stdout.flush()

    # send None 4-tuple when all resized-images are already sent to predictor nodes
    send_to_predictors((None, None, None))


def predict(dataset_size):
    """

    :param dataset_size:
    """
    # device = 'cpu'
    model, input_size = initialize_model(utils.MODEL_NAME, utils.NUM_CLASSES, use_pretrained=False,
                                         feature_extract=False)
    # model = models.resnet34()
    # model.fc = nn.Linear(512, utils.NUM_CLASSES, bias=True)
    # model.load_state_dict(torch.load('./models/best_model.pt'))
    # model.load_state_dict(torch.load('./checkpoints/checkpoint_{}_{}.pt'.format(utils.MODEL_NAME, utils.NUM_EPOCHS-1))['state_dict'])
    # model.load_state_dict(
    #     torch.load('./project/project_git/MPI_Pytorch/checkpoints/checkpoint_{}.pt'.format(utils.MODEL_NAME))[
    #         'state_dict'])
    model.load_state_dict(
        torch.load(utils.CHECKPOINT_DIR+utils.MODEL_NAME)[
            'state_dict'])
    model.eval()
    running_corrects = 0
    while True:
        image, filename, label = comm.recv(source=2)
        if image is None:
            print("Finished node {}, acc {}".format(rank, running_corrects / dataset_size))
            sys.stdout.flush()
            break
        # image = image.to(device)
        # label = label.to(device)
        with torch.no_grad():
            output = model(image[None, ...])
        _, pred = torch.max(output, 1)
        running_corrects += torch.sum(pred == label)
        print("Node {} predicted {} for {} with true label: {}".format(rank, pred, filename, label))
        sys.stdout.flush()


def pipeline():
    """
    Creates the evaluation pipeline where:
    1. Node 0 reads the test dataframe and sends each image to node 1.
    It also randomly labels nodes as predictors (without considering nodes 0, 1 & 2)
    2. Node 1 resizes the dataset images and send them to node 1
    3. Node 2 normalizes the images and transform them to tensors and send them to node [3,N] where N is number of nodes
    4. Nodes 3 to N loads a testing model & checkpoint and predicts the label of each image
    """
    if rank == 0:
        df = pd.read_csv("./project/project_git/MPI_Pytorch/data/test_sample.csv")

        # assign a predictor node for each image, the nodes are assigned from the uniform distribution of
        # numpy rand int
        df["node_predictor"] = np.random.randint(low=3, high=size, size=df.shape[0])

        # send the total number of samples to each predictor
        dataset_size = df.shape[0]
        send_to_predictors(dataset_size)

        # read the images from the dataframe
        read_images(df)

    elif rank == 1:
        resize_images()

    elif rank == 2:
        preprocess_image()

    else:
        dataset_size = comm.recv(source=0)
        predict(dataset_size)


if __name__ == '__main__':
    pipeline()
