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
    for i in range(size-3):
        comm.send(message, dest=i+3)


def read_images(df):
    directory = utils.TRAIN_DIR
    data = list(zip(df["file_name"], df["category_id"], df["node_predictor"]))
    for (fname, category_id, node_predictor) in data:
        pil_image = Image.open(os.path.join(directory, fname))
        print('Node 0 sends', fname)
        sys.stdout.flush()
        comm.send((pil_image, fname, category_id, node_predictor), dest=1)
    comm.send((None, None, None, None), dest=1)


def resize_images():
    while True:
        pil_image, filename, category_id, node_predictor = comm.recv(source=0)
        if pil_image is None:
            break
        resized_image = pil_image.resize((utils.WIDTH, utils.HEIGHT))
        # transform = transforms.Compose(
        #     [
        #         transforms.Resize((utils.WIDTH, utils.HEIGHT))
        #     ])
        # resized_image = transform(pil_image)
        comm.send((resized_image, filename, category_id, node_predictor), dest=2)
        print('Node 1 resized: ', filename)
        sys.stdout.flush()
    comm.send((None, None, None, None), dest=2)


def preprocess_image():
    while True:
        pil_image, filename, category_id, node_predictor = comm.recv(source=1)
        if pil_image is None:
            break

        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.485, 0.456, 0.406),
                                  (0.229, 0.224, 0.225))])
        normalized_img = transform(pil_image)
        # MEAN = 255 * torch.tensor([0.485, 0.456, 0.406])
        # STD = 255 * torch.tensor([0.229, 0.224, 0.225])
        # x = torch.from_numpy(np.array(pil_image))
        # x = x.type(torch.float32)
        # x = x.permute(-1, 0, 1)
        # x = (x - MEAN[:, None, None]) / STD[:, None, None]
        comm.send((normalized_img, filename, category_id), dest=node_predictor)
        print('Node 2 preprocessed: ', filename)
        sys.stdout.flush()
    send_to_predictors((None, None, None))
    # for i in range(size-3):
    #     comm.send((None, None, None), dest=i+3)


def predict(dataset_size):
    device = 'cpu'
    model, input_size = initialize_model(utils.MODEL_NAME, utils.NUM_CLASSES, use_pretrained=False, feature_extract=False)
    # model = models.resnet34()
    # model.fc = nn.Linear(512, utils.NUM_CLASSES, bias=True)
    # model.load_state_dict(torch.load('./models/best_model.pt'))
    # model.load_state_dict(torch.load('./checkpoints/checkpoint_{}_{}.pt'.format(utils.MODEL_NAME, utils.NUM_EPOCHS-1))['state_dict'])
    model.load_state_dict(torch.load('./checkpoints/checkpoint_{}.pt'.format(utils.NUM_EPOCHS-1))['state_dict'])
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
    if rank == 0:
        df = pd.read_csv("./data/train_sample.csv")[:100]
        # assign a predictor node for each image, the nodes are assigned from the uniform distribution of
        # numpy rand int
        df["node_predictor"] = np.random.randint(low=3, high=size, size=df.shape[0])
        dataset_size = df.shape[0]
        send_to_predictors(dataset_size)
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
