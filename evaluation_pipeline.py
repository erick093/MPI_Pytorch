import pandas as pd
import numpy as np
import torch
import os
import sys
from mpi4py import MPI
from PIL import Image
from helpers import load_checkpoint
import utils
import torch.nn as nn
from torchvision import models, transforms

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def read_images(df):
    directory = utils.TRAIN_DIR
    data = list(zip(df["file_name"], df["category_id"]))
    for (fname, category_id) in data:
        pil_image = Image.open(os.path.join(directory, fname))
        print('Node 0 sends', fname)
        sys.stdout.flush()
        comm.send((pil_image, fname, category_id), dest=1)
    comm.send((None, None, None), dest=1)


def resize_images():
    while True:
        pil_image, filename, category_id = comm.recv(source=0)
        if pil_image is None:
            break
        resized_image = pil_image.resize((utils.WIDTH, utils.HEIGHT))
        # transform = transforms.Compose(
        #     [
        #         transforms.Resize((utils.WIDTH, utils.HEIGHT))
        #     ])
        # resized_image = transform(pil_image)
        comm.send((resized_image, filename, category_id), dest=2)
        print('Node 1 resized: ', filename)
        sys.stdout.flush()
    comm.send((None, None, None), dest=2)


def preprocess_image():
    while True:
        pil_image, filename, category_id = comm.recv(source=1)
        if pil_image is None:
            break
        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(
        #         mean=[0.485, 0.456, 0.406],
        #         std=[0.229, 0.224, 0.225],
        #     ),
        # ])
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
        comm.send((normalized_img, filename, category_id), dest=3)
        print('Node 2 preprocessed: ', filename)
        sys.stdout.flush()
    comm.send((None, None, None), dest=3)


def predict():
    device = 'cpu'
    model = models.resnet34()
    model.fc = nn.Linear(512, utils.NUM_CLASSES, bias=True)
    # model.load_state_dict(torch.load('./models/best_model.pt'))
    model.load_state_dict(torch.load('./checkpoints/checkpoint.pt')['state_dict'])
    model.eval()
    running_corrects = 0
    while True:
        image, filename, label = comm.recv(source=2)
        if image is None:
            print("Finished, acc {}".format(running_corrects / 800))
            break
        # image = image.to(device)
        # label = label.to(device)
        with torch.no_grad():
            output = model(image[None, ...])
        _, pred = torch.max(output, 1)
        running_corrects += torch.sum(pred == label)
        print("Node 3 predicted {} for {} with true label: {}".format(pred, filename, label))
        sys.stdout.flush()


def pipeline():
    if rank == 0:
        df = pd.read_csv("./data/train_sample.csv")
        read_images(df)
    elif rank == 1:
        resize_images()
    elif rank == 2:
        preprocess_image()
    elif rank == 3:
        predict()


if __name__ == '__main__':
    pipeline()
