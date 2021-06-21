from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image


class GetData(Dataset):
    """
    Reads images from each input batch, return the PIL transformation of the images
    """
    def __init__(self, Dir, FNames, Labels, Transform):
        """

        :param Dir: Directory
        :param FNames: Filenames to read
        :param Labels: Labels of the filenames
        :param Transform: Transformation to apply each batch of images
        """
        self.dir = Dir
        self.fnames = FNames
        self.transform = Transform
        self.labels = Labels

    def __len__(self):
        """
        :return: length of dataset
        """
        return len(self.fnames)

    def __getitem__(self, index):
        """
        :param index:
        :return:
        """
        x = Image.open(os.path.join(self.dir, self.fnames[index]))

        if "train" in self.dir:
            return self.transform(x), self.labels[index]
        elif "test" in self.dir:
            return self.transform(x), self.fnames[index]
