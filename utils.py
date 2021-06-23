"""
Utils.py
"""
MODEL_NAME = 'resnet18'  # can be: [resnet18, resnet34, alexnet, vgg, squeezenet, densenet, inception]
FROM_CHECKPOINT = True  # if true, training is resumed from latest checkpoint
VALIDATE = True  # if true, a validation step is performed at the end of each epoch
CHECKPOINT_NAME = 'checkpoint_{}.pt'.format(MODEL_NAME)


"""
Debugging Constants
"""
DEBUG = True  # if true, we will use a sample of the dataset for training.
N_IMAGES = 50000  # number of images in the sample to consider for training.


"""
Directories
The Dataset paths should point to the directory where the dataset is stored (images + metadata.json)
"""
# Dataset paths
TRAIN_DIR: str = "./project/project_git/MPI_Pytorch/data/train/"
TRAIN_FILE: str = "metadata.json"
TEST_DIR: str = "./project/project_git/MPI_Pytorch/data/test/"
# model & checkpoint paths
CHECKPOINT_DIR: str = "./project/project_git/MPI_Pytorch/checkpoints/"
MODELS_DIR: str = "./project/project_git/MPI_Pytorch/models/"

"""
Image constants
"""

WIDTH = 128
HEIGHT = 128

"""
Training constants
"""
NUM_CLASSES = 64500  # number of classes in the dataset
BATCH_SIZE = 128
LR = 4e-4  # learning rate
NUM_EPOCHS = 10  # number of training epochs
FEATURE_EXTRACT = False  # Flag for feature extracting. When False, we fine-tune the whole model,
                         # when True we only update the reshaped layer params
USE_PRETRAINED = True # Flag for using pretrained model or not
