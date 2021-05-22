"""
Utils.py
"""
FROM_CHECKPOINT = False
VALIDATE = True
CHECKPOINT_NAME = 'checkpoint2.pt'
MODEL_NAME = 'resnet34'

"""
Debugging Constants
"""
DEBUG = True
N_IMAGES = 50000
NUM_CLASSES = 64500

"""
Directories
"""
TRAIN_DIR: str = "./project/project_git/MPI_Pytorch/data/train/"
# TRAIN_DIR: str = "./data/img/train/"
TRAIN_FILE: str = "metadata.json"
TEST_DIR: str = "./project/project_git/MPI_Pytorch/data/test/"
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
BATCH_SIZE = 128
LR = 4e-4
NUM_EPOCHS = 10
FEATURE_EXTRACT = False  # Flag for feature extracting. When False, we fine-tune the whole model,
                         # when True we only update the reshaped layer params
USE_PRETRAINED = True # Flag for using pretrained model or not
