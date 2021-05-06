"""
Utils.py
"""
FROM_CHECKPOINT = True
VALIDATE = True
CHECKPOINT_NAME = 'checkpoint.pt'
MODEL_NAME = 'densenet'

"""
Debugging Constants
"""
DEBUG = True
N_IMAGES = 50000
NUM_CLASSES = 64500

"""
Directories
"""
TRAIN_DIR: str = "D:/Huge Data Set/train/"
TRAIN_FILE: str = "metadata.json"
TEST_DIR: str = "D:/Huge Data Set/test/"
CHECKPOINT_DIR: str = "./checkpoints/"
MODELS_DIR: str = "./models/"
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
NUM_EPOCHS = 1
FEATURE_EXTRACT = False
