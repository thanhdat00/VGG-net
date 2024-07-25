from os import path

BASE_PATH = "datasets/fer2013"
INPUT_PATH = path.sep.join([BASE_PATH, "fer2013.csv"])
NUM_CLASSES = 6
# define the path to the output training, validation, and testing
# HDF5 files
TRAIN_HDF5 = path.sep.join([BASE_PATH, "hdf5/train.hdf5"])
VAL_HDF5 = path.sep.join([BASE_PATH, "hdf5/val.hdf5"])
TEST_HDF5 = path.sep.join([BASE_PATH, "hdf5/test.hdf5"])
BATCH_SIZE = 128
# define the path to where output logs will be stored
OUTPUT_PATH = path.sep.join([BASE_PATH, "output"])