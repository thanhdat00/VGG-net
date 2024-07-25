# train_recognizer.py

import matplotlib
matplotlib.use("Agg")

from config import emotion_config as config
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.callbacks import EpochCheckpoint
from pyimagesearch.callbacks import TrainingMonitor
from pyimagesearch.io import HDF5DatasetGenerator
from pyimagesearch.nn.conv import EmotionVGGNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import argparse
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True, help="path to output checkpoint directory")
ap.add_argument("-m", "--model", type=str, help="path to *specific* model checkpoint to load")
ap.add_argument("-s", "--start-epoch", type=int, default=0, help="epoch to restart training at")
args = vars(ap.parse_args())

# Initialize the training and validation data augmentations
trainAug = ImageDataGenerator(rotation_range=10, zoom_range=0.1,
                              horizontal_flip=True, rescale=1/255.0, fill_mode="nearest")
valAug = ImageDataGenerator(rescale=1/255.0)

# Initialize the image preprocessors
iap = ImageToArrayPreprocessor()

# Initialize the training and validation dataset generators
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, config.BATCH_SIZE, aug=trainAug, preprocessors=[iap], classes=config.NUM_CLASSES)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, config.BATCH_SIZE, aug=valAug, preprocessors=[iap], classes=config.NUM_CLASSES)

# If there is no specific model checkpoint supplied, then initialize the network and compile the model
if args["model"] is None:
    print("[INFO] compiling model...")
    model = EmotionVGGNet.build(width=48, height=48, depth=1, classes=config.NUM_CLASSES)
    opt = Adam(learning_rate=1e-3)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Otherwise, load the checkpoint from disk
else:
    print("[INFO] loading {}...".format(args["model"]))
    model = load_model(args["model"])
    start_epoch = args["start_epoch"]

# Construct the set of callbacks
callbacks = [
    EpochCheckpoint(args["checkpoints"], every=5, startAt=args["start_epoch"]),
    TrainingMonitor(config.OUTPUT_PATH)
]

# Train the network
model.fit(
    trainGen.generator(),
    steps_per_epoch=trainGen.numImages // config.BATCH_SIZE,
    validation_data=valGen.generator(),
    validation_steps=valGen.numImages // config.BATCH_SIZE,
    epochs=100,
    callbacks=callbacks, verbose=1
)

# Close the HDF5 datasets
trainGen.close()
valGen.close()
