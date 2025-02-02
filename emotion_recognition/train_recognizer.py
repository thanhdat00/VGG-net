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
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import LearningRateScheduler, Callback
import argparse
import os
import sys

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True, help="path to output checkpoint directory")
ap.add_argument("-m", "--model", type=str, help="path to *specific* model checkpoint to load")
ap.add_argument("-s", "--start-epoch", type=int, default=0, help="epoch to restart training at")
ap.add_argument("-exp", "--experiment", required=True, help="Experiment number")
ap.add_argument("-ec", "--total-epoch", required=True, help="Total epoch")
ap.add_argument("-opt", "optimizer", required=True, help="define optimizer")
args = vars(ap.parse_args())


exp_no = args["experiment"]
total_epoch = args["total-epoch"]
optimizer = args["optimizer"]

# # Initialize the training and validation data augmentations
# trainAug = ImageDataGenerator(rotation_range=10, zoom_range=0.1,
#                               horizontal_flip=True, rescale=1/255.0, fill_mode="nearest")
# valAug = ImageDataGenerator(rescale=1/255.0)

# # Initialize the image preprocessors
# iap = ImageToArrayPreprocessor()

# # Initialize the training and validation dataset generators
# trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, config.BATCH_SIZE, aug=trainAug, preprocessors=[iap], classes=config.NUM_CLASSES)
# valGen = HDF5DatasetGenerator(config.VAL_HDF5, config.BATCH_SIZE, aug=valAug, preprocessors=[iap], classes=config.NUM_CLASSES)

# # If there is no specific model checkpoint supplied, then initialize the network and compile the model
# if args["model"] is None:
#     print("[INFO] compiling model...")
#     model = EmotionVGGNet.build(width=48, height=48, depth=1, classes=config.NUM_CLASSES)
#     opt = Adam(learning_rate=1e-2)
#     model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# # Otherwise, load the checkpoint from disk
# else:
#     print("[INFO] loading {}...".format(args["model"]))
#     model = load_model(args["model"])
#     start_epoch = args["start_epoch"]

# # Construct the set of callbacks
# callbacks = [
#     EpochCheckpoint(args["checkpoints"], every=5, startAt=args["start_epoch"]),
#     TrainingMonitor(config.OUTPUT_PATH)
# ]

# # Train the network
# model.fit(
#     trainGen.generator(),
#     steps_per_epoch=trainGen.numImages // config.BATCH_SIZE,
#     validation_data=valGen.generator(),
#     validation_steps=valGen.numImages // config.BATCH_SIZE,
#     epochs=100,
#     callbacks=callbacks, verbose=1
# )

# # Close the HDF5 datasets
# trainGen.close()
# valGen.close()



class PrintLearningRate(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = K.get_value(self.model.optimizer.learning_rate)
        print(f"Epoch {epoch+1}: Learning rate is {lr}")

print_lr = PrintLearningRate()

trainAug = ImageDataGenerator(horizontal_flip=True)
valAug = ImageDataGenerator()
# # Initialize the image preprocessors
iap = ImageToArrayPreprocessor()

# # Initialize the training and validation dataset generators
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, config.BATCH_SIZE, aug=trainAug, preprocessors=[iap], classes=config.NUM_CLASSES)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, config.BATCH_SIZE, aug=valAug, preprocessors=[iap], classes=config.NUM_CLASSES)

if args["model"] is None:
    print("[INFO] compiling model...")
    model = EmotionVGGNet.build(width=48, height=48, depth=1, classes=config.NUM_CLASSES)

    if optimizer not in ["adam", "sgd"]:
        print('Invalid optimizer')
        sys.exit()
    if optimizer == "sg":
        opt = SGD(learning_rate=1e-2, momentum=0.9, nesterov=True)
    if optimizer == "adam":
        opt = Adam(learning_rate=1e-3)

    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
else:
    print("[INFO] loading {}...".format(args["model"]))
    model = load_model(args["model"])

    print("[INFO] old learning rate: {}".format(K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr, 1e-2)
    print("[INFO] new learning rate: {}".format(K.get_value(model.optimizer.lr)))

figPath = os.path.sep.join([config.OUTPUT_PATH, f'vggnet_emotion{exp_no}.png'])
jsonPath = os.path.sep.join([config.OUTPUT_PATH, f'vggnet_emotion{exp_no}.json'])

# Define the learning rate schedule function
def lr_schedule(epoch):
    if epoch > 40:
        return 1e-4
    elif epoch > 20:
        return 1e-3
    return 1e-2

# Initialize the learning rate scheduler callback
lr_scheduler = LearningRateScheduler(lr_schedule)

# Add the lr_scheduler to your callbacks list
callbacks = [
    EpochCheckpoint(args["checkpoints"], every=5, startAt=args["start_epoch"]),
    TrainingMonitor(figPath, jsonPath=jsonPath, startAt=args["start_epoch"]),
    lr_scheduler,
    print_lr
]

# Train the network
model.fit(
    trainGen.generator(),
    steps_per_epoch=trainGen.numImages // config.BATCH_SIZE,
    validation_data=valGen.generator(),
    validation_steps=valGen.numImages // config.BATCH_SIZE,
    epochs=total_epoch,
    # max_queue_size=config.BATCH_SIZE * 2,
    callbacks=callbacks, verbose=1
)

# # Train the network
# model.fit(
#     trainGen.generator(),
#     steps_per_epoch=trainGen.numImages // config.BATCH_SIZE,
#     validation_data=valGen.generator(),
#     validation_steps=valGen.numImages // config.BATCH_SIZE,
#     epochs=100,
#     callbacks=callbacks, verbose=1
# )


trainGen.close()
valGen.close()