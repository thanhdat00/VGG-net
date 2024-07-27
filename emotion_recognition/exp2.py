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
import tensorflow as tf
import argparse
import os
import sys

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True, help="path to output checkpoint directory")
ap.add_argument("-m", "--model", type=str, help="path to *specific* model checkpoint to load")
ap.add_argument("-s", "--start-epoch", type=int, default=0, help="epoch to restart training at")
ap.add_argument("-exp", "--experiment", required=True, help="Experiment number")
ap.add_argument("-te", "--total-epoch", type=int, required=True, help="total number of epochs to train")
ap.add_argument("-l", "--learning-rate", type=float, default=1e-3, help="initial learning rate")
ap.add_argument("-opt", "--optimizer", required=True, help="define optimizer")
args = vars(ap.parse_args())


exp_no = args["experiment"]
total_epoch = args["total_epoch"]
optimizer = args["optimizer"]
learning_rate_float = float(args["learning_rate"])


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
    if optimizer == "adam":
        opt = Adam(learning_rate=learning_rate_float)

    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
else:
    print("[INFO] loading {}...".format(args["model"]))
    model = load_model(args["model"])

    print("[INFO] old learning rate: {}".format(K.get_value(model.optimizer.learning_rate)))
    # print(model.optimizer.learning_rate.dtype)
    # learning_rate = K.variable(0.001)
    # print(learning_rate.dtype)
    # print(type(model.optimizer.learning_rate)).
    previous_opt = model.optimizer

    new_opt = Adam(
    learning_rate=learning_rate_float,
    beta_1=previous_opt.beta_1,
    beta_2=previous_opt.beta_2,
    epsilon=previous_opt.epsilon,
    amsgrad=previous_opt.amsgrad,
    weight_decay=previous_opt.weight_decay,
    clipnorm=previous_opt.clipnorm,
    clipvalue=previous_opt.clipvalue,
    global_clipnorm=previous_opt.global_clipnorm,
    use_ema=previous_opt.use_ema,
    ema_momentum=previous_opt.ema_momentum,
    ema_overwrite_frequency=previous_opt.ema_overwrite_frequency,
    loss_scale_factor=previous_opt.loss_scale_factor,
    gradient_accumulation_steps=previous_opt.gradient_accumulation_steps,
    name = previous_opt.name)

    model.optimizer = new_opt
    # new_optimizer.learning_rate = learning_rate_float

    # model.compile(loss=model.loss, optimizer=new_optimizer, metrics=model.metrics)

    print("[INFO] new learning rate: {}".format(K.get_value(model.optimizer.learning_rate)))

figPath = os.path.sep.join([config.OUTPUT_PATH, f'vggnet_emotion{exp_no}.png'])
jsonPath = os.path.sep.join([config.OUTPUT_PATH, f'vggnet_emotion{exp_no}.json'])

# Add the lr_scheduler to your callbacks list
callbacks = [
    EpochCheckpoint(args["checkpoints"], every=5, startAt=args["start_epoch"], exp=exp_no),
    TrainingMonitor(figPath, jsonPath=jsonPath, startAt=args["start_epoch"]),
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

trainGen.close()
valGen.close()