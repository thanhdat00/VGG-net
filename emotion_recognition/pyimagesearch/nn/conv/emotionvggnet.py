# pyimagesearch/nn/conv/emotionvggnet.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras import backend as K
from tensorflow.keras.layers import ELU


class EmotionVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape, activation="relu"))
        # model.add(BatchNormalization(axis=chanDim))
        # model.add(Conv2D(32, (3, 3), activation="relu"))
        # model.add(BatchNormalization(axis=chanDim))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))
        # model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
        # model.add(BatchNormalization(axis=chanDim))
        # model.add(Conv2D(64, (3, 3), activation="relu"))
        # model.add(BatchNormalization(axis=chanDim))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))
        # model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
        # model.add(BatchNormalization(axis=chanDim))
        # model.add(Conv2D(128, (3, 3), activation="relu"))
        # model.add(BatchNormalization(axis=chanDim))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))
        # model.add(Flatten())
        # model.add(Dense(512, activation="relu"))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.5))
        # model.add(Dense(classes, activation="softmax"))

        # Block #1: first CONV => RELU => CONV => RELU => POOL
        # layer set
        model.add(Conv2D(32, (3, 3), padding="same",
        kernel_initializer="he_normal", input_shape=inputShape))
        model.add(ELU())
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), kernel_initializer="he_normal",
        padding="same"))
        model.add(ELU())
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Block #2: second CONV => RELU => CONV => RELU => POOL
        # layer set
        model.add(Conv2D(64, (3, 3), kernel_initializer="he_normal",
        padding="same"))
        model.add(ELU())
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), kernel_initializer="he_normal",
        padding="same"))
        model.add(ELU())
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Block #3: third CONV => RELU => CONV => RELU => POOL
        # layer set
        model.add(Conv2D(128, (3, 3), kernel_initializer="he_normal",
        padding="same"))
        model.add(ELU())
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), kernel_initializer="he_normal",
        padding="same"))
        model.add(ELU())
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Block #4: first set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(64, kernel_initializer="he_normal"))
        model.add(ELU())
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # Block #6: second set of FC => RELU layers
        model.add(Dense(64, kernel_initializer="he_normal"))
        model.add(ELU())
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # Block #7: softmax classifier
        model.add(Dense(classes, kernel_initializer="he_normal"))
        model.add(Activation("softmax"))

        return model
