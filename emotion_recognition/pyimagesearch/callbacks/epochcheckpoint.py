from tensorflow.keras.callbacks import Callback
import os

class EpochCheckpoint(Callback):
    def __init__(self, outputPath, every=5, startAt=0, exp = 1):
        super(Callback, self).__init__()
        self.outputPath = outputPath
        self.every = every
        self.startAt = startAt
        self.exp = exp

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every == 0:
            p = os.path.sep.join([self.outputPath, f'exp_{self.exp}_epoch_{epoch + 1}.hdf5'])
            print("Checkpoint save as: {p}")
            self.model.save(p, overwrite=True)
