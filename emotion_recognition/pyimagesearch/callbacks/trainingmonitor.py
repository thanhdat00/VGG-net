from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
import numpy as np
import json
import os

class TrainingMonitor(Callback):
    def __init__(self, figPath, jsonPath=None, startAt=0):
        print(f'start at: {startAt}')
        super(TrainingMonitor, self).__init__()
        self.figPath = figPath
        self.jsonPath = jsonPath
        self.startAt = startAt

    def on_train_begin(self, logs=None):
        self.H = {}
        if self.jsonPath is not None:
            if os.path.exists(self.jsonPath):
                self.H = json.loads(open(self.jsonPath).read())
                if self.startAt > 0:
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.startAt]

    def on_epoch_end(self, epoch, logs=None):
        for (k, v) in logs.items():
            l = self.H.get(k, [])
            l.append(v)
            self.H[k] = l
        if self.jsonPath is not None:
            f = open(self.jsonPath, "w")
            f.write(json.dumps(self.H))
            f.close()
        if len(self.H["loss"]) > 1:
            N = np.arange(0, len(self.H["loss"]))
            plt.style.use("ggplot")
            plt.figure()

            # Plot loss on the left y-axis
            plt.plot(N, self.H["loss"], label="train_loss", color="blue")
            plt.plot(N, self.H["val_loss"], label="val_loss", color="black")

            # Plot accuracy on the right y-axis
            ax1 = plt.gca()
            ax2 = ax1.twinx()
            ax2.plot(N, self.H["accuracy"], label="train_acc", color="red")
            ax2.plot(N, self.H["val_accuracy"], label="val_acc", color="green")

            # Set labels and legends
            ax1.set_xlabel("Epoch #")
            ax1.set_ylabel("Loss")
            ax2.set_ylabel("Accuracy")
            ax1.legend(loc="upper left")
            ax2.legend(loc="upper right")

            plt.title("Training Loss and Accuracy [Epoch {}]".format(epoch))
            plt.savefig(self.figPath)
            plt.close()
