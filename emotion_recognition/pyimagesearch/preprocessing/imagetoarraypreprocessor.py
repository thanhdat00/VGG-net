# pyimagesearch/preprocessing/imagetoarraypreprocessor.py

from keras.preprocessing.image import img_to_array

class ImageToArrayPreprocessor:
    def __init__(self, dataFormat=None):
        # Store the image data format
        self.dataFormat = dataFormat

    def preprocess(self, image):
        # Apply the Keras utility that correctly rearranges the dimensions of the image
        return img_to_array(image, data_format=self.dataFormat)
