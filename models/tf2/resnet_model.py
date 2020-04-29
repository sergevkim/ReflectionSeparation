import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, Flatten, Dense
from tensorflow.keras import Model, Sequential


class BasicBlock(Layer):
    def __init__(self, filter_num, strides=1):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2D(filters=filter_num, kernel_size=(3, 3), strides=strides, padding='same')
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(filters=filter_num, kernel_size=(3, 3), strides=1, padding='same')
        self.bn2 = BatchNormalization()
        if strides != 1:    # think about functional api
            self.downsample = Sequential()
            self.downsample.add(Conv2D(filters=filter_num, kernel_size=(1, 1), strides=strides))
            self.downsample.add(BatchNormalization())
        else:
            self.downsample = lambda x: x

    def __call__(self, inputs, training=None):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1



class ResModel(Model):
    def __init__(self):
        super(ResModel, self).__init__()
        self.conv1 = Conv2D(32, 3, padding=1, activation="relu")
        self.flatten = Flatten()
        self.d1 = Dense(128, activation="relu")
        self.d2 = Dense(10, activation="softmax")

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return x

    def make_model(self):
        self.conv1 = 

