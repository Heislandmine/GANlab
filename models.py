import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models


class Generator(models.Model):
    def __init__(self, *args, **kwargs):
        super(Generator, self).__init__(*args, **kwargs)
        self.loss_function = keras.losses.BinaryCrossentropy()
        self.optimizer = keras.optimizers.Adam()
        self.activation = keras.layers.LeakyReLU()
        self.activation_out = keras.layers.Activation("tanh")
        self.layer1 = keras.layers.Dense(units=128)
        self.layer2 = layers.Dense(units=256)
        self.layer_out = keras.layers.Dense(units=784)
        self.reshape = layers.Reshape((28, 28))

    @tf.function
    def call(self, inputs, training=None, mask=None):
        z1 = self.activation(self.layer1(inputs))
        z1 = self.activation(self.layer2(z1))
        y = self.activation_out(self.layer_out(z1))
        out = self.reshape(y)
        return out


class Discriminator(models.Model):
    def __init__(self, *args, **kwargs):
        super(Discriminator, self).__init__(*args, **kwargs)
        self.loss_function = keras.losses.BinaryCrossentropy()
        self.optimizer = keras.optimizers.SGD()
        self.activation = keras.layers.LeakyReLU()
        self.activation_out = keras.layers.Activation("sigmoid")
        self.layer1 = keras.layers.Dense(units=512)
        self.layer2 = layers.Dense(units=256)
        self.layer_out = keras.layers.Dense(units=1)
        self.flatten = layers.Flatten()

    @tf.function
    def call(self, inputs, training=None, mask=None):
        inputs = self.flatten(inputs)
        z1 = self.activation(self.layer1(inputs))
        z1 = self.activation(self.layer2(z1))
        out = self.activation_out(self.layer_out(z1))

        return out
