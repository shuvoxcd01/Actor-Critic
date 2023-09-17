from abc import ABC, abstractmethod

import tensorflow as tf


class BaseNetwork(tf.keras.Model, ABC):
    def __init__(self):
        super().__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(8, 8), strides=4, activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=2, activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(units=512, activation='relu')

    @abstractmethod
    def call(self, inputs):
        pass
