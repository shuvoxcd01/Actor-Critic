from abc import ABC, abstractmethod

import tensorflow as tf


class BaseNetwork(tf.keras.Model, ABC):
    def __init__(self):
        super().__init__()

        self.dense1 = tf.keras.layers.Dense(units=256, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=256, activation='relu')

    @abstractmethod
    def call(self, inputs):
        pass
