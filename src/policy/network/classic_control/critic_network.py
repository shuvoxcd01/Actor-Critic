import tensorflow as tf

from src.policy.network.classic_control.base_network import BaseNetwork


class CriticNetwork(BaseNetwork):
    def __init__(self):
        super(CriticNetwork, self).__init__()
        self.critic = tf.keras.layers.Dense(units=1)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)

        return self.critic(x)
