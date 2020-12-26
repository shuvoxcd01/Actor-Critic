import tensorflow as tf

from src.policy.network.classic_control.base_network import BaseNetwork


class ActorNetwork(BaseNetwork):
    def __init__(self, num_actions: int):
        super(ActorNetwork, self).__init__()
        self.actor = tf.keras.layers.Dense(units=num_actions)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)

        return self.actor(x)
