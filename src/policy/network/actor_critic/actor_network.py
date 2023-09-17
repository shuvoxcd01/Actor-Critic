import tensorflow as tf

from src.policy.network.actor_critic.base_network import BaseNetwork


class ActorNetwork(BaseNetwork):
    def __init__(self, num_actions: int):
        super(ActorNetwork, self).__init__()
        self.actor = tf.keras.layers.Dense(units=num_actions)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc(x)

        return self.actor(x)
