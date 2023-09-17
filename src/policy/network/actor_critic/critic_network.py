import tensorflow as tf

from src.policy.network.actor_critic.base_network import BaseNetwork


class CriticNetwork(BaseNetwork):
    def __init__(self):
        super(CriticNetwork, self).__init__()
        self.critic = tf.keras.layers.Dense(units=1)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc(x)

        return self.critic(x)
