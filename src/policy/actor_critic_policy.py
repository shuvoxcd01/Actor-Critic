import tensorflow as tf

from src.env.base_environment import BaseEnvironment
# from src.policy.network.actor_critic.actor_network import ActorNetwork
# from src.policy.network.actor_critic.critic_network import CriticNetwork
from src.policy.network.classic_control.actor_network import ActorNetwork
from src.policy.network.classic_control.critic_network import CriticNetwork
from src.utils.summary_writer import SummaryWriter


class ActorCriticPolicy:
    def __init__(self, environment: BaseEnvironment, summary_writer: SummaryWriter = None):
        self.env = environment
        self.summary_writer = summary_writer

        self.num_actions = self.env.get_num_actions()
        self.actor_network = ActorNetwork(num_actions=self.num_actions)
        self.critic_network = CriticNetwork()

        self.alpha_theta = 0.0005
        self.alpha_w = 0.0005
        self.gamma = 0.99
        self.max_num_steps = 1e3

    def learn_optimal_policy(self, num_epochs=10000):
        for epoch_num in range(num_epochs):
            episode_return = tf.convert_to_tensor(0, dtype=tf.float32)

            state = self.env.reset()
            done = False
            I = 1
            num_steps = 0
            while not done and num_steps < self.max_num_steps:
                num_steps += 1
                with tf.GradientTape(persistent=True) as tape:
                    unnormalized_action_probs = self.actor_network(state)
                    normalized_action_probs = tf.nn.softmax(unnormalized_action_probs)

                    action = tf.random.categorical(unnormalized_action_probs, 1)[0, 0].numpy()

                    action_prob = normalized_action_probs[0, action]
                    log_action_prob = tf.math.log(action_prob)

                    next_state, reward, done = self.env.step(action)

                    episode_return += reward

                    cur_state_value = self.critic_network(state)[0, 0]
                    next_state_value = self.critic_network(next_state)[0, 0] if not done else 0

                    delta = reward + self.gamma * next_state_value - cur_state_value

                critic_grads = tape.gradient(cur_state_value, self.critic_network.trainable_variables)
                actor_grads = tape.gradient(log_action_prob, self.actor_network.trainable_variables)

                for i in range(len(self.critic_network.trainable_variables)):
                    self.critic_network.trainable_variables[i].assign_add(
                        self.alpha_w * delta * critic_grads[i]
                    )

                for i in range(len(self.actor_network.trainable_variables)):
                    self.actor_network.trainable_variables[i].assign_add(
                        self.alpha_theta * I * delta * actor_grads[i]
                    )

                I = self.gamma ** I
                state = next_state

            if self.summary_writer:
                self.summary_writer.write_summary("Episode Return", episode_return, epoch_num)
