from src.env.classic_control.cartpole_environment import CartPoleEnvironment
from src.policy.actor_critic_policy import ActorCriticPolicy
from src.utils.summary_writer import SummaryWriter

env = CartPoleEnvironment()
summay_writer = SummaryWriter()

actor_critic = ActorCriticPolicy(env, summay_writer)
# state = env.reset()
#
# action_p = actor_critic.get_action(state)
#
# done = False
#
# while not done:
#     env.render()
#     action = env.get_random_action()
#
#     obs, r, done = env.step(0)

actor_critic.learn_optimal_policy()

env.close()
