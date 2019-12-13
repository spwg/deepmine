import os
import sys
import minerl
import gym
from pylab import *
import numpy as np
import tensorflow as tf
from nav_model import Reinforce
from nav_action import interpret_probs_one
from reinforce_with_baseline import ReinforceWithBaseline


def visualize_data(total_rewards):
    """
    Takes in array of rewards from each episode, visualizes reward over episodes.

    :param rewards: List of rewards from all episodes
    """

    x_values = arange(0, len(total_rewards), 1)
    y_values = total_rewards
    plot(x_values, y_values)
    xlabel('episodes')
    ylabel('cumulative rewards')
    title('Reward by Episode')
    grid(True)
    show()


def discount(rewards, discount_factor=.99):
    """
    Takes in a list of rewards for each timestep in an episode, 
    and returns a list of the sum of discounted rewards for
    each timestep. Refer to the slides to see how this is done.

    :param rewards: List of rewards from an episode [r_{t1},r_{t2},...]
    :param discount_factor: Gamma discounting factor to use, defaults to .99
    :return: discounted_rewards: list containing the sum of discounted rewards for each timestep in the original
    rewards list
    """
    if not rewards:
        print('empty rewards!')
        return None
    n = len(rewards)
    if n == 1:
        return rewards
    i = n - 2
    discounted_rewards = [rewards[-1]]
    while i >= 0:
        prev_dis = discounted_rewards[0]
        discounted_rewards.insert(0, (prev_dis * discount_factor) + rewards[i])
        i -= 1
    return discounted_rewards

def generate_trajectory(env, model, epsilon=0.2):
    """
    Generates lists of states, actions, and rewards for one complete episode.

    :param env: The openai gym environment
    :param model: The model used to generate the actions
    :return: A tuple of lists (states, actions, rewards), where each list has length equal to the number of timesteps
    in the episode
    """
    states = []
    actions = []
    rewards = []
    cum_rewards = []
    net_rwd = 0
    state = env.reset()
    done = False
    print(model.num_actions)
    while not done:
        state = tf.convert_to_tensor(state["pov"])
        state = tf.reshape(state, shape=[-1])
        states.append(state)
        if np.random.uniform(0,1) > epsilon:
            probs = model(np.array([state]))[0]
        else:
            probs = tf.random.uniform(shape=[model.num_actions,], minval=0, maxval=1)
            probs = probs / np.sum(probs)
        action_dict, action = interpret_probs_one(probs, env)
        actions.append(action)
        state, rwd, done, _ = env.step(action_dict)
        net_rwd += rwd
        cum_rewards.append(net_rwd)
        rewards.append(rwd)

    return states, actions, rewards, cum_rewards


def train(env, model):
    """
    This function should train your model for one episode.
    Each call to this function should generate a complete trajectory for one episode (lists of states, action_probs,
    and rewards seen/taken in the episode), and then train on that data to minimize your model loss.
    Make sure to return the total reward for the episode.

    :param env: The openai gym environment
    :param model: The model
    :return: The total reward for the episode
    """
    
    with tf.GradientTape() as tape: 
        states, actions, rewards, cum_rwd = generate_trajectory(env, model)
        discounted_rewards = discount(rewards)
        loss = model.loss(np.array(states), actions, discounted_rewards)
        deltas = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(deltas, model.trainable_variables))
    return rewards, cum_rwd

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"REINFORCE", "REINFORCE_BASELINE"}:
        print("USAGE: python assignment.py <Model Type>")
        print("<Model Type>: [REINFORCE/REINFORCE_BASELINE]")
        exit()

    env = gym.make("MineRLNavigateDense-v0") # environment
    num_actions = len(env.action_space.spaces)
    print(env.action_space)
    # Initialize model
    if sys.argv[1] == "REINFORCE":
        model = Reinforce(state_size=64 * 3 * 3, num_actions=num_actions) 
    elif sys.argv[1] == "REINFORCE_BASELINE":
        model = ReinforceWithBaseline(state_size=None, num_actions=num_actions)
    r_i, cum_rwd = train(env, model)
    print("plotting")
    plt.plot(cum_rwd)
    plt.title("Cumulative Reward")
    plt.show()

if __name__ == '__main__':
    main()

