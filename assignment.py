import os
import sys
import gym
import minerl
from pylab import *
import numpy as np
import tensorflow as tf
from reinforce import Reinforce
from reinforce_with_baseline import ReinforceWithBaseline


def visualize_data(total_rewards):
    """
    Takes in array of rewards from each episode, visualizes reward over episodes.

    :param rewards: List of rewards from all episodes
    """

    x_values = arange(0, len(total_rewards), 1)
    y_values = total_rewards
    plot(x_values, y_values)
    xlabel("episodes")
    ylabel("cumulative rewards")
    title("Reward by Episode")
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
    # TODO: Compute discounted rewards
    discounted_rewards = [0.] * len(rewards)
    discounted_rewards[-1] = rewards[-1]
    for i in reversed(range(len(rewards) - 1)):
        discounted_rewards[i] = discounted_rewards[i + 1] * discount_factor + rewards[i]
    # print(f"discount(): discounted_rewards = {discounted_rewards}")
    return discounted_rewards


def generate_trajectory(env, model):
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
    state = env.reset()
    done = False
    while not done:
        # TODO:
        # 1) use model to generate probability distribution over next actions
        states.append(state)
        prbs = model(np.array([state]))
        prbs = tf.reshape(prbs, [prbs.shape[1]])
        prbs = np.array(prbs)
        # 2) sample from this distribution to pick the next action
        action = np.random.choice(range(len(prbs)), p=prbs)
        actions.append(action)
        state, rwd, done, _ = env.step(action)
        rewards.append(rwd)
    return states, actions, rewards


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

    # TODO:
    with tf.GradientTape() as tape:
        # 1) Use generate trajectory to run an episode and get states, actions, and rewards.
        states, actions, rewards = generate_trajectory(env, model)
        states = np.array(states)
        # 2) Compute discounted rewards.
        discounted_rewards = discount(rewards)
        # 3) Compute the loss from the model and run backpropagation on the model.
        loss = model.loss(states, actions, discounted_rewards)
    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return tf.reduce_sum(rewards)


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"REINFORCE", "REINFORCE_BASELINE"}:
        print("USAGE: python assignment.py <Model Type>")
        print("<Model Type>: [REINFORCE/REINFORCE_BASELINE]")
        exit()

    env = gym.make("MineRLNavigateDense-v0")  # environment
    state_size = env.observation_space.shape[0]
    num_actions = env.action_space.n

    # Initialize model
    if sys.argv[1] == "REINFORCE":
        model = Reinforce(state_size, num_actions)
    elif sys.argv[1] == "REINFORCE_BASELINE":
        model = ReinforceWithBaseline(state_size, num_actions)

    # TODO:
    # 1) Train your model for 650 episodes, passing in the environment and the agent.
    episodes = 650
    rewards = []
    for ep in range(episodes):
        rwd = train(env, model)
        print(f"[+] Episode {ep} reward = {rwd}.")
        # 2) Append the total reward of the episode into a list keeping track of all of the rewards.
        rewards.append(rwd)
    # 3) After training, print the average of the last 50 rewards you've collected.
    print(f"[+] Avg of last 50 rewards = {np.mean(rewards[len(rewards)-50:])}.")
    # TODO: Visualize your rewards.
    visualize_data(rewards)


if __name__ == "__main__":
    main()
