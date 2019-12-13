import os
import gym
import numpy as np
import tensorflow as tf

# Killing optional CPU driver warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class Reinforce(tf.keras.Model):
    def __init__(self, state_size, num_actions):
        """
        The Reinforce class that inherits from tf.keras.Model
        The forward pass calculates the policy for the agent given a batch of states.

        :param state_size: number of parameters that define the state. You don't necessarily have to use this, 
                           but it can be used as the input size for your first dense layer.
        :param num_actions: number of actions in an environment
        """
        super(Reinforce, self).__init__()
        self.num_actions = num_actions

        # TODO: Define network parameters and optimizer
        self.state_size = state_size
        self.hidden_size = 512
        self.dense1 = tf.keras.layers.Dense(
            self.hidden_size, input_shape=[-1, self.state_size], activation="relu"
        )
        self.dense2 = tf.keras.layers.Dense(self.num_actions, activation="softmax")
        self.learning_rate = .001
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    @tf.function
    def call(self, states):
        """
        Performs the forward pass on a batch of states to generate the action probabilities.
        This returns a policy tensor of shape [episode_length, num_actions], where each row is a
        probability distribution over actions for each state.

        :param states: An [episode_length, state_size] dimensioned array
        representing the history of states of an episode
        :return: A [episode_length,num_actions] matrix representing the probability distribution over actions
        of each state in the episode
        """
        # TODO: implement this ~
        x = self.dense1(states)
        x = self.dense2(x)
        return x

    def loss(self, states, actions, discounted_rewards):
        """
        Computes the loss for the agent. Make sure to understand the handout clearly when implementing this.

        :param states: A batch of states of shape [episode_length, state_size]
        :param actions: History of actions taken at each timestep of the episode (represented as an [episode_length] array)
        :param discounted_rewards: Discounted rewards throughout a complete episode (represented as an [episode_length] array)
        :return: loss, a TensorFlow scalar
        """
        # TODO: implement this uWu
        # Hint: Use gather_nd to get the probability of each action that was actually taken in the episode.
        prbs = self(states)
        # prbs 1st d is the step of the ep, 2nd d is in that step, for each action, the probability of it
        prb_actions = [prbs[i][actions[i]] for i in range(len(actions))]
        negative_log_prbs = tf.math.negative(tf.math.log(prb_actions))
        discounts = negative_log_prbs * discounted_rewards
        sm = tf.reduce_sum(discounts)
        return sm
