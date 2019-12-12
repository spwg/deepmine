import gym
import minerl
import datetime

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation, LeakyReLU, Flatten, Reshape, GRU, Embedding

from state_space import new_treechop_state
from action_space import new_action_treechop, moves

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

gpu_available = tf.test.is_gpu_available()
print("GPU Available: ", gpu_available)

# @tf.function
# def q_network(state, Q):
#     state = tf.image.rgb_to_grayscale(state)
#     state = tf.reshape(state, [64, 64])
#     state = tf.dtypes.cast(state, dtype=tf.int32)
#     return tf.matmul(state, Q)


class Q_Network(tf.keras.Model):
    def __init__(self):
        super(Q_Network, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
        self.alpha = 0.9
        self.model = tf.keras.Sequential()
        self.model.add(Conv2D(filters=16, kernel_size=3, strides=2, padding='same', activation=LeakyReLU()))
        self.model.add(Conv2D(filters=8, kernel_size=3, strides=2, padding='same', activation=LeakyReLU()))
        self.model.add(Conv2D(filters=4, kernel_size=3, strides=2, padding='same', activation=LeakyReLU()))
        self.model.add(Reshape((-1, )))
        self.model.add(Dense(36, dtype=tf.float32))
        self.model.add(Dense(8, dtype=tf.float32))

    @tf.function
    def call(self, state):
        state = tf.dtypes.cast(state, dtype=tf.float32)
        state = tf.reshape(state, [1, 64, 64, 3])
        return self.model(state)

    def loss(self, q_values, q_values_next, reward, next_action):
        td_error = q_values.numpy()
        td_error[0][next_action] = reward + self.alpha * np.max(q_values_next)
        return tf.reduce_sum(tf.square(td_error - q_values))


def main():
    try:
        y = 0.9
        E = 2000
        epochs = 250
        n_steps = 2500
        r_tot = 0
        model = Q_Network()
        print("Making environment.")
        env = gym.make("MineRLTreechop-v0")
        print("Made environment.")
        all_rewards = []
        with tf.device('/device:GPU:0'):
            for i in range(epochs):
                a = datetime.datetime.now()
                print("Epoch", i, "/", epochs)
                e = E / (i + E)
                state = env.reset()
                actions_history = []
                rewards = []
                for j in range(0, n_steps):
                    if j % 100 == 0:
                        print("Step", j, "/", n_steps)
                    with tf.GradientTape() as tape:
                        q_values = model(state["pov"])
                        if np.random.rand(1) < e:
                            next_action = env.action_space.sample()
                            next_state, reward, done, _ = env.step(next_action)
                            max_action, max_val = None, float('-inf')
                            for i in range(len(moves)):
                                if next_action is None or next_action[moves[i]] > max_val:
                                    max_action = i
                                    max_val = next_action[moves[i]]
                            next_action = max_action
                        else:
                            next_action = np.argmax(q_values, 1)[0]
                            choices = env.action_space.noop()
                            next_state, reward, done, _ = env.step(
                                new_action_treechop(choices, next_action)
                            )
                        q_values_next = model(state["pov"])
                        loss = model.loss(q_values, q_values_next, reward, next_action)
                    gradients = tape.gradient(loss, model.trainable_variables)
                    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                    r_tot += reward
                    rewards.append(reward)
                    if done:
                        break
                    state = next_state
                all_rewards.append(rewards)
                print("Finished in", str(datetime.datetime.now() - a), "with reward", np.sum(rewards))
            print(str(all_rewards))
            print("Total reward", r_tot)
    except RuntimeError as e:
        print(e)


if __name__ == "__main__":
    main()
