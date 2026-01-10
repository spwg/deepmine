# DeepMine

Deep reinforcement learning agents for Minecraft using the MineRL environment.

**CS 1470 Deep Learning Final Project - Fall 2019**
**Brown University**

**Authors:** Nikhil Pant, Spencer Greene, Deniz Bayazit

> **Note:** This is an archived project from 2019 and has not been maintained since. The dependencies (TensorFlow 2.0.0rc0, MineRL 0.2.9) are outdated and may not install on modern systems. The code is preserved here for reference and portfolio purposes.

## Overview

This project explores how deep learning can overcome the sparse reward problem in complex environments like Minecraft. Using the [MineRL](http://minerl.io/) competition framework, we implemented and compared several reinforcement learning algorithms on navigation and resource-gathering tasks.

The sparse reward problem is fundamental in RL: unlike dense reward environments where the agent receives frequent feedback, sparse environments only provide rewards upon task completion, making it difficult for agents to learn effective policies.

## MineRL Environments

We experimented with two MineRL environments:

### Navigate Dense (`MineRLNavigateDense-v0`)
- **Goal:** Navigate to a target block indicated by a compass
- **Observation Space:** 64x64 RGB screen pixels + compass angle
- **Action Space:** attack, back, camera, forward, jump, left, right, sneak, sprint
- **Reward:** Dense rewards based on distance to goal

### Tree Chop (`MineRLTreechop-v0`)
- **Goal:** Collect 64 blocks of wood
- **Observation Space:** 64x64 RGB screen pixels
- **Action Space:** attack, back, camera, forward, jump, left, right, sneak, sprint
- **Reward:** Sparse (+1 only when collecting wood)

## Implemented Algorithms

### REINFORCE (`nav_model.py`)
Policy gradient method that learns a stochastic policy by maximizing expected cumulative reward. Uses a two-layer neural network (512 hidden units) with softmax output.

### REINFORCE with Baseline / A2C (`a2c.py`)
Actor-Critic variant that uses a learned value function as a baseline to reduce variance in policy gradient estimates. Implements both actor (policy) and critic (value) networks.

### DQN+ (`dqn.py`)
Deep Q-Network with convolutional layers designed to process visual input:
- 3 Conv2D layers (16 -> 8 -> 4 filters) with LeakyReLU activation
- Dense layer (36 units)
- Output layer (8 actions)
- Adam optimizer (lr=0.1)
- Temporal difference learning with discount factor (gamma=0.9)

## Project Structure

```
deepmine/
├── main.py              # Training loop for Treechop with REINFORCE
├── dqn.py               # DQN+ implementation for Treechop
├── a2c.py               # REINFORCE with baseline (Actor-Critic)
├── nav_model.py         # REINFORCE model for navigation
├── nav_rein.py          # Training loop for NavigateDense with REINFORCE
├── nav_action.py        # Action space utilities for navigation
├── action_space.py      # Action mapping for Treechop environment
├── state_space.py       # State preprocessing (image to tensor)
├── mine.py              # Smart bot baseline using compass
├── graph_rewards.py     # Utility to plot training rewards
├── Dockerfile           # Docker configuration for reproducibility
└── requirements.txt     # Python dependencies
```

## Results

- **Smart Bot** (uses compass): Achieves cumulative reward ~60 on NavigateDense
- **Random Bot**: Fluctuates around 0 reward
- **REINFORCE**: Performs similarly to random on NavigateDense
- **DQN+**: Shows high variance (-20 to +18 reward range) on NavigateDense
- **Treechop**: All models achieved 0 reward due to extreme sparsity

The results demonstrate that even sophisticated deep learning models struggle with sparse reward environments without additional techniques like behavioral cloning or curriculum learning.

## Key Findings

1. Dense reward shaping is critical for RL success in complex visual environments
2. Action space simplification (fixed camera angles) improved behavior but not overall performance
3. Sparse reward tasks like Treechop require either massive compute or human demonstration data
4. Future work should explore behavioral cloning and option learning using the MineRL human demonstration dataset

## References

- [MineRL Competition](http://minerl.io/)
- [MineRL Human Demonstration Dataset](http://minerl.io/dataset/)
