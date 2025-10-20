<a href="https://github.com/TomRea1/RL-for-Space-Invaders/edit/main/README.md#overview">Overview  </a>
<a href="https://github.com/TomRea1/RL-for-Space-Invaders/edit/main/README.md#architecture">Architecture  </a>
<a href="https://github.com/TomRea1/RL-for-Space-Invaders/edit/main/README.md#hyperparameters">Hyperparameters  </a>
<a href="https://github.com/TomRea1/RL-for-Space-Invaders/edit/main/README.md#results-summary">Results  </a>

# RL-for-Space-Invaders
# Deep Q-Networks for Space Invaders 🕹️

## Overview
This project implements Deep Q-Networks (DQN) and Double DQN (DDQN) to learn to play Space Invaders directly from raw pixels.  
It follows the original DeepMind approach (Mnih et al., 2013) with modern PyTorch tooling and aims to explore stability, sample efficiency, and performance differences between DQN and DDQN.

<p align="center">
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/4dba585d-52c1-4b32-a6e9-0e58afa073c7" />
</p>

-----

## Architecture
**Network:** 
The input to the Q-Network is of shape 4 x 84 x 84, i.e four 84 * 84 greyscale images stacked on top of one another to capture motion. The first convolutional layer in the network convolves this input with 16, 8*8 kernels with a stride of 4 and applies ReLU. The Second convolutional layer does the same with 32, 4*4 kernels with a stride of 2. The resultant tensor is then flattened into a vector of shape (3200,). Finally, the first fully connected layer consists of 256 units with ReLU activation outputting a vector (256,) and the final fully connected layer maps to one unit per action space (No ReLU here obviously). The openai gym action space for Space Invaders is of length 6. Essentially the network will learn through loss + backprop, to estimate a Q value for each unit 1-6 which represents each possible action in a given state. The Q value is an estimate of future expected reward from continued play given this current choice of action. For example if given a state (time step in the game), the network yields a Q-value of 10 for output unit 3, the network thinks that if the agent chooses action 3, it can expect to recieve a cumulative reward of 10 over the next time steps before game over. 

-----

## Hyperparameters 

| Parameter | Value |
|------------|--------|
| Replay Buffer | 1,000,000 |
| Batch Size | 32 |
| Learning Rate | 2.5e-4 |
| Optimiser | RMSProp (α=0.95, ε=0.01) |
| Discount (γ) | 0.99 |
| ε-greedy Schedule | 1.0 → 0.1 over 1M frames |
| Target Network Update | every 10,000 steps |

-----

## Results Summary
<p>
  <img width="470" alt="image" src="https://github.com/user-attachments/assets/68a1a97a-0f2f-4f08-bf09-209d9ca9e6ac" />
  <img width="470" alt="image" src="https://github.com/user-attachments/assets/97c9b9c4-88fb-43a8-b0c5-4e8b9c58abc5" />
</p>

- DDQN shows smoother convergence and higher average rewards.
- DQN learns effectively but occasionally exhibits overestimation spikes.
- Training ran on Google Colab A100 GPU (~8 hours).
