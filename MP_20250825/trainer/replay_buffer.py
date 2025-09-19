import random
from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, next_state, reward, done):
        experience = (state, action, next_state, reward, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(next_states),
            np.array(rewards),
            np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)
