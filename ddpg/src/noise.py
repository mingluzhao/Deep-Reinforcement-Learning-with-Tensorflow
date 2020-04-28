import numpy as np
import random
import copy

EPSILON_MAX = 1
EPSILON_MIN = 0.1
EPSILON_DECAY = 1e-6

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, actionDim, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(actionDim)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.epsilon = EPSILON_MAX
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx

        return self.state

    def update(self):
        self.updateEpsilon()
        self.reset()

    def updateEpsilon(self):
        if self.epsilon - EPSILON_DECAY > EPSILON_MIN:
            self.epsilon -= EPSILON_DECAY
        else:
            self.epsilon = EPSILON_MIN


