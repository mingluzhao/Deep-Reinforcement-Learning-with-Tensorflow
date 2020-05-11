import numpy as np
import random
import copy

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, actionDim, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(actionDim)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        EPSILON_MAX = 1
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
        EPSILON_MIN = 0.1
        EPSILON_DECAY = 1e-6
        if self.epsilon - EPSILON_DECAY > EPSILON_MIN:
            self.epsilon -= EPSILON_DECAY
        else:
            self.epsilon = EPSILON_MIN


class GetExponentialDecayGaussNoise:
    def __init__(self, noiseInitVariance, varianceDiscount, noiseDecayStartStep, minVar = 0):
        self.noiseInitVariance = noiseInitVariance
        self.varianceDiscount = varianceDiscount
        self.noiseDecayStartStep = noiseDecayStartStep
        self.minVar = minVar

    def __call__(self, runStep):
        var = self.noiseInitVariance
        if runStep > self.noiseDecayStartStep:
            var = self.noiseInitVariance* self.varianceDiscount ** (runStep - self.noiseDecayStartStep)
            var = max(var, self.minVar)
        noise = np.random.normal(0, var)
        if runStep % 1000 == 0:
            print('noise Variance', var)

        return noise
