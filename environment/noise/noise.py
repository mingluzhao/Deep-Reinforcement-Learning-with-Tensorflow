import numpy as np

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


class OrnsteinUhlenbeckActionNoise(object):
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class ExponentialDecayGaussNoise(object):
    def __init__(self, noiseInitVariance, varianceDiscount, noiseDecayStartStep, minVar = 0):
        self.noiseInitVariance = noiseInitVariance
        self.varianceDiscount = varianceDiscount
        self.noiseDecayStartStep = noiseDecayStartStep
        self.minVar = minVar
        self.runStep = 0

    def getNoise(self):
        var = self.noiseInitVariance
        if self.runStep > self.noiseDecayStartStep:
            var = self.noiseInitVariance* self.varianceDiscount ** (self.runStep - self.noiseDecayStartStep)
            var = max(var, self.minVar)

        noise = np.random.normal(0, var)
        if self.runStep % 1000 == 0:
            print('noise Variance', var)
        self.runStep += 1
        return noise

    def reset(self):
        return


class MinusDecayGaussNoise(object):
    def __init__(self, noiseInitVariance, varianceDiscount, noiseDecayStartStep, minVar = 0):
        self.noiseInitVariance = noiseInitVariance
        self.varianceDiscount = varianceDiscount
        self.noiseDecayStartStep = noiseDecayStartStep
        self.minVar = minVar
        self.runStep = 0

    def getNoise(self):
        noiseVar = self.noiseInitVariance
        if self.runStep > self.noiseDecayStartStep:
            noiseVar = self.noiseInitVariance - self.varianceDiscount * (self.runStep - self.noiseDecayStartStep)
            noiseVar = max(noiseVar, self.minVar)

        noise = np.random.normal(0, noiseVar)
        if self.runStep % 1000 == 0:
            print('noise Variance', noiseVar)

        self.runStep += 1
        return noise

    def reset(self):
        return