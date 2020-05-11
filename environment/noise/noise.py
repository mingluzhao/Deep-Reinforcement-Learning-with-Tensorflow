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
