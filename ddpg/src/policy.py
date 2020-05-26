import numpy as np


class ActRandom:
    def __init__(self, actionLow, actionHigh):
        self.actionLow = actionLow
        self.actionHigh = actionHigh

    def __call__(self):
        action = (np.random.uniform(self.actionLow, self.actionHigh), np.random.uniform(self.actionLow, self.actionHigh))
        return action


class ActDDPGOneStepWithRandomNoise:
    def __init__(self, actRandom, actByPolicyTrain, actorModel, noiseDecayStartStep):
        self.actRandom = actRandom
        self.actByPolicyTrain = actByPolicyTrain
        self.actorModel = actorModel
        self.actRandom = actRandom
        self.noiseDecayStartStep = noiseDecayStartStep

    def __call__(self, observation, runTime):
        observation = np.asarray(observation).reshape(1, -1)
        if runTime <= self.noiseDecayStartStep:
            action = self.actRandom()
        else:
            action = self.actByPolicyTrain(self.actorModel, observation)[0]

        return action



class ActDDPGOneStep:
    def __init__(self, actionLow, actionHigh, actByPolicyTrain, actorModel, getNoise = None):
        self.actionLow = actionLow
        self.actionHigh = actionHigh
        self.actByPolicyTrain = actByPolicyTrain
        self.actorModel = actorModel
        self.getNoise = getNoise

    def __call__(self, observation, runTime = None):
        observation = np.asarray(observation).reshape(1, -1)
        noise = 0 if self.getNoise is None else self.getNoise(runTime)
        actionPerfect = self.actByPolicyTrain(self.actorModel, observation)[0]
        noisyAction = np.clip(noise + actionPerfect, self.actionLow, self.actionHigh)

        return noisyAction
