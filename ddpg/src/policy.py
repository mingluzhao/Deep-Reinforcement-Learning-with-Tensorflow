import numpy as np

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
