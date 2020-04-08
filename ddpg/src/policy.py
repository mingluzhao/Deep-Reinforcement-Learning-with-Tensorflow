import numpy as np

class HeatSeekingContinuousDeterministicPolicy:
    def __init__(self, getPredatorPos, getPreyPos, actionMagnitude):
        self.getPredatorPos = getPredatorPos
        self.getPreyPos = getPreyPos
        self.actionMagnitude = actionMagnitude

    def __call__(self, state):
        action = np.array(self.getPreyPos(state)) - np.array(self.getPredatorPos(state))
        actionL2Norm = np.linalg.norm(action, ord=2)
        if actionL2Norm != 0:
            action = action / actionL2Norm
            action *= self.actionMagnitude

        actionTuple = tuple(action)
        return actionTuple


class ActByDDPG:
    def __init__(self, actByAngle, actByPolicyTrain, actorModel):
        self.actByAngle = actByAngle
        self.actByPolicyTrain = actByPolicyTrain
        self.actorModel = actorModel

    def __call__(self, states):
        stateBatch = np.asarray(states).reshape(1, -1)
        actionModelOutput = self.actByPolicyTrain(self.actorModel, stateBatch)
        sheepAction = self.actByAngle(actionModelOutput)[0]

        return sheepAction
