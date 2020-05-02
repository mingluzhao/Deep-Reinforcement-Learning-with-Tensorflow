import numpy as np

class ActByAngle:
    def __init__(self,actionMagnitude):
        self.actionMagnitude = actionMagnitude

    def __call__(self, actionAngle):
        positiveX = np.array([1, 0])* self.actionMagnitude
        positiveY = np.array([0, 1])* self.actionMagnitude
        action = np.cos(actionAngle) * positiveX + np.sin(actionAngle) * positiveY
        return action


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