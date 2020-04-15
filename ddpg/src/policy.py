import numpy as np

class AddActionNoise():
    def __init__(self, actionNoise, noiseDecay, actionLow, actionHigh):
        self.actionNoise = actionNoise
        self.noiseDecay = noiseDecay
        self.actionLow, self.actionHigh = actionLow, actionHigh

    def __call__(self, actionPerfect, timeStep):
        noisyAction = np.random.normal(actionPerfect, self.actionNoise * (self.noiseDecay ** timeStep))
        action = np.clip(noisyAction, self.actionLow, self.actionHigh)
        return action

#
# class ActAngleWithNoise:
#     def __init__(self, actByModel, addActionNoise):
#         self.actByModel = actByModel
#         self.addActionNoise = addActionNoise
#
#     def __call__(self, actorModel, stateBatch, timeStep):
#         actionPerfect = self.actByModel(actorModel, stateBatch)
#         actionAngle = self.addActionNoise(actionPerfect, timeStep)
#         return actionAngle


class ActOneStepWithNoise:
    def __init__(self, actByModel, addActionNoise, actByAngle, transitionFunction):
        self.actByModel = actByModel
        self.addActionNoise = addActionNoise
        self.actByAngle = actByAngle
        self.transitionFunction = transitionFunction

    def __call__(self, timeStep, actorModel, state):
        stateBatch = np.asarray(state).reshape(1, -1)
        actionPerfect = self.actByModel(actorModel, stateBatch)
        actionOutput = self.addActionNoise(actionPerfect, timeStep)
        action = self.actByAngle(actionOutput)[0]
        nextState = self.transitionFunction(state, action)
        return state, actionOutput, nextState


# class ActOneStepWithNoise:
#     def __init__(self, actAngleWithNoise, actByAngle, transitionFunction):
#         self.actAngleWithNoise = actAngleWithNoise
#         self.actByAngle = actByAngle
#         self.transitionFunction = transitionFunction
#
#     def __call__(self, timeStep, actorModel, state):
#         stateBatch = np.asarray(state).reshape(1, -1)
#         actionOutput = self.actAngleWithNoise(actorModel, stateBatch, timeStep)
#         action = self.actByAngle(actionOutput)[0]
#         nextState = self.transitionFunction(state, action)
#         return state, actionOutput, nextState


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
