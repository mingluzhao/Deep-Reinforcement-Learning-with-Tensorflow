import numpy as np

np.random.seed(1)

class AddActionNoise():
    def __init__(self, actionNoise, noiseDecay, actionLow, actionHigh, memoryCapacity):
        self.actionNoise = actionNoise
        self.noiseDecay = noiseDecay
        self.actionLow, self.actionHigh = actionLow, actionHigh
        self.memoryCapacity = memoryCapacity

    def __call__(self, actionPerfect, pointer):
        noise = self.actionNoise
        if pointer > self.memoryCapacity:
            noise = self.actionNoise* self.noiseDecay ** (pointer - self.memoryCapacity)
        noisyAction = np.random.normal(actionPerfect, noise)
        action = np.clip(noisyAction, self.actionLow, self.actionHigh)

        return action


class ActOneStepWithNoise2D:
    def __init__(self, actByModel, addActionNoise, transitionFunction, rewardFunction, isTerminal):
        self.actByModel = actByModel
        self.addActionNoise = addActionNoise
        self.transitionFunction = transitionFunction
        self.rewardFunction = rewardFunction
        self.isTerminal = isTerminal

    def __call__(self, pointer, actorModel, state):
        stateBatch = np.asarray(state).reshape(1, -1)
        actionPerfect = self.actByModel(actorModel, stateBatch)[0]
        action = self.addActionNoise(actionPerfect, pointer)
        nextState = self.transitionFunction(state, action)
        reward = np.array([self.rewardFunction(state) for state in stateBatch])
        terminal = self.isTerminal(nextState)
        return state, action, reward, nextState, terminal


class ActByDDPG1D:
    def __init__(self, actByAngle, actByPolicyTrain, actorModel):
        self.actByAngle = actByAngle
        self.actByPolicyTrain = actByPolicyTrain
        self.actorModel = actorModel

    def __call__(self, states):
        stateBatch = np.asarray(states).reshape(1, -1)
        actionModelOutput = self.actByPolicyTrain(self.actorModel, stateBatch)
        sheepAction = self.actByAngle(actionModelOutput)[0]

        return sheepAction


class ActByDDPG2D:
    def __init__(self, actByPolicyTrain, actorModel):
        self.actByPolicyTrain = actByPolicyTrain
        self.actorModel = actorModel

    def __call__(self, states):
        stateBatch = np.asarray(states).reshape(1, -1)
        sheepAction = self.actByPolicyTrain(self.actorModel, stateBatch)[0]

        return sheepAction


class ActInGymWithNoise:
    def __init__(self, actionRange, actByPolicyTrain, getNoise, env):
        self.actionRange = abs(actionRange)
        self.actByPolicyTrain = actByPolicyTrain
        self.getNoise = getNoise
        self.env = env

    def __call__(self, actorModel, state, runStep):#
        noise = self.getNoise(runStep)
        stateBatch = np.asarray(state).reshape(1, -1)
        actionPerfect = self.actByPolicyTrain(actorModel, stateBatch)[0]
        action = np.clip(noise + actionPerfect, -self.actionRange, self.actionRange)
        nextState, reward, terminal, info = self.env.step(action)

        return state, action, reward, nextState, terminal


class ActDDPGOneStepWithNoise:
    def __init__(self, actionLow, actionHigh, actByPolicyTrain, getNoise):
        self.actionLow = actionLow
        self.actionHigh = actionHigh
        self.actByPolicyTrain = actByPolicyTrain
        self.getNoise = getNoise

    def __call__(self, modelList, observation, runTime):
        actorModel, criticModel = modelList
        noise = self.getNoise(runTime)
        actionPerfect = self.actByPolicyTrain(actorModel, observation)[0]
        noisyAction = np.clip(noise + actionPerfect, self.actionLow, self.actionHigh)

        return noisyAction

