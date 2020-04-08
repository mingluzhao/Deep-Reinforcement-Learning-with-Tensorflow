import os
import numpy as np

from src.ddpg import BuildActorModel, BuildCriticModel, actByPolicyTrain, actByPolicyTarget, evaluateCriticTarget,\
    TrainCritic, getActionGradients, TrainActor, AddActionNoise, addToMemory, initializeMemory, \
    ActAngleWithNoise, ActByAngle,RunDDPGTimeStep, RunEpisode, DDPG
from src.reward import RewardFunctionCompete
from src.policy import HeatSeekingContinuousDeterministicPolicy
from src.envNoPhysics import Reset, TransitForNoPhysics, StayWithinBoundary, TransitWithSingleWolf, GetAgentPosFromState,\
    IsTerminal


class GetSavePath:
    def __init__(self, dataDirectory, extension, fixedParameters={}):
        self.dataDirectory = dataDirectory
        self.extension = extension
        self.fixedParameters = fixedParameters

    def __call__(self, parameters):
        allParameters = dict(list(parameters.items()) + list(self.fixedParameters.items()))
        sortedParameters = sorted(allParameters.items())
        nameValueStringPairs = [parameter[0] + '=' + str(parameter[1]) for parameter in sortedParameters]

        fileName = '_'.join(nameValueStringPairs) + self.extension
        fileName = fileName.replace(" ", "")

        path = os.path.join(self.dataDirectory, fileName)
        return path

def main():
    numAgents = 2
    numStateSpace = numAgents * 2
    actionLow = -np.pi
    actionHigh = np.pi
    actionRange = (actionHigh - actionLow) / 2.0
    actionDim = 1

    buildActorModel = BuildActorModel(numStateSpace, actionDim, actionRange)
    actorTrainingLayerWidths = [20, 20]
    actorTargetLayerWidths = actorTrainingLayerWidths
    actorSaver, actorWriter, actorModel = buildActorModel(actorTrainingLayerWidths, actorTargetLayerWidths)

    buildCriticModel = BuildCriticModel(numStateSpace, actionDim)
    criticTrainingLayerWidths = [100, 100]
    criticTargetLayerWidths = criticTrainingLayerWidths
    criticSaver, criticWriter, criticModel = buildCriticModel(criticTrainingLayerWidths, criticTargetLayerWidths)

    xBoundary = (0, 50)
    yBoundary = (0, 50)
    stayWithinBoundary = StayWithinBoundary(xBoundary, yBoundary)
    physicalTransition = TransitForNoPhysics(stayWithinBoundary)

    sheepId = 0
    wolfId = 1
    getSheepXPos = GetAgentPosFromState(sheepId)
    getWolfXPos = GetAgentPosFromState(wolfId)

    actionMagnitude = 1
    wolfPolicy = HeatSeekingContinuousDeterministicPolicy(getWolfXPos, getSheepXPos, actionMagnitude)
    transitionFunction = TransitWithSingleWolf(physicalTransition, wolfPolicy)

    killzoneRadius = 1
    isTerminal = IsTerminal(getWolfXPos, getSheepXPos, killzoneRadius)

    reset = Reset(xBoundary, yBoundary, numAgents)
    actionNoise = 0.1
    noiseDecay = 0.999
    addActionNoise = AddActionNoise(actionNoise, noiseDecay, actionLow, actionHigh)

    maxRunningSteps = 20
    sheepAliveBonus = 1 / maxRunningSteps
    sheepTerminalPenalty = -1
    rewardSheep = RewardFunctionCompete(sheepAliveBonus, sheepTerminalPenalty, isTerminal)

    tau = 0.01
    gamma = 0.95
    learningRateActor = 0.0001
    learningRateCritic = 0.001

    trainCritic = TrainCritic(learningRateCritic, gamma, tau, criticWriter, actByPolicyTarget, rewardSheep, evaluateCriticTarget)
    trainActor = TrainActor(learningRateActor, tau, actorWriter, actByPolicyTrain, getActionGradients)

    actWithNoise = ActAngleWithNoise(actByPolicyTrain, addActionNoise)

    minibatchSize = 32
    velocity = 1
    actByAngle = ActByAngle(velocity)
    runDDPGTimeStep = RunDDPGTimeStep(actWithNoise, actByAngle, transitionFunction, addToMemory,
                                      trainCritic, trainActor, minibatchSize, numStateSpace)

    maxTimeStep = 200
    runEpisode = RunEpisode(reset, isTerminal, runDDPGTimeStep, maxTimeStep)

    memoryCapacity = 100000
    maxEpisode = 200
    ddpg = DDPG(initializeMemory, runEpisode, memoryCapacity, maxEpisode)
    trainedActorModel, trainedCriticModel = ddpg(actorModel, criticModel)


    modelIndex = 0
    actorFixedParam = {'actorModel': modelIndex}
    criticFixedParam = {'criticModel': modelIndex}
    parameters = {'maxEpisode': maxEpisode, 'maxTimeStep': maxTimeStep, 'minibatchSize': minibatchSize, 'gamma': gamma,
                                 'learningRateActor': learningRateActor, 'learningRateCritic': learningRateCritic}

    modelSaveDirectory = "../trainedDDPGModels"
    modelSaveExtension = '.ckpt'
    getActorSavePath = GetSavePath(modelSaveDirectory, modelSaveExtension, actorFixedParam)
    getCriticSavePath = GetSavePath(modelSaveDirectory, modelSaveExtension, criticFixedParam)
    savePathActor = getActorSavePath(parameters)
    savePathCritic = getCriticSavePath(parameters)

    with actorModel.as_default():
        actorSaver.save(trainedActorModel, savePathActor)
    with criticModel.as_default():
        criticSaver.save(trainedCriticModel, savePathCritic)

if __name__ == '__main__':
    main()

