import os
import numpy as np

from src.ddpg_withModifiedCritic import BuildActorModel, BuildCriticModel, actByPolicyTrain, \
    actByPolicyTarget, evaluateCriticTarget, TrainCritic, getActionGradients, TrainActor, \
    addToMemory, initializeMemory, UpdateModelsByMiniBatch, RunDDPGTimeStep, RunEpisode, DDPG,\
    UpdateParameters, TrainCriticBySASRQ, TrainActorFromGradients, TrainActorOneStep
from src.reward import RewardFunctionCompete
from src.policy import AddActionNoise, HeatSeekingContinuousDeterministicPolicy, \
    ActByAngle, ActOneStepWithNoise
from src.envNoPhysics import Reset, TransitForNoPhysics, StayWithinBoundary, \
    TransitWithSingleWolf, GetAgentPosFromState, IsTerminal


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


def saveVariables(model, path):
    graph = model.graph
    saver = graph.get_collection_ref("saver")[0]
    saver.save(model, path)
    print("Model saved in {}".format(path))


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
    actorWriter, actorModel = buildActorModel(actorTrainingLayerWidths, actorTargetLayerWidths)

    buildCriticModel = BuildCriticModel(numStateSpace, actionDim)
    criticTrainingLayerWidths = [100, 100]
    criticTargetLayerWidths = criticTrainingLayerWidths
    criticWriter, criticModel = buildCriticModel(criticTrainingLayerWidths, criticTargetLayerWidths)

    sheepId = 0
    wolfId = 1
    getSheepXPos = GetAgentPosFromState(sheepId)
    getWolfXPos = GetAgentPosFromState(wolfId)
    killzoneRadius = 1
    isTerminal = IsTerminal(getWolfXPos, getSheepXPos, killzoneRadius)

    maxRunningSteps = 20
    sheepAliveBonus = 1 / maxRunningSteps
    # sheepAliveBonus = 0

    sheepTerminalPenalty = -1
    rewardSheep = RewardFunctionCompete(sheepAliveBonus, sheepTerminalPenalty, isTerminal)

    tau = 0.01
    updateParameters = UpdateParameters(tau)

    learningRateCritic = 0.001
    gamma = 0.95
    trainCriticBySASRQ = TrainCriticBySASRQ(learningRateCritic, gamma, criticWriter)
    trainCritic = TrainCritic(rewardSheep, actByPolicyTarget, evaluateCriticTarget, trainCriticBySASRQ, updateParameters)

    learningRateActor = 0.0001
    trainActorFromGradients = TrainActorFromGradients(learningRateActor, actorWriter)
    trainActorOneStep = TrainActorOneStep(actByPolicyTrain, trainActorFromGradients, getActionGradients)
    trainActor = TrainActor(trainActorOneStep, updateParameters)

    updateModelsByMiniBatch = UpdateModelsByMiniBatch(trainActor, trainCritic)

    actionNoise = 0.1
    noiseDecay = 0.999
    addActionNoise = AddActionNoise(actionNoise, noiseDecay, actionLow, actionHigh)

    actionMagnitude = 1
    actByAngle = ActByAngle(actionMagnitude)
    wolfPolicy = HeatSeekingContinuousDeterministicPolicy(getWolfXPos, getSheepXPos, actionMagnitude)

    xBoundary = (0, 20)
    yBoundary = (0, 20)
    stayWithinBoundary = StayWithinBoundary(xBoundary, yBoundary)
    physicalTransition = TransitForNoPhysics(stayWithinBoundary)
    transitionFunction = TransitWithSingleWolf(physicalTransition, wolfPolicy)

    actOneStepWithNoise = ActOneStepWithNoise(actByPolicyTrain, addActionNoise, actByAngle, transitionFunction)
    minibatchSize = 32
    runDDPGTimeStep = RunDDPGTimeStep(actOneStepWithNoise, addToMemory, updateModelsByMiniBatch, minibatchSize)
#
    maxTimeStep = 200
    reset = Reset(xBoundary, yBoundary, numAgents)
    runEpisode = RunEpisode(reset, isTerminal, runDDPGTimeStep, maxTimeStep)

    memoryCapacity = 10000
    maxEpisode = 50
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
        saveVariables(trainedActorModel, savePathActor)
    with criticModel.as_default():
        saveVariables(trainedCriticModel, savePathCritic)

if __name__ == '__main__':
    main()

