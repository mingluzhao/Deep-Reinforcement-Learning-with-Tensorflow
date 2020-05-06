from src.ddpg import *
from RLframework.RLrun import *
from environment.chasingEnv.reward import RewardFunctionCompete
from environment.chasingEnv.chasingPolicy import HeatSeekingContinuousDeterministicPolicy
from src.policy import AddActionNoise, ActOneStepWithNoise2D
from environment.chasingEnv.envNoPhysics import Reset, TransitForNoPhysics, StayWithinBoundary, \
    TransitWithSingleWolf, GetAgentPosFromState, IsTerminal
from functionTools.loadSaveModel import *

np.random.seed(1)
tf.set_random_seed(1)

visualize = True
learningRateCritic = 0.01  #
gamma = 0.95  #
tau = 0.01  #
learningRateActor = 0.01  #
minibatchSize = 32  #


def main():
    numAgents = 2
    stateDim = numAgents * 2
    actionLow = -1
    actionHigh = 1
    actionBound = 1
    actionDim = 2

    buildActorModel = BuildActorModel(stateDim, actionDim, actionBound)
    actorLayerWidths = [64, 64, 64]
    actorWriter, actorModel = buildActorModel(actorLayerWidths)

    buildCriticModel = BuildCriticModel(stateDim, actionDim)
    criticLayerWidths = [64, 64, 64]
    criticWriter, criticModel = buildCriticModel(criticLayerWidths)

    trainCriticBySASRQ = TrainCriticBySASRQ(learningRateCritic, gamma, criticWriter)
    trainCritic = TrainCritic(actByPolicyTarget, evaluateCriticTarget, trainCriticBySASRQ)

    trainActorFromGradients = TrainActorFromGradients(learningRateActor, actorWriter)
    trainActorOneStep = TrainActorOneStep(actByPolicyTrain, trainActorFromGradients, getActionGradients)
    trainActor = TrainActor(trainActorOneStep)

    paramUpdateInterval = 1
    updateParameters = UpdateParameters(paramUpdateInterval, tau)
    trainModels = TrainDDPGModels(updateParameters, trainActor, trainCritic)

    sheepId = 0
    wolfId = 1
    getSheepXPos = GetAgentPosFromState(sheepId)
    getWolfXPos = GetAgentPosFromState(wolfId)
    killzoneRadius = 1
    isTerminal = IsTerminal(getWolfXPos, getSheepXPos, killzoneRadius)

    maxRunningSteps = 20
    sheepAliveBonus = 1 / maxRunningSteps

    wolfSpeed = 3
    wolfPolicy = HeatSeekingContinuousDeterministicPolicy(getWolfXPos, getSheepXPos, wolfSpeed)

    xBoundary = (0, 20)
    yBoundary = (0, 20)
    stayWithinBoundary = StayWithinBoundary(xBoundary, yBoundary)
    physicalTransition = TransitForNoPhysics(stayWithinBoundary)
    transit = TransitWithSingleWolf(physicalTransition, wolfPolicy)

    actionNoise = 0.1
    noiseDecay = 0.999
    memoryCapacity = 1000000 #
    addActionNoise = AddActionNoise(actionNoise, noiseDecay, actionLow, actionHigh, memoryCapacity)
    sheepTerminalPenalty = -10
    getReward = RewardFunctionCompete(sheepAliveBonus, sheepTerminalPenalty, isTerminal)
    actOneStepWithNoise = ActOneStepWithNoise2D(actByPolicyTrain, addActionNoise, transit, getReward, isTerminal)

    learningStartBufferSize = minibatchSize
    runDDPGTimeStep = RunTimeStep(actOneStepWithNoise, transit, getReward, isTerminal, addToMemory,
                 trainModels, minibatchSize, learningStartBufferSize)

    maxTimeStep = 25 #
    reset = Reset(xBoundary, yBoundary, numAgents)
    runEpisode = RunEpisode(reset, runDDPGTimeStep, maxTimeStep)

    maxEpisode = 5000#
    ddpg = RunAlgorithm(runEpisode, memoryCapacity, maxEpisode)

    modelList = [actorModel, criticModel]
    modelList = resetTargetParamToTrainParam(modelList)
    meanRewardList, trajectory, trainedModelList = ddpg(modelList)

    trainedActorModel, trainedCriticModel = trainedModelList

    modelIndex = 0
    actorFixedParam = {'actorModel': modelIndex}
    criticFixedParam = {'criticModel': modelIndex}
    parameters = {'wolfSpeed':wolfSpeed, 'dimension': actionDim, 'maxEpisode': maxEpisode, 'maxTimeStep': maxTimeStep, 'minibatchSize': minibatchSize, 'gamma': gamma,
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

