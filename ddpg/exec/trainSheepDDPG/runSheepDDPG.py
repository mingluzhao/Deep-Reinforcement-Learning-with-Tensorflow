import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))
sys.path.append(os.path.join(dirName, '..'))

from collections import deque
import matplotlib.pyplot as plt

from src.ddpg import actByPolicyTrain, actByPolicyTarget, evaluateCriticTarget, getActionGradients, \
    BuildActorModel, BuildCriticModel, TrainCriticBySASRQ, TrainCritic, TrainActorFromGradients, TrainActorOneStep, \
    TrainActor, TrainDDPGModels
from RLframework.RLrun import resetTargetParamToTrainParam, UpdateParameters, SampleOneStep, SampleFromMemory,\
    LearnFromBuffer, RunTimeStep, RunEpisode, RunAlgorithm
from src.policy import ActDDPGOneStep
from functionTools.loadSaveModel import GetSavePath, saveVariables
from environment.noise.noise import GetExponentialDecayGaussNoise
from environment.chasingEnv.reward import RewardSheepWithBoundaryHeuristics, GetBoundaryPunishment, RewardFunctionCompete
from environment.chasingEnv.chasingPolicy import HeatSeekingContinuousDeterministicPolicy
from environment.chasingEnv.envNoPhysics import Reset, TransitForNoPhysics, getIntendedNextState, StayWithinBoundary, \
    TransitWithSingleWolf, GetAgentPosFromState, IsTerminal


learningRateCritic = 0.01
gamma = 0.95
tau = 0.01
learningRateActor = 0.01
minibatchSize = 32
maxTimeStep = 100  #
maxEpisode = 1000
bufferSize = 20000


def main():
    numAgents = 2
    stateDim = numAgents * 2
    actionLow = -1
    actionHigh = 1
    actionBound = (actionHigh - actionLow)/2
    actionDim = 2

    buildActorModel = BuildActorModel(stateDim, actionDim, actionBound)
    actorLayerWidths = [64, 64, 64, 64]
    actorWriter, actorModel = buildActorModel(actorLayerWidths)

    buildCriticModel = BuildCriticModel(stateDim, actionDim)
    criticLayerWidths = [64, 64, 64, 64]
    criticWriter, criticModel = buildCriticModel(criticLayerWidths)

    trainCriticBySASRQ = TrainCriticBySASRQ(learningRateCritic, gamma, criticWriter)
    trainCritic = TrainCritic(actByPolicyTarget, evaluateCriticTarget, trainCriticBySASRQ)

    trainActorFromGradients = TrainActorFromGradients(learningRateActor, actorWriter)
    trainActorOneStep = TrainActorOneStep(actByPolicyTrain, trainActorFromGradients, getActionGradients)
    trainActor = TrainActor(trainActorOneStep)

    paramUpdateInterval = 1
    updateParameters = UpdateParameters(paramUpdateInterval, tau)

    modelList = [actorModel, criticModel]
    actorModel, criticModel = resetTargetParamToTrainParam(modelList)
    trainModels = TrainDDPGModels(updateParameters, trainActor, trainCritic, actorModel, criticModel)

    noiseInitVariance = 1
    varianceDiscount = .995
    noiseDecayStartStep = bufferSize
    getNoise = GetExponentialDecayGaussNoise(noiseInitVariance, varianceDiscount, noiseDecayStartStep)
    actOneStepWithNoise = ActDDPGOneStep(actionLow, actionHigh, actByPolicyTrain, actorModel, getNoise)

    learningStartBufferSize = minibatchSize
    sampleFromMemory = SampleFromMemory(minibatchSize)
    learnFromBuffer = LearnFromBuffer(learningStartBufferSize, sampleFromMemory, trainModels)

    sheepId = 0
    wolfId = 1
    getSheepPos = GetAgentPosFromState(sheepId)
    getWolfPos = GetAgentPosFromState(wolfId)

    wolfSpeed = 3
    wolfPolicy = HeatSeekingContinuousDeterministicPolicy(getWolfPos, getSheepPos, wolfSpeed)
    xBoundary = (0, 20)
    yBoundary = (0, 20)
    stayWithinBoundary = StayWithinBoundary(xBoundary, yBoundary)
    physicalTransition = TransitForNoPhysics(getIntendedNextState, stayWithinBoundary)
    transit = TransitWithSingleWolf(physicalTransition, wolfPolicy)

    sheepAliveBonus = 0 / maxTimeStep
    sheepTerminalPenalty = 20

    killzoneRadius = 1
    isTerminal = IsTerminal(getWolfPos, getSheepPos, killzoneRadius)
    getBoundaryPunishment = GetBoundaryPunishment(xBoundary, yBoundary, sheepIndex=0, punishmentVal= 10)
    rewardSheep = RewardFunctionCompete(sheepAliveBonus, sheepTerminalPenalty, isTerminal)
    getReward = RewardSheepWithBoundaryHeuristics(rewardSheep, getIntendedNextState, getBoundaryPunishment, getSheepPos)
    sampleOneStep = SampleOneStep(transit, getReward)

    runDDPGTimeStep = RunTimeStep(actOneStepWithNoise, sampleOneStep, learnFromBuffer)

    reset = Reset(xBoundary, yBoundary, numAgents)
    runEpisode = RunEpisode(reset, runDDPGTimeStep, maxTimeStep, isTerminal)

    ddpg = RunAlgorithm(runEpisode, maxEpisode)

    replayBuffer = deque(maxlen=int(bufferSize))
    meanRewardList, trajectory = ddpg(replayBuffer)

    trainedActorModel, trainedCriticModel = trainModels.getTrainedModels()

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

    plotResult = True
    if plotResult:
        plt.plot(list(range(maxEpisode)), meanRewardList)
        plt.show()

if __name__ == '__main__':
    main()

