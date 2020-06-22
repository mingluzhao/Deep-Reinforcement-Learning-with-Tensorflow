import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import matplotlib.pyplot as plt
from collections import deque

from src.ddpg_followPaper import *
from RLframework.RLrun import resetTargetParamToTrainParam, UpdateParameters, SampleOneStep, SampleFromMemory,\
    LearnFromBuffer, RunMultiAgentTimeStep, RunEpisode, RunAlgorithm, getBuffer
from functionTools.loadSaveModel import GetSavePath, saveVariables, saveToPickle
from environment.chasingEnv.multiAgentEnv import *
from visualize.visualizeMultiAgent import *


maxEpisode = 30000
maxTimeStep = 25

learningRateActor = 0.01#
learningRateCritic = 0.01#
gamma = 0.95 #
tau=0.01 #
bufferSize = 1e6#
minibatchSize = 32#
learningStartBufferSize = minibatchSize * maxTimeStep#
########


wolfSize = 0.075
sheepSize = 0.05
blockSize = 0.2

sheepMaxSpeed = 1.3
wolfMaxSpeed = 1.0
blockMaxSpeed = None

wolfColor = np.array([0.85, 0.35, 0.35])
sheepColor = np.array([0.35, 0.85, 0.35])
blockColor = np.array([0.25, 0.25, 0.25])

policyPath = os.path.join(dirName, '..', 'myDDPGpolicy1WolfDDPG1SheepDDPG_followPaper')


def main():
    wolvesID = [0]
    sheepsID = [1]
    blocksID = []

    numWolves = len(wolvesID)
    numSheeps = len(sheepsID)
    numBlocks = len(blocksID)

    numAgents = numWolves + numSheeps
    numEntities = numAgents + numBlocks

    entitiesSizeList = [wolfSize] * numWolves + [sheepSize] * numSheeps + [blockSize] * numBlocks
    entityMaxSpeedList = [wolfMaxSpeed] * numWolves + [sheepMaxSpeed] * numSheeps + [blockMaxSpeed] * numBlocks
    entitiesMovableList = [True] * numAgents + [False] * numBlocks
    massList = [1.0] * numEntities

    isCollision = IsCollision(getPosFromAgentState)
    rewardWolf = RewardWolf(wolvesID, sheepsID, entitiesSizeList, isCollision)
    punishForOutOfBound = PunishForOutOfBound()
    rewardSheep = RewardSheep(wolvesID, sheepsID, entitiesSizeList, getPosFromAgentState, isCollision,
                              punishForOutOfBound)

    rewardFunc = lambda state, action, nextState: \
        list(rewardWolf(state, action, nextState)) + list(rewardSheep(state, action, nextState))

    reset = ResetMultiAgentChasing(numAgents, numBlocks)
    observeOneAgent = lambda agentID: Observe(agentID, wolvesID, sheepsID, blocksID, getPosFromAgentState,
                                              getVelFromAgentState)
    observe = lambda state: [observeOneAgent(agentID)(state) for agentID in range(numAgents)]

    reshapeAction = ReshapeAction()
    getCollisionForce = GetCollisionForce()
    applyActionForce = ApplyActionForce(wolvesID, sheepsID, entitiesMovableList)
    applyEnvironForce = ApplyEnvironForce(numEntities, entitiesMovableList, entitiesSizeList,
                                          getCollisionForce, getPosFromAgentState)
    integrateState = IntegrateState(numEntities, entitiesMovableList, massList,
                                    entityMaxSpeedList, getVelFromAgentState, getPosFromAgentState)
    transit = TransitMultiAgentChasing(numEntities, reshapeAction, applyActionForce, applyEnvironForce, integrateState)

    isTerminal = lambda state: [False]* numAgents
    initObsForParams = observe(reset())
    obsShape = [initObsForParams[obsID].shape for obsID in range(len(initObsForParams))]

    worldDim = 2
    wolfObsDim = np.array(obsShape)[wolvesID[0]][0]
    sheepObsDim = np.array(obsShape)[sheepsID[0]][0]
    actionDim = worldDim * 2 + 1

    layerWidth = [64, 64]


#------------ wolf ------------------------
    buildWolfActorModel = BuildActorModel(wolfObsDim, actionDim)
    actorWriterWolf, actorModelWolf = buildWolfActorModel(layerWidth, wolvesID[0])

    buildWolfCriticModel = BuildCriticModel(wolfObsDim, actionDim)
    criticWriterWolf, criticModelWolf = buildWolfCriticModel(layerWidth, wolvesID[0])

    trainCriticBySASRQWolf = TrainCriticBySASRQ(learningRateCritic, gamma, criticWriterWolf)
    trainCriticWolf = TrainCritic(actByPolicyTarget, evaluateCriticTarget, trainCriticBySASRQWolf)

    trainActorFromGradientsWolf = TrainActorFromQVal(learningRateActor, actorWriterWolf)
    trainActorOneStepWolf = TrainActorOneStep(actByPolicyTrain, trainActorFromGradientsWolf, evaluateCriticTrain)
    trainActorWolf = TrainActor(trainActorOneStepWolf)

    paramUpdateInterval = 1 #
    updateParameters = UpdateParameters(paramUpdateInterval, tau)

    modelList = [actorModelWolf, criticModelWolf]
    # actorModelWolf, criticModelWolf = resetTargetParamToTrainParam(modelList)
    trainWolfModels = TrainDDPGModels(updateParameters, trainActorWolf, trainCriticWolf, actorModelWolf, criticModelWolf)

#------------ sheep ------------------------
    buildSheepActorModel = BuildActorModel(sheepObsDim, actionDim)
    actorWriterSheep, actorModelSheep = buildSheepActorModel(layerWidth, sheepsID[0])

    buildSheepCriticModel = BuildCriticModel(sheepObsDim, actionDim)
    criticWriterSheep, criticModelSheep = buildSheepCriticModel(layerWidth, sheepsID[0])

    trainCriticBySASRQSheep = TrainCriticBySASRQ(learningRateCritic, gamma, criticWriterSheep)
    trainCriticSheep = TrainCritic(actByPolicyTarget, evaluateCriticTarget, trainCriticBySASRQSheep)

    trainActorFromGradientsSheep = TrainActorFromQVal(learningRateActor, actorWriterSheep)
    trainActorOneStepSheep = TrainActorOneStep(actByPolicyTrain, trainActorFromGradientsSheep, evaluateCriticTrain)
    trainActorSheep = TrainActor(trainActorOneStepSheep)

    updateParameters = UpdateParameters(paramUpdateInterval, tau)

    modelList = [actorModelSheep, criticModelSheep]
    actorModelSheep, criticModelSheep = resetTargetParamToTrainParam(modelList)
    trainSheepModels = TrainDDPGModels(updateParameters, trainActorSheep, trainCriticSheep, actorModelSheep, criticModelSheep)

# ------------------------------------
    act1AgentOneStepWithNoiseWolf = ActOneStepMADDPGWithNoise(actorModelWolf, actByPolicyTrain)
    act1AgentOneStepWithNoiseSheep = ActOneStepMADDPGWithNoise(actorModelSheep, actByPolicyTrain)

    sample1AgentFromAgentMemory = SampleFromMemory(minibatchSize)
    learnFromBufferWolf = LearnFromBuffer(learningStartBufferSize, sample1AgentFromAgentMemory, trainWolfModels)
    learnFromBufferSheep = LearnFromBuffer(learningStartBufferSize, sample1AgentFromAgentMemory, trainSheepModels)

    sampleOneStep = SampleOneStep(transit, rewardFunc)

    agentsActOneStep = [act1AgentOneStepWithNoiseWolf, act1AgentOneStepWithNoiseSheep]
    # actOneStepWithNoise = lambda observation, runTime: [act1AgentOneStepWithNoiseWolf(observation[0][None]), [0, 0, 0, 0 ,0]]
    actOneStepWithNoise = lambda observation, runTime: [act1Agent(observation[i][None], runTime) for i, act1Agent in enumerate(agentsActOneStep)] # return all actions
    learnFromBuffer = [learnFromBufferWolf, learnFromBufferSheep]
    runDDPGTimeStep = RunMultiAgentTimeStep(actOneStepWithNoise, sampleOneStep, learnFromBuffer, observe = observe, multiagent= True)

    runEpisode = RunEpisode(reset, runDDPGTimeStep, maxTimeStep, isTerminal)
    ddpg = RunAlgorithm(runEpisode, maxEpisode)
    # replayBuffer = deque(maxlen=int(bufferSize))
    replayBuffer = getBuffer(bufferSize)
    meanRewardList, trajectory = ddpg(replayBuffer)

    plotResult = True
    if plotResult:
        plt.plot(list(range(len(meanRewardList))), meanRewardList)
        plt.show()

    trainedActorModelSheep, trainedCriticModelSheep = trainSheepModels.getTrainedModels()
    trainedActorModelWolf, trainedCriticModelWolf = trainWolfModels.getTrainedModels()


    with actorModelWolf.as_default():
        saveVariables(trainedActorModelWolf, os.path.join(dirName, '..', 'myDDPGWolfPolicy'))
    with actorModelSheep.as_default():
        saveVariables(trainedActorModelSheep, os.path.join(dirName, '..', 'myDDPGSheepPolicy'))



if __name__ == '__main__':
    main()

