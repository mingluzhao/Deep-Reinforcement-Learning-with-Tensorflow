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

from ddpg.src.newddpg_followPaper import *
from RLframework.RLrun_MultiAgent import resetTargetParamToTrainParam, UpdateParameters, SampleOneStep, SampleFromMemory,\
    LearnFromBuffer, RunTimeStep, RunEpisode, RunAlgorithm, getBuffer, SaveModel
from functionTools.loadSaveModel import saveVariables
from environment.chasingEnv.multiAgentEnv import *
from visualize.visualizeMultiAgent import *


maxEpisode = 60000
# maxEpisode = 2000
maxTimeStep = 25

learningRateActor = 0.01#
learningRateCritic = 0.01#
gamma = 0.95 #
tau=0.01 #
bufferSize = 1e6#
minibatchSize = 1024#
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
    buildWolfModel = BuildDDPGModels(wolfObsDim, actionDim)
    writerWolf, modelWolf = buildWolfModel(layerWidth, wolvesID[0])

    trainWolfCriticBySASR = TrainCriticBySASR(learningRateCritic, gamma, writerWolf)
    trainWolfCritic = TrainCritic(reshapeBatchToGetSASR, trainWolfCriticBySASR)

    trainWolfActorFromState = TrainActorFromState(learningRateActor, writerWolf)
    trainWolfActor = TrainActor(reshapeBatchToGetSASR, trainWolfActorFromState)

    paramUpdateInterval = 1 #
    updateParameters = UpdateParameters(paramUpdateInterval, tau)

    trainWolfModels = TrainDDPGModels(updateParameters, trainWolfActor, trainWolfCritic, modelWolf)

#------------ sheep ------------------------
    buildSheepModel = BuildDDPGModels(sheepObsDim, actionDim)
    writerSheep, modelSheep = buildSheepModel(layerWidth, sheepsID[0])

    trainSheepCriticBySASR = TrainCriticBySASR(learningRateCritic, gamma, writerSheep)
    trainSheepCritic = TrainCritic(reshapeBatchToGetSASR, trainSheepCriticBySASR)

    trainSheepActorFromState = TrainActorFromState(learningRateActor, writerSheep)
    trainSheepActor = TrainActor(reshapeBatchToGetSASR, trainSheepActorFromState)


    paramUpdateInterval = 1 #
    updateParameters = UpdateParameters(paramUpdateInterval, tau)

    trainSheepModels = TrainDDPGModels(updateParameters, trainSheepActor, trainSheepCritic, modelSheep)

# ------------------------------------
    act1AgentOneStepWithNoiseWolf = ActOneStepMADDPGWithNoise(modelWolf, actByPolicyTrainNoisy)
    act1AgentOneStepWithNoiseSheep = ActOneStepMADDPGWithNoise(modelSheep, actByPolicyTrainNoisy)

    sample1AgentFromAgentMemory = SampleFromMemory(minibatchSize)
    learnInterval = 100
    learnFromBufferWolf = LearnFromBuffer(learningStartBufferSize, sample1AgentFromAgentMemory, trainWolfModels, learnInterval)
    learnFromBufferSheep = LearnFromBuffer(learningStartBufferSize, sample1AgentFromAgentMemory, trainSheepModels, learnInterval)

    sampleOneStep = SampleOneStep(transit, rewardFunc)

    agentsActOneStep = [act1AgentOneStepWithNoiseWolf, act1AgentOneStepWithNoiseSheep]
    actOneStepWithNoise = lambda observation, runTime: [act1Agent(observation[i][None], runTime) for i, act1Agent in enumerate(agentsActOneStep)] # return all actions
    learnFromBuffer = [learnFromBufferWolf, learnFromBufferSheep]
    runDDPGTimeStep = RunTimeStep(actOneStepWithNoise, sampleOneStep, learnFromBuffer, observe = observe)

    runEpisode = RunEpisode(reset, runDDPGTimeStep, maxTimeStep, isTerminal)

    getWolfTrainedModel = lambda: trainWolfModels.getTrainedModels()
    modelSaveRate = 5000
    wolfModelPath = os.path.join(dirName, '..', 'newDDPGWolfPolicy12')
    saveWolfModel = SaveModel(modelSaveRate, saveVariables, getWolfTrainedModel, wolfModelPath)

    getSheepTrainedModel = lambda: trainSheepModels.getTrainedModels()
    sheepModelPath = os.path.join(dirName, '..', 'newDDPGSheepPolicy12')
    saveSheepModel = SaveModel(modelSaveRate, saveVariables, getSheepTrainedModel, sheepModelPath)

    saveModels = [saveWolfModel, saveSheepModel]

    ddpg = RunAlgorithm(runEpisode, maxEpisode, saveModels, numAgents)
    replayBuffer = getBuffer(bufferSize)
    meanRewardList, trajectory = ddpg(replayBuffer)

if __name__ == '__main__':
    main()

