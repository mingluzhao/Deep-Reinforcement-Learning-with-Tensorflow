import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import numpy as np
import json

from maddpg.maddpgAlgor.trainer.myMADDPG import BuildMADDPGModels, TrainCritic, TrainActor, TrainCriticBySASR, \
    TrainActorFromSA, TrainMADDPGModelsWithBuffer, ActOneStep, actByPolicyTrainNoisy, actByPolicyTargetNoisyForNextState
from RLframework.RLrun import UpdateParameters, SampleOneStep, SampleFromMemory,\
    RunTimeStep, RunEpisode, RunAlgorithm, getBuffer, SaveModel, StartLearn
from functionTools.loadSaveModel import saveVariables
from environment.chasingEnv.multiAgentEnv import TransitMultiAgentChasing, ApplyActionForce, ApplyEnvironForce, \
    ResetMultiAgentChasing, ReshapeAction, RewardSheep, Observe, GetCollisionForce, IntegrateState, \
    IsCollision, PunishForOutOfBound, getPosFromAgentState, getVelFromAgentState, GetActionCost
from environment.chasingEnv.rewardWithKillProbSensitiveToDist import *

# fixed training parameters
maxEpisode = 60000
learningRateActor = 0.01#
learningRateCritic = 0.01#
gamma = 0.95 #
tau=0.01 #
bufferSize = 1e6#
minibatchSize = 32#


# 7.13 add action cost
# 7.29 constant sheep bonus = 30
# 8.9 add degree of individuality
# 8.23 changed sheep punishment to 10 to test consistency with source performance
# 8.26 changed constant sheep bonus = 10
# 9.3 add prob of kill, reward sensitive to distance

def main():
    debug = 0
    if debug:
        numWolves = 3
        sheepSpeedMultiplier = 1
        costActionRatio = 0.0
        rewardSensitivityToDistance = 0
        biteReward = 0.1

    else:
        print(sys.argv)
        condition = json.loads(sys.argv[1])
        numWolves = int(condition['numWolves'])
        sheepSpeedMultiplier = float(condition['sheepSpeedMultiplier'])
        costActionRatio = float(condition['costActionRatio'])
        rewardSensitivityToDistance = float(condition['rewardSensitivityToDistance'])
        biteReward = float(condition['biteReward'])
        # rewardSensitivityToDistance = float(condition['rewardSensitivityToDistance'])

    numSheeps = 1
    numBlocks = 2
    saveAllmodels = 1
    maxTimeStep = 75
    killReward = 10
    killProportion = 0.2

    print("maddpg: {} wolves, {} sheep, {} blocks, {} episodes with {} steps each eps, sheepSpeed: {}x, save all models: {}".
          format(numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, str(saveAllmodels)))

    numAgents = numWolves + numSheeps
    numEntities = numAgents + numBlocks
    wolvesID = list(range(numWolves))
    sheepsID = list(range(numWolves, numAgents))
    blocksID = list(range(numAgents, numEntities))

    wolfSize = 0.075
    sheepSize = 0.05
    blockSize = 0.2
    entitiesSizeList = [wolfSize] * numWolves + [sheepSize] * numSheeps + [blockSize] * numBlocks

    wolfMaxSpeed = 1.0
    blockMaxSpeed = None
    sheepMaxSpeedOriginal = 1.3
    sheepMaxSpeed = sheepMaxSpeedOriginal * sheepSpeedMultiplier

    entityMaxSpeedList = [wolfMaxSpeed] * numWolves + [sheepMaxSpeed] * numSheeps + [blockMaxSpeed] * numBlocks
    entitiesMovableList = [True] * numAgents + [False] * numBlocks
    massList = [1.0] * numEntities

    collisionReward = 10 # originalPaper = 10*3
    isCollision = IsCollision(getPosFromAgentState)
    punishForOutOfBound = PunishForOutOfBound()
    rewardSheep = RewardSheep(wolvesID, sheepsID, entitiesSizeList, getPosFromAgentState, isCollision,
                              punishForOutOfBound, collisionPunishment = collisionReward)

    collisionDist = wolfSize + sheepSize
    getAgentsPercentageOfRewards = GetAgentsPercentageOfRewards(rewardSensitivityToDistance, collisionDist)
    terminalCheck = TerminalCheck()
    getCollisionWolfReward = GetCollisionWolfReward(biteReward, killReward, killProportion, sampleFromDistribution, terminalCheck)
    getWolfSheepDistance = GetWolfSheepDistance(computeVectorNorm, getPosFromAgentState)
    rewardWolf = RewardWolvesWithKillProb(wolvesID, sheepsID, entitiesSizeList, isCollision, terminalCheck, getWolfSheepDistance,
                 getAgentsPercentageOfRewards, getCollisionWolfReward)

    reshapeAction = ReshapeAction()
    getActionCost = GetActionCost(costActionRatio, reshapeAction, individualCost=True)
    getWolvesAction = lambda action: [action[wolfID] for wolfID in wolvesID]
    rewardWolfWithActionCost = lambda state, action, nextState: np.array(rewardWolf(state, action, nextState)) - np.array(getActionCost(getWolvesAction(action)))

    rewardFunc = lambda state, action, nextState: \
        list(rewardWolfWithActionCost(state, action, nextState)) + list(rewardSheep(state, action, nextState))

    reset = ResetMultiAgentChasing(numAgents, numBlocks)
    observeOneAgent = lambda agentID: Observe(agentID, wolvesID, sheepsID, blocksID, getPosFromAgentState,
                                              getVelFromAgentState)
    observe = lambda state: [observeOneAgent(agentID)(state) for agentID in range(numAgents)]

    getCollisionForce = GetCollisionForce()
    applyActionForce = ApplyActionForce(wolvesID, sheepsID, entitiesMovableList)
    applyEnvironForce = ApplyEnvironForce(numEntities, entitiesMovableList, entitiesSizeList,
                                          getCollisionForce, getPosFromAgentState)
    integrateState = IntegrateState(numEntities, entitiesMovableList, massList,
                                    entityMaxSpeedList, getVelFromAgentState, getPosFromAgentState)
    transit = TransitMultiAgentChasing(numEntities, reshapeAction, applyActionForce, applyEnvironForce, integrateState)

    isTerminal = lambda state: terminalCheck.terminal
    initObsForParams = observe(reset())
    obsShape = [initObsForParams[obsID].shape[0] for obsID in range(len(initObsForParams))]

    worldDim = 2
    actionDim = worldDim * 2 + 1

    layerWidth = [128, 128]

#------------ models ------------------------

    buildMADDPGModels = BuildMADDPGModels(actionDim, numAgents, obsShape)
    modelsList = [buildMADDPGModels(layerWidth, agentID) for agentID in range(numAgents)]

    trainCriticBySASR = TrainCriticBySASR(actByPolicyTargetNoisyForNextState, learningRateCritic, gamma)
    trainCritic = TrainCritic(trainCriticBySASR)
    trainActorFromSA = TrainActorFromSA(learningRateActor)
    trainActor = TrainActor(trainActorFromSA)

    paramUpdateInterval = 1 #
    updateParameters = UpdateParameters(paramUpdateInterval, tau)
    sampleBatchFromMemory = SampleFromMemory(minibatchSize)

    learnInterval = 100
    learningStartBufferSize = minibatchSize * maxTimeStep
    startLearn = StartLearn(learningStartBufferSize, learnInterval)

    trainMADDPGModels = TrainMADDPGModelsWithBuffer(updateParameters, trainActor, trainCritic, sampleBatchFromMemory, startLearn, modelsList)

    actOneStepOneModel = ActOneStep(actByPolicyTrainNoisy)
    actOneStep = lambda allAgentsStates, runTime: [actOneStepOneModel(model, allAgentsStates) for model in modelsList]

    sampleOneStep = SampleOneStep(transit, rewardFunc)
    runTimeStep = RunTimeStep(actOneStep, sampleOneStep, trainMADDPGModels, observe = observe)

    runEpisode = RunEpisode(reset, runTimeStep, maxTimeStep, isTerminal)

    getAgentModel = lambda agentId: lambda: trainMADDPGModels.getTrainedModels()[agentId]
    getModelList = [getAgentModel(i) for i in range(numAgents)]
    modelSaveRate = 300
    fileName = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}WolfActCost{}sensitive{}biteReward{}_agent".format(
        numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, costActionRatio, rewardSensitivityToDistance, biteReward)

    folderName = 'maddpg_rewardSensitiveToDist'
    modelPath = os.path.join(dirName, '..', 'trainedModels', folderName, fileName)
    saveModels = [SaveModel(modelSaveRate, saveVariables, getTrainedModel, modelPath + str(i), saveAllmodels) for i, getTrainedModel in enumerate(getModelList)]

    maddpg = RunAlgorithm(runEpisode, maxEpisode, saveModels, numAgents)
    replayBuffer = getBuffer(bufferSize)
    meanRewardList = maddpg(replayBuffer)


if __name__ == '__main__':
    main()


