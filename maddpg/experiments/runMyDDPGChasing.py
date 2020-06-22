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

from ddpg.src.newddpg_withCombinedTrain import BuildDDPGModels, TrainActorFromState, TrainCriticBySASR, TrainActor, \
    TrainCritic, reshapeBatchToGetSASR, TrainDDPGModelsWithBuffer, ActOneStepWithSoftMaxNoise, actByPolicyTrainNoisy
from RLframework.RLrun_MultiAgent import UpdateParameters, SampleOneStep, SampleFromMemory,\
    RunTimeStep, RunEpisode, RunAlgorithm, getBuffer, SaveModel, StartLearn
from functionTools.loadSaveModel import saveVariables
from environment.chasingEnv.multiAgentEnv import TransitMultiAgentChasing, ApplyActionForce, ApplyEnvironForce, \
    ResetMultiAgentChasing, ReshapeAction, RewardSheep, RewardWolf, Observe, GetCollisionForce, IntegrateState, \
    IsCollision, PunishForOutOfBound, getPosFromAgentState, getVelFromAgentState

maxEpisode = 60000
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

    debug = 0
    if debug:
        numWolves = 1
        numSheeps = 1
        numBlocks = 0
        saveAllmodels = True

    else:
        print(sys.argv)
        condition = json.loads(sys.argv[1])
        numWolves = int(condition['numWolves'])
        numSheeps = int(condition['numSheeps'])
        numBlocks = int(condition['numBlocks'])
        saveAllmodels = True
    print("ddpg: {} wolves, {} sheep, {} blocks, {} total episodes, save all models: {}".format(numWolves, numSheeps, numBlocks, maxEpisode, str(saveAllmodels)))


    numAgents = numWolves + numSheeps
    numEntities = numAgents + numBlocks
    wolvesID = list(range(numWolves))
    sheepsID = list(range(numWolves, numAgents))
    blocksID = list(range(numAgents, numEntities))

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
    obsShapeList = [initObsForParams[obsID].shape[0] for obsID in range(len(initObsForParams))] # in range(numAgents)

    worldDim = 2
    actionDim = worldDim * 2 + 1
    layerWidth = [64, 64]

    #------------ models ------------------------
    buildModels = [BuildDDPGModels(agentObsShape, actionDim) for agentObsShape in obsShapeList]
    allModels = [buildAgentModel(layerWidth, agentID) for agentID, buildAgentModel in enumerate(buildModels)]

    trainCriticBySASR = TrainCriticBySASR(learningRateCritic, gamma)
    trainCritic = TrainCritic(reshapeBatchToGetSASR, trainCriticBySASR)

    trainActorFromState = TrainActorFromState(learningRateActor)
    trainActor = TrainActor(reshapeBatchToGetSASR, trainActorFromState)

    paramUpdateInterval = 1 #
    updateParameters = UpdateParameters(paramUpdateInterval, tau)

    sampleFromMemory = SampleFromMemory(minibatchSize)

    learnInterval = 100
    startLearn = StartLearn(learningStartBufferSize, learnInterval)
    learnFromBuffer = TrainDDPGModelsWithBuffer(updateParameters, trainActor, trainCritic, sampleFromMemory, startLearn, allModels)

# ------------------------------------
    agentsActOneStep = [ActOneStepWithSoftMaxNoise(model, actByPolicyTrainNoisy) for model in allModels]
    actOneStepWithNoise = lambda observation, runTime: [act1Agent(observation[i][None], runTime) for i, act1Agent in enumerate(agentsActOneStep)] # return all actions

    sampleOneStep = SampleOneStep(transit, rewardFunc)

    runDDPGTimeStep = RunTimeStep(actOneStepWithNoise, sampleOneStep, learnFromBuffer, observe = observe)

    runEpisode = RunEpisode(reset, runDDPGTimeStep, maxTimeStep, isTerminal)

    getAgentModel = lambda agentId: lambda: learnFromBuffer.getTrainedModels()[agentId]
    getModelList = [getAgentModel(i) for i in range(numAgents)]
    modelSaveRate = 10000
    fileName = "ddpg{}wolves{}sheep{}blocks{}eps_agent".format(numWolves, numSheeps, numBlocks, maxEpisode)
    modelPath = os.path.join(dirName, '..', 'trainedModels', fileName)

    saveModels = [SaveModel(modelSaveRate, saveVariables, getTrainedModel, modelPath+ str(i), saveAllmodels) for i, getTrainedModel in enumerate(getModelList)]

    ddpg = RunAlgorithm(runEpisode, maxEpisode, saveModels, numAgents)
    replayBuffer = getBuffer(bufferSize)
    meanRewardList, trajectory = ddpg(replayBuffer)

if __name__ == '__main__':
    main()

