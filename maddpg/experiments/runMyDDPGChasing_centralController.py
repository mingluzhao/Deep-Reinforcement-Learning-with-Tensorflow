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

from ddpg.src.newddpg_centralController import BuildDDPGModels, BuildDDPGModelsForCentralController, TrainActorFromState, TrainCriticBySASR, TrainActor, \
    TrainCritic, reshapeBatchToGetSASR, TrainDDPGModelsWithBuffer, ActOneStepWithSoftMaxNoise, actByPolicyTrainNoisy
from RLframework.RLrun_MultiAgent import UpdateParameters, SampleOneStep, SampleFromMemory,\
    RunTimeStep, RunEpisode, RunAlgorithm, getBuffer, SaveModel, StartLearn
from functionTools.loadSaveModel import saveVariables
from environment.chasingEnv.multiAgentEnv import TransitMultiAgentChasing, ApplyActionForce, ApplyEnvironForce, \
    ResetMultiAgentChasing, ReshapeAction, RewardSheep, RewardWolf, Observe, GetCollisionForce, IntegrateState, \
    IsCollision, PunishForOutOfBound, getPosFromAgentState, getVelFromAgentState
from visualize.visualizeMultiAgent import *

maxEpisode = 60000
maxTimeStep = 25
learningRateActor = 0.001#
learningRateCritic = 0.001#
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




class ResetWithStaticSheep:
    def __init__(self, numTotalAgents, numBlocks):
        self.positionDimension = 2
        self.numTotalAgents = numTotalAgents
        self.numBlocks = numBlocks

    def __call__(self):
        getAgentRandomPos = lambda: np.random.uniform(-1, +1, self.positionDimension)
        getAgentRandomVel = lambda: np.zeros(self.positionDimension)
        agentsState = [list(getAgentRandomPos()) + list(getAgentRandomVel()) for ID in range(2)] + [[0, 0, 0, 0]]

        getBlockRandomPos = lambda: np.random.uniform(-0.9, +0.9, self.positionDimension)
        getBlockSpeed = lambda: np.zeros(self.positionDimension)

        blocksState = [list(getBlockRandomPos()) + list(getBlockSpeed()) for blockID in range(self.numBlocks)]
        state = np.array(agentsState + blocksState)
        return state



def main():

    debug = 1
    if debug:
        numWolves = 2
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
    print("ddpgCentralControlle: {} wolves, {} sheep, {} blocks, {} total episodes, save all models: {}".format(numWolves, numSheeps, numBlocks, maxEpisode, str(saveAllmodels)))


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

    # reset = ResetMultiAgentChasing(numAgents, numBlocks)
    reset = ResetWithStaticSheep(numAgents, numBlocks)
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

    #------------ central control ------------------------

    reshapeObsForCentralControl = lambda observation: [np.concatenate([observation[id] for id in wolvesID])] + [observation[id] for id in sheepsID]
    observeCentralControl = lambda state: reshapeObsForCentralControl(observe(state))
    initObsForParams = observeCentralControl(reset())
    obsShapeList = [initObsForParams[obsID].shape[0] for obsID in range(len(initObsForParams))] # in range(numAgents)

    worldDim = 2
    actionDim = worldDim * 2 + 1
    layerWidth = [128, 128, 128]

    actionDimListCentralController = [numWolves* actionDim, actionDim]
    numModels = 1 + numSheeps

    controllerObsDim = obsShapeList[0]
    sheepObsDim = obsShapeList[1]

    #------------ models ------------------------

    buildControllerModel = BuildDDPGModelsForCentralController(controllerObsDim, actionDim, numWolves)
    buildAgentModels = [BuildDDPGModels(sheepObsDim, actionDim) for id in range(numSheeps)]

    buildModels = [buildControllerModel] + buildAgentModels
    allModels = [buildAgentModel(layerWidth, agentID) for agentID, buildAgentModel in enumerate(buildModels)]# 2 models

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
#     agentsActOneStep = [ActOneStepWithSoftMaxNoise(model, actByPolicyTrainNoisy) for model in allModels]
    agentsActOneStep = [ActOneStepWithSoftMaxNoise(allModels[0], actByPolicyTrainNoisy), lambda obs, runTime: [0, 0, 0, 0, 0]]

    actOneStepWithNoise = lambda observation, runTime: [act1Agent(observation[i][None], runTime) for i, act1Agent in enumerate(agentsActOneStep)] # return all actions

    reshapeReward = lambda reward: [np.sum([reward[id] for id in wolvesID])]+ [reward[id] for id in sheepsID]
    rewardCentralControl = lambda state, action, nextState: reshapeReward(rewardFunc(state, action, nextState))

    sampleOneStep = SampleOneStep(transit, rewardCentralControl)
    reshapeActionForCentralControl = lambda actionCentral: list(actionCentral[0].reshape(numWolves, actionDim)) + [actionCentral[id] for id in range(1, len(actionCentral))]
    sampleOneStepWithCentralControl = lambda state, action: sampleOneStep(state, reshapeActionForCentralControl(action))

    runDDPGTimeStep = RunTimeStep(actOneStepWithNoise, sampleOneStepWithCentralControl, learnFromBuffer, observe = observeCentralControl)

    runEpisode = RunEpisode(reset, runDDPGTimeStep, maxTimeStep, isTerminal)

    getAgentModel = lambda agentId: lambda: learnFromBuffer.getTrainedModels()[agentId]
    getModelList = [getAgentModel(i) for i in range(1 + numSheeps)]
    modelSaveRate = 10000
    fileName = "ddpgCentralControllerStaticSheep{}wolves{}sheep{}blocks{}eps_agent".format(numWolves, numSheeps, numBlocks, maxEpisode)
    modelPath = os.path.join(dirName, '..', 'trainedModels', fileName)

    saveModels = [SaveModel(modelSaveRate, saveVariables, getTrainedModel, modelPath+ str(i), saveAllmodels) for i, getTrainedModel in enumerate(getModelList)]

    ddpg = RunAlgorithm(runEpisode, maxEpisode, saveModels, numModels)
    replayBuffer = getBuffer(bufferSize)
    meanRewardList, trajectory = ddpg(replayBuffer)

    # visualize = True
    # if visualize:
    #     entitiesColorList = [wolfColor] * numWolves + [sheepColor] * numSheeps + [blockColor] * numBlocks
    #     render = Render(entitiesSizeList, entitiesColorList, numAgents, getPosFromAgentState)
    #     render(trajectory)

if __name__ == '__main__':
    main()
