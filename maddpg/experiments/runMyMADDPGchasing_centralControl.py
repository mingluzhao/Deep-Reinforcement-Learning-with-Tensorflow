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

from maddpg.maddpgAlgor.trainer.myMADDPG_centralControl import BuildMADDPGModels, BuildMADDPGModelsForCentralControl, TrainCritic, TrainActor, TrainCriticBySASR, \
    TrainActorFromSA, TrainMADDPGModelsWithBuffer, ActOneStep, actByPolicyTrainNoisy, actByPolicyTargetNoisyForNextState
from RLframework.RLrun_MultiAgent import UpdateParameters, SampleOneStep, SampleFromMemory,\
    RunTimeStep, RunEpisode, RunAlgorithm, getBuffer, SaveModel, StartLearn
from functionTools.loadSaveModel import saveVariables
from environment.chasingEnv.multiAgentEnv import TransitMultiAgentChasing, ApplyActionForce, ApplyEnvironForce, \
    ResetMultiAgentChasing, ReshapeAction, RewardSheep, RewardWolf, Observe, GetCollisionForce, IntegrateState, \
    IsCollision, PunishForOutOfBound, getPosFromAgentState, getVelFromAgentState
from environment.chasingEnv.multiAgentEnvWithIndividReward import RewardWolfIndividual

# fixed training parameters
maxEpisode = 60000
learningRateActor = 0.01#
learningRateCritic = 0.01#
gamma = 0.95 #
tau=0.01 #
bufferSize = 1e6#
minibatchSize = 1024#


# arguments: numWolves numSheeps numBlocks saveAllmodels = True or False

def main():
    debug = 1
    if debug:
        numWolves = 3
        numSheeps = 1
        numBlocks = 2
        saveAllmodels = False
        maxTimeStep = 25
        sheepSpeedMultiplier = 1
        individualRewardWolf = int(False)

    else:
        print(sys.argv)
        condition = json.loads(sys.argv[1])
        numWolves = int(condition['numWolves'])
        numSheeps = int(condition['numSheeps'])
        numBlocks = int(condition['numBlocks'])

        maxTimeStep = int(condition['maxTimeStep'])
        sheepSpeedMultiplier = float(condition['sheepSpeedMultiplier'])
        individualRewardWolf = int(condition['individualRewardWolf'])

        saveAllmodels = False

    print("maddpg central control: {} wolves, {} sheep, {} blocks, {} episodes with {} steps each eps, sheepSpeed: {}x, wolfIndividualReward: {}, save all models: {}".
          format(numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, individualRewardWolf, str(saveAllmodels)))


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

    isCollision = IsCollision(getPosFromAgentState)
    punishForOutOfBound = PunishForOutOfBound()
    rewardSheep = RewardSheep(wolvesID, sheepsID, entitiesSizeList, getPosFromAgentState, isCollision,
                              punishForOutOfBound)

    if individualRewardWolf:
        rewardWolf = RewardWolfIndividual(wolvesID, sheepsID, entitiesSizeList, isCollision)
    else:
        rewardWolf = RewardWolf(wolvesID, sheepsID, entitiesSizeList, isCollision)

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

    # ------------ central control ------------------------
    reshapeObsForCentralControl = lambda observation: [np.concatenate([observation[id] for id in wolvesID])] + [observation[id] for id in sheepsID]
    observeCentralControl = lambda state: reshapeObsForCentralControl(observe(state))
    initObsForParams = observeCentralControl(reset())
    obsShapeList = [initObsForParams[obsID].shape[0] for obsID in range(len(initObsForParams))] # in range(numAgents)

    worldDim = 2
    actionDim = worldDim * 2 + 1
    actionDimListCentralController = [numWolves* actionDim, actionDim]
    numModels = 1 + numSheeps

    controllerObsDim = obsShapeList[0]
    sheepObsDim = obsShapeList[1]

    #------------ models ------------------------
    layerWidth = [128, 128]
    buildMADDPGModels = BuildMADDPGModels(actionDim, actionDimListCentralController, obsShapeList)
    buildMADDPGCentralControl = BuildMADDPGModelsForCentralControl(actionDimListCentralController, numWolves, obsShapeList)
    modelsList = [buildMADDPGCentralControl(layerWidth, 0), buildMADDPGModels(layerWidth, 1)]

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

    reshapeReward = lambda reward: [np.sum([reward[id] for id in wolvesID])]+ [reward[id] for id in sheepsID]
    rewardCentralControl = lambda state, action, nextState: reshapeReward(rewardFunc(state, action, nextState))

    sampleOneStep = SampleOneStep(transit, rewardCentralControl)
    reshapeActionForCentralControl = lambda actionCentral: list(actionCentral[0].reshape(numWolves, actionDim)) + [actionCentral[id] for id in range(1, len(actionCentral))]
    sampleOneStepWithCentralControl = lambda state, action: sampleOneStep(state, reshapeActionForCentralControl(action))

    runDDPGTimeStep = RunTimeStep(actOneStep, sampleOneStepWithCentralControl, trainMADDPGModels, observe = observeCentralControl)

    runEpisode = RunEpisode(reset, runDDPGTimeStep, maxTimeStep, isTerminal)

    getAgentModel = lambda agentId: lambda: trainMADDPGModels.getTrainedModels()[agentId]
    getModelList = [getAgentModel(i) for i in range(numModels)]
    modelSaveRate = 1000
    individStr = 'individ' if individualRewardWolf else 'shared'
    fileName = "maddpgCentralControl{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}{}_agent".format(
        numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, individStr)

    modelPath = os.path.join(dirName, '..', 'trainedModels', 'centralControl3Wolves', fileName)
    saveModels = [SaveModel(modelSaveRate, saveVariables, getTrainedModel, modelPath+ str(i), saveAllmodels) for i, getTrainedModel in enumerate(getModelList)]

    maddpg = RunAlgorithm(runEpisode, maxEpisode, saveModels, numModels)
    replayBuffer = getBuffer(bufferSize)
    meanRewardList, trajectory = maddpg(replayBuffer)





if __name__ == '__main__':
    main()


