import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from environment.chasingEnv.multiAgentEnv import *
from functionTools.loadSaveModel import saveToPickle, restoreVariables
from functionTools.trajectory import SampleTrajectory
from visualize.visualizeMultiAgent import *
from maddpg.maddpgAlgor.trainer.myMADDPG_centralControl import *
from environment.chasingEnv.multiAgentEnvWithIndividReward import RewardWolfIndividual


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
    numWolves = 3
    numSheeps = 1
    numBlocks = 2

    sheepSpeedMultiplier = 1
    individualRewardWolf = False
    saveTraj = False
    visualizeTraj = True

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

    reshapeObsForCentralControl = lambda observation: [np.concatenate([observation[id] for id in wolvesID])] + [observation[id] for id in sheepsID]
    observeCentralControl = lambda state: reshapeObsForCentralControl(observe(state))
    initObsForParams = observeCentralControl(reset())
    obsShapeList = [initObsForParams[obsID].shape[0] for obsID in range(len(initObsForParams))] # in range(numAgents)
    worldDim = 2
    actionDim = worldDim * 2 + 1
    actionDimListCentralController = [numWolves* actionDim, actionDim]
    numModels = 1 + numSheeps

    reshapeReward = lambda reward: [np.sum([reward[id] for id in wolvesID])]+ [reward[id] for id in sheepsID]
    rewardCentralControl = lambda state, action, nextState: reshapeReward(rewardFunc(state, action, nextState))

    isTerminal = lambda state: False
    maxRunningSteps = 25
    sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, rewardCentralControl, reset)

    #------------ models ------------------------

    layerWidth = [128, 128]
    buildMADDPGModels = BuildMADDPGModels(actionDim, actionDimListCentralController, obsShapeList)
    buildMADDPGCentralControl = BuildMADDPGModelsForCentralControl(actionDimListCentralController, numWolves, obsShapeList)
    modelsList = [buildMADDPGCentralControl(layerWidth, 0), buildMADDPGModels(layerWidth, 1)]

    dirName = os.path.dirname(__file__)
    individStr = 'individ' if individualRewardWolf else 'shared'
    maxEpisode = 60000
    maxTimeStep = 25
    fileName = "maddpgCentralControl{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}{}_agent".format(
        numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, individStr)
    modelPaths = [os.path.join(dirName, '..', 'trainedModels', 'centralControl3Wolves', fileName + str(i)) for i in range(numModels)]
    [restoreVariables(model, path) for model, path in zip(modelsList, modelPaths)]

    actOneStepOneModel = ActOneStep(actByPolicyTrainNoisy)
    policyCentral = lambda allAgentsStates: [actOneStepOneModel(model, observeCentralControl(allAgentsStates)) for model in modelsList]
    reshapeActionForCentralControl = lambda actionCentral: list(actionCentral[0].reshape(numWolves, actionDim)) + [actionCentral[id] for id in range(1, len(actionCentral))]
    policy = lambda state: reshapeActionForCentralControl(policyCentral(state))

    trajList = []
    numTrajToSample = 50
    for i in range(numTrajToSample):
        traj = sampleTrajectory(policy)
        trajList.append(list(traj))

    # saveTraj
    if saveTraj:
        trajFileName = "maddpg{}wolves{}sheep{}blocks{}eps{}stepSheepSpeed{}{}Traj".format(numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, individStr)
        trajSavePath = os.path.join(dirName, '..', 'trajectory', trajFileName)
        saveToPickle(trajList, trajSavePath)


    # visualize
    if visualizeTraj:
        entitiesColorList = [wolfColor] * numWolves + [sheepColor] * numSheeps + [blockColor] * numBlocks
        render = Render(entitiesSizeList, entitiesColorList, numAgents, getPosFromAgentState)
        trajToRender = np.concatenate(trajList)
        render(trajToRender)


if __name__ == '__main__':
    main()
