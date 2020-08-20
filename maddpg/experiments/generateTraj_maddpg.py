import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
import logging

logging.getLogger('tensorflow').setLevel(logging.ERROR)

from environment.chasingEnv.multiAgentEnv import *
from functionTools.loadSaveModel import saveToPickle, restoreVariables
from functionTools.trajectory import SampleTrajectory
from visualize.visualizeMultiAgent import *

from maddpg.maddpgAlgor.trainer.myMADDPG import *
import json

wolfColor = np.array([0.85, 0.35, 0.35])
sheepColor = np.array([0.35, 0.85, 0.35])
blockColor = np.array([0.25, 0.25, 0.25])

maxEpisode = 60000
maxRunningStepsToSample = 75  # num of timesteps in one eps

def calcWolvesTrajReward(traj, wolvesID):
    rewardIDinTraj = 2
    getWolfReward = lambda allAgentsReward: np.sum([allAgentsReward[wolfID] for wolfID in wolvesID])
    rewardList = [getWolfReward(timeStepInfo[rewardIDinTraj]) for timeStepInfo in traj]
    print(rewardList)
    trajReward = np.sum(rewardList)
    return trajReward

def main():
    debug = 1
    if debug:
        numWolves = 4
        numSheeps = 1
        numBlocks = 2
        maxTimeStep = 75
        sheepSpeedMultiplier = 1.0
        individualRewardWolf = 0.0
        costActionRatio = 0.0

    else:
        print(sys.argv)
        condition = json.loads(sys.argv[1])
        numWolves = int(condition['numWolves'])
        numSheeps = int(condition['numSheeps'])
        numBlocks = int(condition['numBlocks'])

        maxTimeStep = int(condition['maxTimeStep'])
        sheepSpeedMultiplier = float(condition['sheepSpeedMultiplier'])
        individualRewardWolf = int(condition['individualRewardWolf'])
        costActionRatio = float(condition['costActionRatio'])

    print("maddpg: {} wolves, {} sheep, {} blocks, {} episodes with {} steps each eps, sheepSpeed: {}x, wolfIndividualReward: {}, costActionRatio: {}".
          format(numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, individualRewardWolf, costActionRatio))

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

    collisionReward = 30 # originalPaper = 10*3
    isCollision = IsCollision(getPosFromAgentState)
    punishForOutOfBound = PunishForOutOfBound()
    rewardSheep = RewardSheep(wolvesID, sheepsID, entitiesSizeList, getPosFromAgentState, isCollision,
                              punishForOutOfBound, collisionPunishment = collisionReward)

    rewardWolf = RewardWolf(wolvesID, sheepsID, entitiesSizeList, isCollision, collisionReward, individualRewardWolf)
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

    isTerminal = lambda state: False
    initObsForParams = observe(reset())
    obsShape = [initObsForParams[obsID].shape[0] for obsID in range(len(initObsForParams))]

    sampleTrajectory = SampleTrajectory(maxRunningStepsToSample, transit, isTerminal, rewardFunc, reset)

    worldDim = 2
    actionDim = worldDim * 2 + 1

    layerWidth = [128, 128]

    # ------------ model ------------------------
    buildMADDPGModels = BuildMADDPGModels(actionDim, numAgents, obsShape)
    modelsList = [buildMADDPGModels(layerWidth, agentID) for agentID in range(numAgents)]

    dirName = os.path.dirname(__file__)
    individStr = 'individ' if individualRewardWolf else 'shared'
    # fileName = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}WolfActCost{}individ{}_agent".format(
    #     numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, costActionRatio, individualRewardWolf)
    fileName = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}WolfActCost{}{}_agent".format(
        numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, costActionRatio, individStr)
    #
    folderName = 'maddpgWolfNum_WolfReward_ActionCost_SheepSpeed_correctTransit'
    modelPaths = [os.path.join(dirName, '..', 'trainedModels', folderName, fileName + str(i)) for i in range(numAgents)]

    # folderName = 'maddpg_fullConfirm0816'
    # modelPaths = [os.path.join(dirName, '..', 'trainedModels', folderName, fileName + str(i) + '60000eps') for i in range(numAgents)]

    [restoreVariables(model, path) for model, path in zip(modelsList, modelPaths)]

    actOneStepOneModel = ActOneStep(actByPolicyTrainNoisy)
    policy = lambda allAgentsStates: [actOneStepOneModel(model, observe(allAgentsStates)) for model in modelsList]

    rewardList = []
    numTrajToSample =3
    trajList = []
    for i in range(numTrajToSample):
        traj = sampleTrajectory(policy)
        rew = calcWolvesTrajReward(traj, wolvesID)
        rewardList.append(rew)
        trajList.append(list(traj))

    meanTrajReward = np.mean(rewardList)
    seTrajReward = np.std(rewardList) / np.sqrt(len(rewardList) - 1)
    print('meanTrajReward', meanTrajReward, 'se ', seTrajReward)

    # trajSavePath = os.path.join(dirName, '..', 'trajectory', fileName)
    # saveToPickle(trajList, trajSavePath)

    # trajectoryDirectory = os.path.join(dirName, '..', 'trajectories', folderName)
    # if not os.path.exists(trajectoryDirectory):
    #     os.makedirs(trajectoryDirectory)
    # trajFileName = "maddpg{}wolves{}sheep{}blocks{}eps{}stepSheepSpeed{}{}Traj".format(
    #     numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, individStr)
    # trajSavePath = os.path.join(trajectoryDirectory, trajFileName)
    # saveToPickle(trajList, trajSavePath)

    wolfColor = np.array([0.85, 0.35, 0.35])
    sheepColor = np.array([0.35, 0.85, 0.35])
    blockColor = np.array([0.25, 0.25, 0.25])
    entitiesColorList = [wolfColor] * numWolves + [sheepColor] * numSheeps + [blockColor] * numBlocks
    render = Render(entitiesSizeList, entitiesColorList, numAgents, getPosFromAgentState)
    trajToRender = np.concatenate(trajList)
    render(trajToRender)

if __name__ == '__main__':
    main()