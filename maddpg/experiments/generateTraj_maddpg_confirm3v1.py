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
from functionTools.loadSaveModel import saveToPickle, restoreVariables, loadFromPickle
from functionTools.trajectory import SampleTrajectory
from environment.chasingEnv.multiAgentEnvWithIndividReward import RewardWolfIndividual

from maddpg.maddpgAlgor.trainer.myMADDPG import *
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

wolfColor = np.array([0.85, 0.35, 0.35])
sheepColor = np.array([0.35, 0.85, 0.35])
blockColor = np.array([0.25, 0.25, 0.25])

def calcTrajRewardWithSharedWolfReward(traj):
    rewardIDinTraj = 2
    rewardList = [timeStepInfo[rewardIDinTraj][0] for timeStepInfo in traj]
    trajReward = np.sum(rewardList)
    return trajReward

def calcTrajRewardWithIndividualWolfReward(traj, wolvesID):
    rewardIDinTraj = 2
    getWolfReward = lambda allAgentsReward: np.sum([allAgentsReward[wolfID] for wolfID in wolvesID])
    rewardList = [getWolfReward(timeStepInfo[rewardIDinTraj]) for timeStepInfo in traj]
    trajReward = np.sum(rewardList)
    return trajReward

def evaluateWolfSheepTrain(df):
    numWolves = df.index.get_level_values('numWolves')[0]
    fileID = df.index.get_level_values('fileID')[0]
    wolfIndividual = df.index.get_level_values('wolfIndividual')[0] #[shared, individ]

    numSheeps = 1
    numBlocks = 2
    maxTimeStep = 25
    maxEpisode = 60000
    sheepSpeedMultiplier = 1.0
    costActionRatio = 0.0

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

    reshapeAction = ReshapeAction()
    rewardWolfIndivid = RewardWolfIndividual(wolvesID, sheepsID, entitiesSizeList, isCollision)
    getActionCostIndivid = GetActionCost(costActionRatio, reshapeAction, individualCost= True)
    rewardWolfShared = RewardWolf(wolvesID, sheepsID, entitiesSizeList, isCollision)
    getActionCostShared = GetActionCost(costActionRatio, reshapeAction, individualCost= False)

    getWolvesAction = lambda action: [action[wolfID] for wolfID in wolvesID]
    rewardWolfIndividWithActionCost = lambda state, action, nextState: np.array(rewardWolfIndivid(state, action, nextState)) - np.array(getActionCostIndivid(getWolvesAction(action)))
    rewardWolfSharedWithActionCost = lambda state, action, nextState: np.array(rewardWolfShared(state, action, nextState)) - np.array(getActionCostShared(getWolvesAction(action)))

    rewardFuncIndividWolf = lambda state, action, nextState: \
        list(rewardWolfIndividWithActionCost(state, action, nextState)) + list(rewardSheep(state, action, nextState))
    rewardFuncSharedWolf = lambda state, action, nextState: \
        list(rewardWolfSharedWithActionCost(state, action, nextState)) + list(rewardSheep(state, action, nextState))

    reset = ResetMultiAgentChasing(numAgents, numBlocks)
    observeOneAgent = lambda agentID: Observe(agentID, wolvesID, sheepsID, blocksID, getPosFromAgentState, getVelFromAgentState)
    observe = lambda state: [observeOneAgent(agentID)(state) for agentID in range(numAgents)]

    getCollisionForce = GetCollisionForce()
    applyActionForce = ApplyActionForce(wolvesID, sheepsID, entitiesMovableList)
    applyEnvironForce = ApplyEnvironForce(numEntities, entitiesMovableList, entitiesSizeList,
                                          getCollisionForce, getPosFromAgentState)
    integrateState = IntegrateState(numEntities, entitiesMovableList, massList,
                                    entityMaxSpeedList, getVelFromAgentState, getPosFromAgentState)
    transit = TransitMultiAgentChasing(numEntities, reshapeAction, applyActionForce, applyEnvironForce, integrateState)

    isTerminal = lambda state: False
    maxRunningStepsToSample = 75
    sampleTrajectoryIndivid = SampleTrajectory(maxRunningStepsToSample, transit, isTerminal, rewardFuncIndividWolf, reset)
    sampleTrajectoryShared = SampleTrajectory(maxRunningStepsToSample, transit, isTerminal, rewardFuncSharedWolf, reset)
    sampleTrajectory = sampleTrajectoryIndivid if wolfIndividual == 'individ' else sampleTrajectoryShared

    initObsForParams = observe(reset())
    obsShape = [initObsForParams[obsID].shape[0] for obsID in range(len(initObsForParams))]
    worldDim = 2
    actionDim = worldDim * 2 + 1
    buildMADDPGModels = BuildMADDPGModels(actionDim, numAgents, obsShape)
    layerWidth = [128, 128]
    modelsList = [buildMADDPGModels(layerWidth, agentID) for agentID in range(numAgents)]

    dirName = os.path.dirname(__file__)
    fileName = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}WolfActCost{}{}file{}_agent".format(
        numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, costActionRatio, wolfIndividual, fileID)
    modelPaths = [os.path.join(dirName, '..', 'trainedModels', 'runManyTimes', fileName + str(i)) for i in
                  range(numAgents)]


    [restoreVariables(model, path) for model, path in zip(modelsList, modelPaths)]

    actOneStepOneModel = ActOneStep(actByPolicyTrainNoisy)
    policy = lambda allAgentsStates: [actOneStepOneModel(model, observe(allAgentsStates)) for model in modelsList]

    rewardList = []
    numTrajToSample = 500
    trajList = []
    for i in range(numTrajToSample):
        traj = sampleTrajectory(policy)
        rew = calcTrajRewardWithSharedWolfReward(traj) if wolfIndividual == 'shared' else calcTrajRewardWithIndividualWolfReward(traj, wolvesID)
        rewardList.append(rew)
        trajList.append(list(traj))

    meanTrajReward = np.mean(rewardList)
    seTrajReward = np.std(rewardList) / np.sqrt(len(rewardList) - 1)
    print('meanTrajReward', meanTrajReward, 'se ', seTrajReward)

    trajSavePath = os.path.join(dirName, '..', 'trajectory', fileName)
    saveToPickle(trajList, trajSavePath)

    return pd.Series({'mean': meanTrajReward, 'se': seTrajReward})

def main():
    independentVariables = OrderedDict()
    independentVariables['wolfIndividual'] = ['shared', 'individ']
    independentVariables['fileID'] = [0, 1, 2, 3, 4]
    independentVariables['numWolves'] = [3, 4, 6]

    levelNames = list(independentVariables.keys())
    levelValues = list(independentVariables.values())
    levelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=levelIndex)
    resultDF = toSplitFrame.groupby(levelNames).apply(evaluateWolfSheepTrain)
    print(resultDF)

    resultPath = os.path.join(dirName, '..', 'evalResults')
    resultLoc = os.path.join(resultPath, 'confirmTrainResults.pkl')
    saveToPickle(resultDF, resultLoc)


if __name__ == '__main__':
    main()
