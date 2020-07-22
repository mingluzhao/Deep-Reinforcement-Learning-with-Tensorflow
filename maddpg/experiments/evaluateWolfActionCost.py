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

dirName = os.path.dirname(__file__)

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
    costActionRatio = df.index.get_level_values('costActionRatio')[0]
    wolfIndividual = df.index.get_level_values('wolfIndividual')[0] #[sharedWithSharedCost, sharedWithIndividCost, individWithIndividCost]

    individualCost = False if 'SharedCost' in wolfIndividual else True
    individStr = 'shared' if 'sharedWith' in wolfIndividual else 'individ'

    numWolves = 3
    numSheeps = 1
    numBlocks = 2
    maxTimeStep = 75
    maxRunningStepsToSample = 75
    maxEpisode = 60000
    sheepSpeedMultiplier = 1.0

    print("maddpg: {} wolves, {} sheep, {} blocks, wolfIndividual: {}, individualCost: {}, costActionRatio: {}, train{} sample{}".format(
        numWolves, numSheeps, numBlocks, str(individStr), str(individualCost), costActionRatio, maxTimeStep, maxRunningStepsToSample))

    numAgents = numWolves + numSheeps
    numEntities = numAgents + numBlocks
    wolvesID = list(range(numWolves))
    sheepsID = list(range(numWolves, numAgents))
    blocksID = list(range(numAgents, numEntities))

    if wolfIndividual == 'sharedWithSharedCost':
        folderName = '3wolvesMaddpgWithActionCost_sharedWolvesHasSharedCost'
        trajFileName = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}WolfActCost{}{}".format(
            numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, costActionRatio, individStr)
        trajSavePath = os.path.join(dirName, '..', 'trajectories', folderName, trajFileName)
        trajList = loadFromPickle(trajSavePath)
        rewardList = [calcTrajRewardWithSharedWolfReward(traj) for traj in trajList]
        meanTrajReward = np.mean(rewardList)
        seTrajReward = np.std(rewardList) / np.sqrt(len(rewardList) - 1)
        print('meanTrajReward', meanTrajReward, 'se ', seTrajReward)

        return pd.Series({'mean': meanTrajReward, 'se': seTrajReward})


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
    rewardWolfShared = RewardWolf(wolvesID, sheepsID, entitiesSizeList, isCollision)
    getActionCost = GetActionCost(costActionRatio, reshapeAction, individualCost= individualCost)

    getWolvesAction = lambda action: [action[wolfID] for wolfID in wolvesID]
    rewardWolfIndividWithActionCost = lambda state, action, nextState: np.array(rewardWolfIndivid(state, action, nextState)) - np.array(getActionCost(getWolvesAction(action)))
    rewardWolfSharedWithActionCost = lambda state, action, nextState: np.array(rewardWolfShared(state, action, nextState)) - np.array(getActionCost(getWolvesAction(action)))

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
    sampleTrajectoryIndivid = SampleTrajectory(maxRunningStepsToSample, transit, isTerminal, rewardFuncIndividWolf, reset)
    sampleTrajectoryShared = SampleTrajectory(maxRunningStepsToSample, transit, isTerminal, rewardFuncSharedWolf, reset)
    sampleTrajectory = sampleTrajectoryIndivid if individStr == 'individ' else sampleTrajectoryShared

    initObsForParams = observe(reset())
    obsShape = [initObsForParams[obsID].shape[0] for obsID in range(len(initObsForParams))]
    worldDim = 2
    actionDim = worldDim * 2 + 1
    buildMADDPGModels = BuildMADDPGModels(actionDim, numAgents, obsShape)
    layerWidth = [128, 128]
    modelsList = [buildMADDPGModels(layerWidth, agentID) for agentID in range(numAgents)]

    fileName = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}WolfActCost{}{}_agent".format(
        numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, costActionRatio, individStr)
    folderName = '3wolvesMaddpgWithActionCost_sharedWolvesHasIndividCost'
    modelPaths = [os.path.join(dirName, '..', 'trainedModels', folderName, fileName + str(i)) for i in
                  range(numAgents)]


    [restoreVariables(model, path) for model, path in zip(modelsList, modelPaths)]

    actOneStepOneModel = ActOneStep(actByPolicyTrainNoisy)
    policy = lambda allAgentsStates: [actOneStepOneModel(model, observe(allAgentsStates)) for model in modelsList]

    rewardList = []
    trajList = []
    numTrajToSample = 500
    for i in range(numTrajToSample):
        traj = sampleTrajectory(policy)
        rew = calcTrajRewardWithSharedWolfReward(traj) if individStr == 'shared' else calcTrajRewardWithIndividualWolfReward(traj, wolvesID)
        rewardList.append(rew)
        trajList.append(list(traj))

    trajectoryDirectory = os.path.join(dirName, '..', 'trajectories', folderName)
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)
    trajFileName = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}WolfActCost{}{}".format(
        numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, costActionRatio, individStr)
    trajSavePath = os.path.join(trajectoryDirectory, trajFileName)
    # saveToPickle(trajList, trajSavePath)

    meanTrajReward = np.mean(rewardList)
    seTrajReward = np.std(rewardList) / np.sqrt(len(rewardList) - 1)
    print('meanTrajReward', meanTrajReward, 'se ', seTrajReward)

    return pd.Series({'mean': meanTrajReward, 'se': seTrajReward})



def main():
    independentVariables = OrderedDict()
    # independentVariables['wolfIndividual'] = ['sharedWithSharedCost', 'sharedWithIndividCost', 'individWithIndividCost']
    independentVariables['wolfIndividual'] = [ 'sharedWithIndividCost' ]
    independentVariables['costActionRatio'] = [0.0, 0.01, 0.05, 0.1, 0.2]

    levelNames = list(independentVariables.keys())
    levelValues = list(independentVariables.values())
    levelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=levelIndex)
    resultDF = toSplitFrame.groupby(levelNames).apply(evaluateWolfSheepTrain)
    print(resultDF)

    # resultPath = os.path.join(dirName, '..', 'evalResults')
    # resultLoc = os.path.join(resultPath, 'evalWolfActionCost_75steps_sample75_5levels.pkl')
    # saveToPickle(resultDF, resultLoc)
    #
    # figure = plt.figure(figsize=(6, 6))
    # plotCounter = 1
    # numRows = 1
    # numColumns = 1
    # axForDraw = figure.add_subplot(numRows, numColumns, plotCounter)
    #
    # for keyRow, innerSubDf in resultDF.groupby('wolfIndividual'):
    #     innerSubDf.index = innerSubDf.index.droplevel('wolfIndividual')
    #     plt.ylim([-5, 150])
    #     innerSubDf.plot.line(ax = axForDraw, y='mean', yerr='se', label = keyRow, uplims=True, lolims=True, capsize=3)
    #
    # axForDraw.set_ylabel('Mean Eps Reward')
    # axForDraw.set_xlabel('action cost/magnitude ratio')
    # # axForDraw.set_aspect(0.0007, adjustable='box')
    # plt.xticks(independentVariables['costActionRatio'])
    # plt.legend(title='Wolf type')
    # plt.suptitle('Wolves with action cost, train 75 steps, sampling 75 steps/eps')
    #
    #
    # plt.savefig(os.path.join(resultPath, 'evalWolfActionCost_75steps_sample75_5levels'))
    # plt.show()


if __name__ == '__main__':
    main()
    # resultDF = loadFromPickle(resultLoc)

    # figure = plt.figure(figsize=(10, 6))
    # plotCounter = 1
    # numRows = 1
    # numColumns = len(independentVariables['numWolves'])
    # for keyCol, outterSubDf in resultDF.groupby('numWolves'):
    #     outterSubDf.index = outterSubDf.index.droplevel('numWolves')
    #     axForDraw = figure.add_subplot(numRows, numColumns, plotCounter)
    #     for keyRow, innerSubDf in outterSubDf.groupby('wolfIndividual'):
    #         innerSubDf.index = innerSubDf.index.droplevel('wolfIndividual')
    #         plt.ylim([-5, 150])
    #         innerSubDf.plot.line(ax = axForDraw, y='mean', yerr='se', label = keyRow, uplims=True, lolims=True, capsize=3)
    #
    #     axForDraw.title.set_text('Number of wolves = ' + str(keyCol))
    #     if plotCounter == 1:
    #         axForDraw.set_ylabel('Mean Eps Reward')
    #     axForDraw.set_xlabel('action cost/magnitude ratio')
    #     plotCounter += 1
    #     axForDraw.set_aspect(0.0007, adjustable='box')
    #     plt.xticks(independentVariables['costActionRatio'])
    #     plt.legend(title='Wolf type')
