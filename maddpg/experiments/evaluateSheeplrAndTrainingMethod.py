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
    wolfType = df.index.get_level_values('wolfType')[0] # [shared, individ, mix1shared, mix1individ, mix5shared, mix5individ, mixRandomshared, mixRandomindivid]
    sheepType = df.index.get_level_values('sheepType')[0] # [shared, individ, mixed1, mix5, mixrandom]
    sheepLr = df.index.get_level_values('sheepLr')[0]

    print('wolfType {}, sheepType {}, sheepLR {}'.format(wolfType, sheepType, sheepLr))

    numWolves = 3
    numSheeps = 1
    numBlocks = 2
    maxTimeStep = 25
    sheepSpeedMultiplier = 1
    maxEpisode = 120000

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
    rewardSheep = RewardSheep(wolvesID, sheepsID, entitiesSizeList, getPosFromAgentState, isCollision, punishForOutOfBound)

    rewardWolfIndivid = RewardWolfIndividual(wolvesID, sheepsID, entitiesSizeList, isCollision)
    rewardWolfShared = RewardWolf(wolvesID, sheepsID, entitiesSizeList, isCollision)

    rewardFuncIndividWolf = lambda state, action, nextState: \
        list(rewardWolfIndivid(state, action, nextState)) + list(rewardSheep(state, action, nextState))
    rewardFuncSharedWolf = lambda state, action, nextState: \
        list(rewardWolfShared(state, action, nextState)) + list(rewardSheep(state, action, nextState))

    reset = ResetMultiAgentChasing(numAgents, numBlocks)
    observeOneAgent = lambda agentID: Observe(agentID, wolvesID, sheepsID, blocksID, getPosFromAgentState, getVelFromAgentState)
    observe = lambda state: [observeOneAgent(agentID)(state) for agentID in range(numAgents)]

    reshapeAction = ReshapeAction()
    getCollisionForce = GetCollisionForce()
    applyActionForce = ApplyActionForce(wolvesID, sheepsID, entitiesMovableList)
    applyEnvironForce = ApplyEnvironForce(numEntities, entitiesMovableList, entitiesSizeList,
                                          getCollisionForce, getPosFromAgentState)
    integrateState = IntegrateState(numEntities, entitiesMovableList, massList,
                                    entityMaxSpeedList, getVelFromAgentState, getPosFromAgentState)
    transit = TransitMultiAgentChasing(numEntities, reshapeAction, applyActionForce, applyEnvironForce, integrateState)

    isTerminal = lambda state: False
    maxRunningStepsToSample = 50
    sampleTrajectoryIndivid = SampleTrajectory(maxRunningStepsToSample, transit, isTerminal, rewardFuncIndividWolf, reset)
    sampleTrajectoryShared = SampleTrajectory(maxRunningStepsToSample, transit, isTerminal, rewardFuncSharedWolf, reset)

    initObsForParams = observe(reset())
    obsShape = [initObsForParams[obsID].shape[0] for obsID in range(len(initObsForParams))]
    worldDim = 2
    actionDim = worldDim * 2 + 1
    buildMADDPGModels = BuildMADDPGModels(actionDim, numAgents, obsShape)
    layerWidth = [128, 128]
    modelsList = [buildMADDPGModels(layerWidth, agentID) for agentID in range(numAgents)]

    sheepModel = modelsList[sheepsID[0]]
    wolvesModel = modelsList[:-1]

    dirName = os.path.dirname(__file__)

# --- sheep ----
    if sheepType == 'shared' or sheepType == 'individ':
        sheepIndividStr = sheepType
        fileName = "maddpg{}wolves{}sheep{}blocks60000episodes25stepSheepSpeed1.0{}_agent".format(
            numWolves, numSheeps, numBlocks, sheepIndividStr)
        sheepModelPath = os.path.join(dirName, '..', 'trainedModels', '3wolvesMaddpg_ExpEpsLengthAndSheepSpeed',
                                      fileName + str(sheepsID[0]))
    else:
        if sheepType == 'mixed1':
            sampleMethod = '1'
        elif sheepType == 'mixed5':
            sampleMethod = '5'
        else:
            sampleMethod = 'random'

        sheepIndividStr = 'individ'
        fileName = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}Lr{}SampleMethod{}{}_agent".format(
            numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, sheepLr,
            sampleMethod, sheepIndividStr)
        sheepModelPath = os.path.join(dirName, '..', 'trainedModels', 'IterTrainSheep_evalSheeplrAndSampleMethod',
                                 fileName + str(sheepsID[0]))


# --- wolf ----
    wolfIndividStr = 'shared' if 'shared' in wolfType else 'individ'
    if wolfType == 'shared' or wolfType == 'individ':
        fileName = "maddpg{}wolves{}sheep{}blocks60000episodes25stepSheepSpeed1.0{}_agent".format(numWolves, numSheeps, numBlocks, wolfIndividStr)
        wolvesModelPaths = [os.path.join(dirName, '..', 'trainedModels', '3wolvesMaddpg_ExpEpsLengthAndSheepSpeed',
                                             fileName + str(i)) for i in wolvesID]
    else:
        if '1' in wolfType:
            sampleMethod = '1'
        elif '5' in wolfType:
            sampleMethod = '5'
        else:
            sampleMethod = 'random'

        fileName = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}Lr{}SampleMethod{}{}_agent".format(
            numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, sheepLr,
            sampleMethod, wolfIndividStr)
        wolvesModelPaths = [os.path.join(dirName, '..', 'trainedModels', 'IterTrainSheep_evalSheeplrAndSampleMethod',
                                 fileName + str(i)) for i in wolvesID]


    [restoreVariables(model, path) for model, path in zip(wolvesModel, wolvesModelPaths)]
    restoreVariables(sheepModel, sheepModelPath)

    actOneStepOneModel = ActOneStep(actByPolicyTrainNoisy)
    policy = lambda allAgentsStates: [actOneStepOneModel(model, observe(allAgentsStates)) for model in modelsList]
    sampleTrajectory = sampleTrajectoryIndivid if wolfIndividStr == 'individ' else sampleTrajectoryShared

    trajList = []
    rewardList = []
    numTrajToSample = 500
    for i in range(numTrajToSample):
        traj = sampleTrajectory(policy)
        rew = calcTrajRewardWithSharedWolfReward(traj) if wolfIndividStr == 'shared' else calcTrajRewardWithIndividualWolfReward(traj, wolvesID)
        rewardList.append(rew)
        trajList.append(list(traj))

    meanTrajReward = np.mean(rewardList)
    seTrajReward = np.std(rewardList) / np.sqrt(len(rewardList) - 1)
    print('meanTrajRewardSharedWolf', meanTrajReward, 'se ', seTrajReward)

    return pd.Series({'mean': meanTrajReward, 'se': seTrajReward})


def main():
    independentVariables = OrderedDict()
    independentVariables['sheepLr'] = [0.01, 0.005, 0.001]
    independentVariables['wolfType'] = ["shared", "individ", "mix1shared", "mix1individ", "mix5shared", "mix5individ", "mixRandomshared", "mixRandomindivid"]
    independentVariables['sheepType'] = ["shared", "individ", "mixed1", "mix5", "mixrandom"]

    levelNames = list(independentVariables.keys())
    levelValues = list(independentVariables.values())
    levelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=levelIndex)
    # resultDF = toSplitFrame.groupby(levelNames).apply(evaluateWolfSheepTrain)
    # print(resultDF)

    resultPath = os.path.join(dirName, '..', 'evalResults')
    resultLoc = os.path.join(resultPath, 'evalSheepLrAndTrainingMethodResult.pkl')
    # saveToPickle(resultDF, resultLoc)

    resultDF = loadFromPickle(resultLoc)

    figure = plt.figure(figsize=(18, 6))
    plotCounter = 1
    numRows = 1
    numColumns = len(independentVariables['sheepLr'])
    for keyCol, outterSubDf in resultDF.groupby('sheepLr'):
        outterSubDf.index = outterSubDf.index.droplevel('sheepLr')
        axForDraw = figure.add_subplot(numRows, numColumns, plotCounter)
        for keyRow, innerSubDf in outterSubDf.groupby('wolfType'):
            innerSubDf.index = innerSubDf.index.droplevel('wolfType')
            plt.ylim([0, 150])
            innerSubDf.plot.line(ax = axForDraw, y='mean', yerr='se', label = keyRow, uplims=True, lolims=True, capsize=3)

        axForDraw.title.set_text('sheep learning rate: ' + str(keyCol))
        if plotCounter == 1:
            axForDraw.set_ylabel('Reward')
        axForDraw.set_xlabel('Sheep type')
        axForDraw.tick_params(axis='both', which='major')
        axForDraw.tick_params(axis='both', which='minor')
        plotCounter += 1
        axForDraw.set_aspect(0.025, adjustable='box')

        plt.legend(title='Wolf type', prop={'size': 8}, title_fontsize = 8)

    plt.suptitle('3v1 maddpg train 25steps, sampling 50 steps/eps')


    plt.savefig(os.path.join(resultPath, 'evalSheepLrAndTrainingMethodResult'))
    plt.show()


if __name__ == '__main__':
    main()
