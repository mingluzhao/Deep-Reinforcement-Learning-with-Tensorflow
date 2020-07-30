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
import random

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

class EvaluateWolfSheepTrain:
    def __init__(self, getSheepModelPaths):
        self.getSheepModelPaths = getSheepModelPaths
        self.getSampledSheepPath = lambda sheepPaths: sheepPaths[random.randint(0, len(sheepPaths) - 1)]

    def __call__(self, df):
        numWolves = df.index.get_level_values('numWolves')[0]
        maxEpisode = df.index.get_level_values('maxEpisode')[0]
        wolfIndividual = df.index.get_level_values('wolfIndividual')[0] #[shared, individ]

        numSheeps = 1
        numBlocks = 2
        maxTimeStep = 75
        sheepSpeedMultiplier = 1.0

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

        sheepModel = modelsList[sheepsID[0]]
        wolvesModels = modelsList[:-1]

        dirName = os.path.dirname(__file__)
        if maxEpisode == 120000:
            fileName = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}WolfActCost0.0{}_agent".format(
                numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, wolfIndividual)
            folderName = '3456wolvesRun120k'
        else:
            fileName = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}{}_agent".format(
                numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, wolfIndividual)
            folderName = 'evalOneSheep_2575steps_1to6wolves_11.25speed'

        wolfModelPaths = [os.path.join(dirName, '..', 'trainedModels', folderName, fileName + str(i)) for i in wolvesID]
        [restoreVariables(model, path) for model, path in zip(wolvesModels, wolfModelPaths)]

        sheepPaths = self.getSheepModelPaths(numWolves, numSheeps, numBlocks, maxTimeStep, sheepSpeedMultiplier)

        rewardList = []
        numTrajToSample = 1000
        for i in range(numTrajToSample):
            # sheep
            sheepPath = self.getSampledSheepPath(sheepPaths)
            restoreVariables(sheepModel, sheepPath)

            actOneStepOneModel = ActOneStep(actByPolicyTrainNoisy)
            policy = lambda allAgentsStates: [actOneStepOneModel(model, observe(allAgentsStates)) for model in modelsList]

            traj = sampleTrajectory(policy)
            rew = calcTrajRewardWithSharedWolfReward(traj) if wolfIndividual == 'shared' else calcTrajRewardWithIndividualWolfReward(traj, wolvesID)
            rewardList.append(rew)

        meanTrajReward = np.mean(rewardList)
        seTrajReward = np.std(rewardList) / np.sqrt(len(rewardList) - 1)
        print('meanTrajRewardSharedWolf', meanTrajReward, 'se ', seTrajReward)

        return pd.Series({'mean': meanTrajReward, 'se': seTrajReward})


class GetSheepModelPaths:
    def __init__(self, wolfTypeList):
        self.wolfTypeList = wolfTypeList

    def __call__(self, numWolves, numSheeps, numBlocks, maxTimeStep, sheepSpeedMultiplier):
        dirName = os.path.dirname(__file__)
        maxEpisode = 60000
        folderName = 'evalOneSheep_2575steps_1to6wolves_11.25speed'
        fileNameList60k = ["maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}{}_agent{}".format(
            numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, wolfIndividual, numWolves) for wolfIndividual in self.wolfTypeList]
        sheepPaths60k = [os.path.join(dirName, '..', 'trainedModels', folderName, fileName) for fileName in fileNameList60k]

        maxEpisode = 120000
        folderName = '3456wolvesRun120k'
        fileNameList120k = ["maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}WolfActCost0.0{}_agent{}".format(
            numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, wolfIndividual, numWolves) for wolfIndividual in self.wolfTypeList]
        sheepPaths120k = [os.path.join(dirName, '..', 'trainedModels', folderName, fileName) for fileName in fileNameList120k]

        sheepPaths = sheepPaths60k + sheepPaths120k

        return sheepPaths


def main():
    independentVariables = OrderedDict()
    independentVariables['wolfIndividual'] = ['shared', 'individ']
    independentVariables['numWolves'] = [3, 4, 5, 6]
    independentVariables['maxEpisode'] = [60000, 120000]

    getSheepModelPaths = GetSheepModelPaths(independentVariables['wolfIndividual'])
    evaluateWolfSheepTrain = EvaluateWolfSheepTrain(getSheepModelPaths)

    levelNames = list(independentVariables.keys())
    levelValues = list(independentVariables.values())
    levelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=levelIndex)
    # resultDF = toSplitFrame.groupby(levelNames).apply(evaluateWolfSheepTrain)

    resultPath = os.path.join(dirName, '..', 'evalResults')
    resultLoc = os.path.join(resultPath, 'evalTrain12k.pkl')

    # saveToPickle(resultDF, resultLoc)

    resultDF = loadFromPickle(resultLoc)
    print(resultDF)
    figure = plt.figure(figsize=(10, 5))
    plotCounter = 1
    numRows = 1
    numColumns = len(independentVariables['wolfIndividual'])
    for keyCol, outterSubDf in resultDF.groupby('wolfIndividual'):
        outterSubDf.index = outterSubDf.index.droplevel('wolfIndividual')
        axForDraw = figure.add_subplot(numRows, numColumns, plotCounter)
        for keyRow, innerSubDf in outterSubDf.groupby('maxEpisode'):
            innerSubDf.index = innerSubDf.index.droplevel('maxEpisode')
            plt.ylim([0, 500])
            innerSubDf.plot.line(ax = axForDraw, y='mean', yerr='se', label = keyRow, uplims=True, lolims=True, capsize=3)

        axForDraw.title.set_text(keyCol + ' wolves')
        if plotCounter == 1:
            axForDraw.set_ylabel('Mean Eps Reward')
        axForDraw.set_xlabel('Number of Wolves')
        plotCounter += 1
        axForDraw.set_aspect(0.007, adjustable='box')

        plt.legend(title='maxEpisode')

    plt.suptitle('eval train 120k')
    plt.savefig(os.path.join(resultPath, 'evalTrain12k'))
    plt.show()


if __name__ == '__main__':
    main()
