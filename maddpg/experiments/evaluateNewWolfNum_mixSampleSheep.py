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

def calcWolfTrajReward(traj, wolvesID):
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
        sheepSpeedMultiplier = df.index.get_level_values('sheepSpeedMultiplier')[0]# [1, 1.25]
        wolfIndividual = df.index.get_level_values('wolfIndividual')[0] #[shared, individ]
        costActionRatio = df.index.get_level_values('costActionRatio')[0]# [0.01, 0.05, 0.1]

        numSheeps = 1
        numBlocks = 2
        maxTimeStep = 75
        maxEpisode = 60000
        collisionReward = 10
        maxRunningStepsToSample = 75

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
        rewardSheep = RewardSheep(wolvesID, sheepsID, entitiesSizeList, getPosFromAgentState, isCollision, punishForOutOfBound, collisionPunishment = collisionReward)

        rewardWolf = RewardWolf(wolvesID, sheepsID, entitiesSizeList, isCollision, collisionReward, wolfIndividual)
        reshapeAction = ReshapeAction()
        getActionCost = GetActionCost(costActionRatio, reshapeAction, individualCost=True)
        getWolvesAction = lambda action: [action[wolfID] for wolfID in wolvesID]
        rewardWolfWithActionCost = lambda state, action, nextState: np.array(rewardWolf(state, action, nextState)) - np.array(getActionCost(getWolvesAction(action)))

        rewardFunc = lambda state, action, nextState: list(rewardWolfWithActionCost(state, action, nextState)) + list(rewardSheep(state, action, nextState))

        reset = ResetMultiAgentChasing(numAgents, numBlocks)
        observeOneAgent = lambda agentID: Observe(agentID, wolvesID, sheepsID, blocksID, getPosFromAgentState, getVelFromAgentState)
        observe = lambda state: [observeOneAgent(agentID)(state) for agentID in range(numAgents)]

        getCollisionForce = GetCollisionForce()
        applyActionForce = ApplyActionForce(wolvesID, sheepsID, entitiesMovableList)
        applyEnvironForce = ApplyEnvironForce(numEntities, entitiesMovableList, entitiesSizeList, getCollisionForce, getPosFromAgentState)
        integrateState = IntegrateState(numEntities, entitiesMovableList, massList, entityMaxSpeedList, getVelFromAgentState, getPosFromAgentState)
        transit = TransitMultiAgentChasing(numEntities, reshapeAction, applyActionForce, applyEnvironForce, integrateState)

        isTerminal = lambda state: False
        sampleTrajectory = SampleTrajectory(maxRunningStepsToSample, transit, isTerminal, rewardFunc, reset)

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
        fileName = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}WolfActCost{}individ{}_agent".format(
            numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, costActionRatio, wolfIndividual)

        folderName = 'maddpg_10reward_full'
        wolfModelPaths = [os.path.join(dirName, '..', 'trainedModels', folderName, fileName + str(i)) for i in wolvesID]
        [restoreVariables(model, path) for model, path in zip(wolvesModels, wolfModelPaths)]

        sheepPaths = self.getSheepModelPaths(numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep)

        rewardList = []
        trajList = []
        numTrajToSample = 500
        for i in range(numTrajToSample):
            sheepPath = self.getSampledSheepPath(sheepPaths)
            restoreVariables(sheepModel, sheepPath)
            actOneStepOneModel = ActOneStep(actByPolicyTrainNoisy)
            policy = lambda allAgentsStates: [actOneStepOneModel(model, observe(allAgentsStates)) for model in modelsList]
            traj = sampleTrajectory(policy)
            rew = calcWolfTrajReward(traj, wolvesID)
            rewardList.append(rew)
            trajList.append(list(traj))

        trajectoryDirectory = os.path.join(dirName, '..', 'trajectories', folderName)
        if not os.path.exists(trajectoryDirectory):
            os.makedirs(trajectoryDirectory)
        trajFileName = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}WolfActCost{}individ{}_mixTraj".format(
            numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, costActionRatio, wolfIndividual)
        trajSavePath = os.path.join(trajectoryDirectory, trajFileName)
        saveToPickle(trajList, trajSavePath)

        meanTrajReward = np.mean(rewardList)
        seTrajReward = np.std(rewardList) / np.sqrt(len(rewardList) - 1)
        print('meanTrajReward', meanTrajReward, 'se ', seTrajReward)

        return pd.Series({'mean': meanTrajReward, 'se': seTrajReward})


class GetSheepModelPaths:
    def __init__(self, sheepSpeedList, costActionRatioList, wolfTypeList):
        self.sheepSpeedList = sheepSpeedList
        self.wolfTypeList = wolfTypeList
        self.costActionRatioList = costActionRatioList

    def __call__(self, numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep):
        dirName = os.path.dirname(__file__)
        fileNameList = ["maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}WolfActCost{}individ{}_agent{}".format(
            numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, costActionRatio, wolfIndividual, numWolves)
            for sheepSpeedMultiplier in self.sheepSpeedList for wolfIndividual in self.wolfTypeList for costActionRatio in self.costActionRatioList]
        folderName = 'maddpg_10reward_full'
        sheepPaths = [os.path.join(dirName, '..', 'trainedModels', folderName, fileName) for fileName in fileNameList]

        return sheepPaths


def main():
    independentVariables = OrderedDict()
    independentVariables['wolfIndividual'] = [0.0, 1.0]
    independentVariables['numWolves'] = [2, 3, 4, 5, 6]
    independentVariables['sheepSpeedMultiplier'] = [1.0]
    independentVariables['costActionRatio'] = [0.0, 0.01, 0.02]

    getSheepModelPaths = GetSheepModelPaths(independentVariables['sheepSpeedMultiplier'], independentVariables['costActionRatio'], independentVariables['wolfIndividual'])
    evaluateWolfSheepTrain = EvaluateWolfSheepTrain(getSheepModelPaths)

    levelNames = list(independentVariables.keys())
    levelValues = list(independentVariables.values())
    levelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=levelIndex)
    # resultDF = toSplitFrame.groupby(levelNames).apply(evaluateWolfSheepTrain)

    resultPath = os.path.join(dirName, '..', 'evalResults')
    resultLoc = os.path.join(resultPath, 'eval10RewardFullResult.pkl')

    # saveToPickle(resultDF, resultLoc)

    resultDF = loadFromPickle(resultLoc)
    print(resultDF)
    figure = plt.figure(figsize=(7, 11))
    plotCounter = 1

    numRows = len(independentVariables['costActionRatio'])
    numColumns = len(independentVariables['sheepSpeedMultiplier'])

    for key, outmostSubDf in resultDF.groupby('costActionRatio'):
        outmostSubDf.index = outmostSubDf.index.droplevel('costActionRatio')
        for keyCol, outterSubDf in outmostSubDf.groupby('sheepSpeedMultiplier'):
            outterSubDf.index = outterSubDf.index.droplevel('sheepSpeedMultiplier')
            axForDraw = figure.add_subplot(numRows, numColumns, plotCounter)
            for keyRow, innerSubDf in outterSubDf.groupby('wolfIndividual'):
                innerSubDf.index = innerSubDf.index.droplevel('wolfIndividual')
                plt.ylim([0, 500])

                innerSubDf.plot.line(ax = axForDraw, y='mean', yerr='se', label = keyRow, uplims=True, lolims=True, capsize=3)
                if plotCounter <= numColumns:
                    axForDraw.title.set_text('sheepSpeed' + str(keyCol) + 'x')
                if plotCounter% numColumns == 1:
                    axForDraw.set_ylabel('actionCost/actionMagnitude = ' + str(key))
                axForDraw.set_xlabel('Number of Wolves')

            plotCounter += 1
            axForDraw.set_aspect(0.01, adjustable='box')
            plt.xticks(independentVariables['numWolves'])

            plt.legend(title='Wolf type')

    figure.text(x=0.03, y=0.5, s='Mean Episode Reward', ha='center', va='center', rotation=90)
    plt.suptitle('MADDPG Evaluate wolfType/ sheepSpeed/ actionCost')
    plt.savefig(os.path.join(resultPath, 'eval10RewardFullResult'))
    plt.show()




if __name__ == '__main__':
    main()
