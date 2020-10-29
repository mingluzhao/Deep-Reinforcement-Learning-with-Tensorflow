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
from functionTools.trajectory import SampleTrajectoryResetAtTerminal
from environment.chasingEnv.rewardWithKillProbSensitiveToDist import *

from maddpg.maddpgAlgor.trainer.myMADDPG import *
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
import random

wolfColor = np.array([0.85, 0.35, 0.35])
sheepColor = np.array([0.35, 0.85, 0.35])
blockColor = np.array([0.25, 0.25, 0.25])

def calcWolfTrajBiteAmount(traj, wolvesID, singleReward = 10):
    rewardIDinTraj = 2
    getWolfReward = lambda allAgentsReward: np.sum([allAgentsReward[wolfID] for wolfID in wolvesID])
    rewardList = [getWolfReward(timeStepInfo[rewardIDinTraj]) for timeStepInfo in traj]
    trajReward = np.sum(rewardList)
    biteNumber = trajReward/ singleReward

    return biteNumber


class CalcWolfDistance:
    def __init__(self, reshapeAction):
        self.reshapeAction = reshapeAction

    def __call__(self, traj, wolvesID):
        epsActionTot = 0

        for timeStepInfo in traj:
            action = timeStepInfo[1]
            wolvesActions = [self.reshapeAction(action[wolfID]) for wolfID in wolvesID]
            actionMagnitudeTot = np.sum([np.linalg.norm(np.array(agentAction), ord=2) for agentAction in wolvesActions])
            epsActionTot += actionMagnitudeTot

        return epsActionTot

class EvaluateWolfSheepTrain:
    def __init__(self, getSheepModelPaths):
        self.getSheepModelPaths = getSheepModelPaths
        self.getSampledSheepPath = lambda sheepPaths: sheepPaths[random.randint(0, len(sheepPaths) - 1)]

    def __call__(self, df):
        numWolves = df.index.get_level_values('numWolves')[0]
        trainingSequence = df.index.get_level_values('trainingSequence')[0]
        recoveredIndividualRewardWolf, continueTrainRewardWolf = trainingSequence

        numSheeps = 1
        numBlocks = 2
        maxTimeStep = 75
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

        collisionReward = 10
        isCollision = IsCollision(getPosFromAgentState)
        punishForOutOfBound = PunishForOutOfBound()
        rewardSheep = RewardSheep(wolvesID, sheepsID, entitiesSizeList, getPosFromAgentState, isCollision,
                                  punishForOutOfBound, collisionPunishment=collisionReward)

        rewardWolf = RewardWolf(wolvesID, sheepsID, entitiesSizeList, isCollision, collisionReward,
                                continueTrainRewardWolf)
        reshapeAction = ReshapeAction()
        getActionCost = GetActionCost(costActionRatio, reshapeAction, individualCost=True)
        getWolvesAction = lambda action: [action[wolfID] for wolfID in wolvesID]
        rewardWolfWithActionCost = lambda state, action, nextState: np.array(
            rewardWolf(state, action, nextState)) - np.array(getActionCost(getWolvesAction(action)))

        rewardFunc = lambda state, action, nextState: \
            list(rewardWolfWithActionCost(state, action, nextState)) + list(rewardSheep(state, action, nextState))

        reset = ResetMultiAgentChasing(numAgents, numBlocks)
        observeOneAgent = lambda agentID: Observe(agentID, wolvesID, sheepsID, blocksID, getPosFromAgentState, getVelFromAgentState)
        observe = lambda state: [observeOneAgent(agentID)(state) for agentID in range(numAgents)]

        getCollisionForce = GetCollisionForce()
        reshapeAction = ReshapeAction()
        applyActionForce = ApplyActionForce(wolvesID, sheepsID, entitiesMovableList)
        applyEnvironForce = ApplyEnvironForce(numEntities, entitiesMovableList, entitiesSizeList, getCollisionForce, getPosFromAgentState)
        integrateState = IntegrateState(numEntities, entitiesMovableList, massList, entityMaxSpeedList, getVelFromAgentState, getPosFromAgentState)
        transit = TransitMultiAgentChasing(numEntities, reshapeAction, applyActionForce, applyEnvironForce, integrateState)

        isTerminal = lambda state: False
        maxRunningStepsToSample = 75
        sampleTrajectory = SampleTrajectoryResetAtTerminal(maxRunningStepsToSample, transit, isTerminal, rewardFunc, reset)

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
        fileName = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}WolfActCost{}initIndivid{}laterIndivid{}_agent".format(
            numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, costActionRatio,
            recoveredIndividualRewardWolf, continueTrainRewardWolf)

        folderName = 'maddpg_testEquilib'
        wolfModelPaths = [os.path.join(dirName, '..', 'trainedModels', folderName, fileName + str(i)) for i in wolvesID]
        [restoreVariables(model, path) for model, path in zip(wolvesModels, wolfModelPaths)]

        sheepPaths = self.getSheepModelPaths(numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep)

        biteNumberList = []
        actionMagnSumList = []

        numTrajToSample = 500
        calcWolfDistance = CalcWolfDistance(reshapeAction)

        for i in range(numTrajToSample):
            sheepPath = self.getSampledSheepPath(sheepPaths)
            restoreVariables(sheepModel, sheepPath)
            actOneStepOneModel = ActOneStep(actByPolicyTrainNoisy)
            policy = lambda allAgentsStates: [actOneStepOneModel(model, observe(allAgentsStates)) for model in modelsList]

            traj = sampleTrajectory(policy)
            biteNumber = calcWolfTrajBiteAmount(traj, wolvesID)
            actionMagnitude = calcWolfDistance(traj, wolvesID)

            biteNumberList.append(biteNumber)
            actionMagnSumList.append(actionMagnitude)

        meanTrajBite = np.mean(biteNumberList)
        seTrajBite = np.std(biteNumberList) / np.sqrt(len(biteNumberList) - 1)
        print('meanTrajBite', meanTrajBite, 'seTrajBite ', seTrajBite)

        meanTrajAction = np.mean(actionMagnSumList)
        seTrajAction = np.std(actionMagnSumList) / np.sqrt(len(actionMagnSumList) - 1)

        return pd.Series({'meanBite': meanTrajBite, 'seBite': seTrajBite, 'meanTrajAction': meanTrajAction, 'seTrajAction': seTrajAction})


class GetSheepModelPaths:
    def __init__(self, sheepSpeedList, costActionRatioList, trainingSequenceList):
        self.sheepSpeedList = sheepSpeedList
        self.costActionRatioList = costActionRatioList
        self.trainingSequenceList = trainingSequenceList

    def __call__(self, numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep):
        dirName = os.path.dirname(__file__)
        fileNameList = ["maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}WolfActCost{}initIndivid{}laterIndivid{}_agent{}".format(
            numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, costActionRatio,
            recoveredIndividualRewardWolf, continueTrainRewardWolf, numWolves) for sheepSpeedMultiplier in self.sheepSpeedList
            for recoveredIndividualRewardWolf, continueTrainRewardWolf in self.trainingSequenceList
            for costActionRatio in self.costActionRatioList]

        folderName = 'maddpg_testEquilib'
        sheepPaths = [os.path.join(dirName, '..', 'trainedModels', folderName, fileName) for fileName in fileNameList]

        return sheepPaths


def main():
    independentVariables = OrderedDict()
    independentVariables['numWolves'] = [2, 3, 4, 5, 6]
    independentVariables['trainingSequence'] = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    sheepSpeedList = [1.0]
    costActionRatioList = [0.0]

    getSheepModelPaths = GetSheepModelPaths(sheepSpeedList, costActionRatioList,independentVariables['trainingSequence'])
    evaluateWolfSheepTrain = EvaluateWolfSheepTrain(getSheepModelPaths)

    levelNames = list(independentVariables.keys())
    levelValues = list(independentVariables.values())
    levelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=levelIndex)
    # resultDF = toSplitFrame.groupby(levelNames).apply(evaluateWolfSheepTrain)

    resultPath = os.path.join(dirName, '..', 'evalResults')
    if not os.path.exists(resultPath):
        os.makedirs(resultPath)

    resultLoc = os.path.join(resultPath, 'evalReward6v1WithKillProbAndDistSensitive.pkl')
    # saveToPickle(resultDF, resultLoc)

    resultDF = loadFromPickle(resultLoc)
    print(resultDF)

    figure = plt.figure(figsize=(7, 7))
    plotCounter = 1
    numRows = 1
    numColumns = 1
    axForDraw = figure.add_subplot(numRows, numColumns, plotCounter)

    for keyRow, innerSubDf in resultDF.groupby('trainingSequence'):
        innerSubDf.index = innerSubDf.index.droplevel('trainingSequence')
        plt.ylim([0, 25])
        innerSubDf.plot.line(ax = axForDraw, y='meanBite', yerr='seBite', label = keyRow, uplims=True, lolims=True, capsize=3)

    axForDraw.set_ylabel('Mean Eps Bite')
    axForDraw.set_xlabel('Number of wolves')
    plt.xticks(independentVariables['numWolves'])
    plt.legend(title='Training Sequence')
    plt.suptitle('Test Reward Structure Equilibrium, ActionCost=0, SheepSpeed=1.3wolfSpeed', fontsize=10)

    plt.savefig(os.path.join(resultPath, 'evalEquilibrium_reward'))
    plt.show()
    plt.close()

    figure = plt.figure(figsize=(7, 7))
    plotCounter = 1
    numRows = 1
    numColumns = 1
    axForDraw = figure.add_subplot(numRows, numColumns, plotCounter)

    for keyRow, innerSubDf in resultDF.groupby('trainingSequence'):
        innerSubDf.index = innerSubDf.index.droplevel('trainingSequence')
        plt.ylim([0, 2100])
        innerSubDf.plot.line(ax = axForDraw, y='meanTrajAction', yerr='seTrajAction', label = keyRow, uplims=True, lolims=True, capsize=3)

    axForDraw.set_ylabel('Mean Moving Distance')
    axForDraw.set_xlabel('Number of wolves')
    plt.xticks(independentVariables['numWolves'])
    plt.legend(title='Training Sequence')
    plt.suptitle('Test Reward Structure Equilibrium, ActionCost=0, SheepSpeed=1.3wolfSpeed', fontsize=10)

    plt.savefig(os.path.join(resultPath, 'evalEquilibrium_dist'))
    plt.show()



if __name__ == '__main__':
    main()
