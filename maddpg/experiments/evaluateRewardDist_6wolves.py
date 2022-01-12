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


maxRunningStepsToSample = 75

class EvaluateWolfSheepTrain:
    def __init__(self, getSheepModelPaths):
        self.getSheepModelPaths = getSheepModelPaths
        self.getSampledSheepPath = lambda sheepPaths: sheepPaths[random.randint(0, len(sheepPaths) - 1)]

    def __call__(self, df):
        sheepSpeedMultiplier = df.index.get_level_values('sheepSpeedMultiplier')[0]
        costActionRatio = df.index.get_level_values('costActionRatio')[0]
        rewardSensitivityToDistance = df.index.get_level_values('rewardSensitivityToDistance')[0]
        biteReward = df.index.get_level_values('biteReward')[0]

        numWolves = 6
        numSheeps = 1
        numBlocks = 2
        maxTimeStep = 75
        killReward = 10
        killProportion = 0.2
        maxEpisode = 60000

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

        collisionReward = 10  # originalPaper = 10*3
        isCollision = IsCollision(getPosFromAgentState)
        punishForOutOfBound = PunishForOutOfBound()
        rewardSheep = RewardSheep(wolvesID, sheepsID, entitiesSizeList, getPosFromAgentState, isCollision,
                                  punishForOutOfBound, collisionPunishment=collisionReward)

        collisionDist = wolfSize + sheepSize
        getAgentsPercentageOfRewards = GetAgentsPercentageOfRewards(rewardSensitivityToDistance, collisionDist)
        terminalCheck = TerminalCheck()
        biteRewardToUse = 0
        getCollisionWolfReward = GetCollisionWolfReward(biteRewardToUse, killReward, killProportion, sampleFromDistribution, terminalCheck)
        getWolfSheepDistance = GetWolfSheepDistance(computeVectorNorm, getPosFromAgentState)
        rewardWolf = RewardWolvesWithKillProb(wolvesID, sheepsID, entitiesSizeList, isCollision, terminalCheck,
                                              getWolfSheepDistance, getAgentsPercentageOfRewards, getCollisionWolfReward)

        rewardFunc = lambda state, action, nextState: \
            list(rewardWolf(state, action, nextState)) + list(rewardSheep(state, action, nextState))

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
        fileName = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}WolfActCost{}sensitive{}biteReward{}killPercent{}_agent".format(
            numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, costActionRatio,
            rewardSensitivityToDistance, biteReward, killProportion)

        folderName = 'maddpg_rewardSensitiveToDist'
        wolfModelPaths = [os.path.join(dirName, '..', 'trainedModels', folderName, fileName + str(i)) for i in wolvesID]
        [restoreVariables(model, path) for model, path in zip(wolvesModels, wolfModelPaths)]

        sheepPaths = self.getSheepModelPaths(numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, killProportion)

        killNumberList = []
        actionMagnSumList = []

        numTrajToSample = 300
        calcWolfDistance = CalcWolfDistance(reshapeAction)

        for i in range(numTrajToSample):
            sheepPath = self.getSampledSheepPath(sheepPaths)
            restoreVariables(sheepModel, sheepPath)
            actOneStepOneModel = ActOneStep(actByPolicyTrainNoisy)
            policy = lambda allAgentsStates: [actOneStepOneModel(model, observe(allAgentsStates)) for model in modelsList]

            traj = sampleTrajectory(policy)
            killNumber = calcWolfTrajBiteAmount(traj, wolvesID)
            actionMagnitude = calcWolfDistance(traj, wolvesID)

            killNumberList.append(killNumber)
            actionMagnSumList.append(actionMagnitude)

        meanTrajKill = np.mean(killNumberList)
        seTrajKill = np.std(killNumberList) / np.sqrt(len(killNumberList) - 1)
        print('meanTrajKill', meanTrajKill, 'seTrajKill ', seTrajKill)

        meanTrajAction = np.mean(actionMagnSumList)
        seTrajAction = np.std(actionMagnSumList) / np.sqrt(len(actionMagnSumList) - 1)

        return pd.Series({'meanKill': meanTrajKill, 'seKill': seTrajKill, 'meanTrajAction': meanTrajAction, 'seTrajAction': seTrajAction})


class GetSheepModelPaths:
    def __init__(self, sheepSpeedList, costActionRatioList, rewardSensitivityList, biteRewardList):
        self.sheepSpeedList = [0.5, 0.75, 1.0]
        self.costActionRatioList = [0.0, 0.01, 0.02, 0.03]
        self.rewardSensitivityList = [0.0, 1.0, 2.0, 10000.0]
        self.biteRewardList = [0.0, 0.05, 0.1, 1, 5, 10]

    def __call__(self, numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, killProportion):
        dirName = os.path.dirname(__file__)
        fileNameList = ["maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}WolfActCost{}sensitive{}biteReward{}killPercent{}_agent{}".format(
            numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, costActionRatio,
            rewardSensitivityToDistance, biteReward, killProportion, numWolves) for
            sheepSpeedMultiplier in self.sheepSpeedList for rewardSensitivityToDistance in self.rewardSensitivityList
            for biteReward in self.biteRewardList for costActionRatio in self.costActionRatioList]

        folderName = 'maddpg_rewardSensitiveToDist'
        sheepPaths = [os.path.join(dirName, '..', 'trainedModels', folderName, fileName) for fileName in fileNameList]

        return sheepPaths



class Modify:
    def __init__(self, resultDF):
        self.resultDF = resultDF

    def __call__(self, df):
        sheepSpeedMultiplier = df.index.get_level_values('sheepSpeedMultiplier')[0]
        costActionRatio = df.index.get_level_values('costActionRatio')[0]
        rewardSensitivityToDistance = df.index.get_level_values('rewardSensitivityToDistance')[0]
        biteReward = df.index.get_level_values('biteReward')[0]
        if biteReward == 3.0:
            biteReward = 10000.0

        meanTrajKill = self.resultDF.loc[sheepSpeedMultiplier, costActionRatio, rewardSensitivityToDistance]['meanKill']
        seKill = self.resultDF.loc[sheepSpeedMultiplier, costActionRatio, rewardSensitivityToDistance]['seKill']
        meanTrajAction = self.resultDF.loc[sheepSpeedMultiplier, costActionRatio, rewardSensitivityToDistance]['meanTrajAction']
        seTrajAction = self.resultDF.loc[sheepSpeedMultiplier, costActionRatio, rewardSensitivityToDistance]['seTrajAction']

        return pd.Series({'meanKill': meanTrajKill, 'seKill': seKill,
                          'meanTrajAction': meanTrajAction, 'seTrajAction': seTrajAction})


def main():
    independentVariables = OrderedDict()
    independentVariables['sheepSpeedMultiplier'] = [0.5, 0.75, 1.0]
    independentVariables['costActionRatio'] = [0.0, 0.01, 0.02, 0.03]
    independentVariables['rewardSensitivityToDistance'] = [0.0, 1.0, 2.0, 10000.0]
    independentVariables['biteReward'] = [0.0, 0.05, 0.1, 1, 5, 10] #[0.0, 0.05, 0.1, 0.15, 0.2, 0.25]

    biteRewardLabels = [0.0, 0.05, 0.1, 1, 5, 10]

    getSheepModelPaths = GetSheepModelPaths(independentVariables['sheepSpeedMultiplier'], independentVariables['costActionRatio'],
                                            independentVariables['rewardSensitivityToDistance'], independentVariables['biteReward'])
    evaluateWolfSheepTrain = EvaluateWolfSheepTrain(getSheepModelPaths)

    levelNames = list(independentVariables.keys())
    levelValues = list(independentVariables.values())
    levelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=levelIndex)
    # resultDF = toSplitFrame.groupby(levelNames).apply(evaluateWolfSheepTrain)

    resultPath = os.path.join(dirName, '..', 'evalResults')
    if not os.path.exists(resultPath):
        os.makedirs(resultPath)

    resultLoc = os.path.join(resultPath, 'evalReward6v1WithKillProbAndDistSensitive_allBiteReward.pkl')
    resultDF = loadFromPickle(resultLoc)


    # print(resultDF)
    figure = plt.figure(figsize=(35, 45))
    plotCounter = 1

    numRows = len(independentVariables['costActionRatio'])
    numColumns = len(independentVariables['sheepSpeedMultiplier'])

    for key, outmostSubDf in resultDF.groupby('costActionRatio'):
        outmostSubDf.index = outmostSubDf.index.droplevel('costActionRatio')
        for keyCol, outterSubDf in outmostSubDf.groupby('sheepSpeedMultiplier'):
            outterSubDf.index = outterSubDf.index.droplevel('sheepSpeedMultiplier')
            axForDraw = figure.add_subplot(numRows, numColumns, plotCounter)
            for keyRow, innerSubDf in outterSubDf.groupby('rewardSensitivityToDistance'):
                innerSubDf.index = innerSubDf.index.droplevel('rewardSensitivityToDistance')
                plt.ylim([0, 25])

                innerSubDf.plot.line(ax = axForDraw, y='meanKill', yerr='seKill', label = keyRow, uplims=True, lolims=True, capsize=3)
                if plotCounter <= numColumns:
                    axForDraw.title.set_text('sheepSpeed' + str(keyCol) + 'x')
                if plotCounter% numColumns == 1:
                    axForDraw.set_ylabel('actionCost = ' + str(key))
                axForDraw.set_xlabel('biteReward')

            plotCounter += 1
            plt.xticks(independentVariables['biteReward'])
            # axForDraw.set_aspect(0.002, adjustable='box')
            plt.legend(title='Selfish Index', title_fontsize = 8, prop={'size': 8})
            # plt.legend(title='reward sensitivity')

    figure.text(x=0.03, y=0.5, s='Mean Episode Kill', ha='center', va='center', rotation=90)
    plt.suptitle('MADDPG Evaluate wolfType/ sheepSpeed/ actionCost/ rewardDist')
    plt.savefig(os.path.join(resultPath, 'eval{}With{}evalReward6v1WithKillProbAndDistSensitive_killNum_allBiteReward.pdf'), dpi=600)

    # plt.savefig(os.path.join(resultPath, 'evalReward6v1WithKillProbAndDistSensitive_killNum_allBiteReward'))
    plt.show()
    plt.close()

    # --------
#
#     figure = plt.figure(figsize=(7, 11))
#     plotCounter = 1
#
#     numRows = len(independentVariables['rewardSensitivityToDistance'])#
#     numColumns = len(independentVariables['sheepSpeedMultiplier'])
#
#     for key, outmostSubDf in resultDF.groupby('rewardSensitivityToDistance'):#
#         outmostSubDf.index = outmostSubDf.index.droplevel('rewardSensitivityToDistance')#
#         for keyCol, outterSubDf in outmostSubDf.groupby('sheepSpeedMultiplier'):
#             outterSubDf.index = outterSubDf.index.droplevel('sheepSpeedMultiplier')
#             axForDraw = figure.add_subplot(numRows, numColumns, plotCounter)
#             for keyRow, innerSubDf in outterSubDf.groupby('costActionRatio'):
#                 innerSubDf.index = innerSubDf.index.droplevel('costActionRatio')
#                 plt.ylim([0, 25])
#
#                 innerSubDf.plot.line(ax = axForDraw, y='meanKill', yerr='seKill', label = keyRow, uplims=True, lolims=True, capsize=3)
#                 if plotCounter <= numColumns:
#                     axForDraw.title.set_text('sheepSpeed' + str(keyCol) + 'x')
#                 if plotCounter% numColumns == 1:
#                     axForDraw.set_ylabel('Wolf Selfish Level = ' + str(key))
#                 axForDraw.set_xlabel('biteReward')
#
#             plotCounter += 1
#             # plt.xticks(independentVariables['biteReward'])
#             plt.xticks(outterSubDf.reset_index()['biteReward'], independentVariables['biteReward'] * 4)
#
#             # axForDraw.set_aspect(0.002, adjustable='box')
#             plt.legend(title='Selfish Index', title_fontsize = 8, prop={'size': 8}) #
#
#     figure.text(x=0.03, y=0.5, s='Mean Episode Kill', ha='center', va='center', rotation=90)
#     plt.suptitle('MADDPG Evaluate wolfType/ sheepSpeed/ actionCost/ rewardDist')
#     plt.savefig(os.path.join(resultPath, 'evalReward6v1WithKillProbAndDistSensitive_killNum_allBiteReward'))
#     plt.show()
#     plt.close()
#
#
#
#     # --------
#     figure = plt.figure(figsize=(13, 11))
#     plotCounter = 1
#
#     numRows = len(independentVariables['sheepSpeedMultiplier'])
#     numColumns = len(independentVariables['rewardSensitivityToDistance'])
#
#     for key, outmostSubDf in resultDF.groupby('sheepSpeedMultiplier'):
#         outmostSubDf.index = outmostSubDf.index.droplevel('sheepSpeedMultiplier')
#         for keyCol, outterSubDf in outmostSubDf.groupby('rewardSensitivityToDistance'):
#             outterSubDf.index = outterSubDf.index.droplevel('rewardSensitivityToDistance')
#             axForDraw = figure.add_subplot(numRows, numColumns, plotCounter)
#             for keyRow, innerSubDf in outterSubDf.groupby('costActionRatio'):
#                 innerSubDf.index = innerSubDf.index.droplevel('costActionRatio')
#                 plt.ylim([0, 2000])
#
#                 innerSubDf.plot.line(ax = axForDraw, y='meanTrajAction', yerr='seTrajAction', label = keyRow, uplims=True, lolims=True, capsize=3)
#                 if plotCounter <= numColumns:
#                     axForDraw.title.set_text('Selfish Index = ' + str(keyCol))
#                 if plotCounter% numColumns == 1:
#                     axForDraw.set_ylabel('sheepSpeed' + str(key) + 'x')
#                 axForDraw.set_xlabel('biteReward')
#
#             plotCounter += 1
#             plt.xticks(independentVariables['biteReward'])
#
#             plt.legend(title='action cost')
#
#     figure.text(x=0.03, y=0.5, s='Mean Episode Moving Distance', ha='center', va='center', rotation=90)
#     plt.suptitle('MADDPG Evaluate wolfType/ sheepSpeed/ actionCost/ rewardDist')
#     plt.savefig(os.path.join(resultPath, 'evalReward6v1WithKillProbAndDistSensitive_MoveDistance_allBiteReward'))
#     plt.show()
#
###
    figure = plt.figure(figsize=(13, 11))
    plotCounter = 1

    numRows = len(independentVariables['sheepSpeedMultiplier'])
    numColumns = len(independentVariables['biteReward'])#

    for key, outmostSubDf in resultDF.groupby('sheepSpeedMultiplier'):
        outmostSubDf.index = outmostSubDf.index.droplevel('sheepSpeedMultiplier')
        for keyCol, outterSubDf in outmostSubDf.groupby('biteReward'):#
            outterSubDf.index = outterSubDf.index.droplevel('biteReward')#
            axForDraw = figure.add_subplot(numRows, numColumns, plotCounter)
            for keyRow, innerSubDf in outterSubDf.groupby('rewardSensitivityToDistance'):
                innerSubDf.index = innerSubDf.index.droplevel('rewardSensitivityToDistance')
                plt.ylim([0, 25])

                innerSubDf.plot.line(ax = axForDraw, y='meanKill', yerr='seKill', label = keyRow, uplims=True, lolims=True, capsize=3)
                if plotCounter <= numColumns:
                    axForDraw.title.set_text('biteReward' + str(keyCol))
                if plotCounter% numColumns == 1:
                    axForDraw.set_ylabel('sheepSpeed' + str(key) + 'x')
                axForDraw.set_xlabel('action cost')

            plotCounter += 1
            plt.xticks(independentVariables['costActionRatio'])
            # axForDraw.set_aspect(0.002, adjustable='box')
            plt.legend(title='Selfish Index')
            # plt.legend(title='reward sensitivity')

    figure.text(x=0.03, y=0.5, s='Mean Episode Kill', ha='center', va='center', rotation=90)
    plt.suptitle('MADDPG Evaluate wolfType/ sheepSpeed/ actionCost/ rewardDist')
    # plt.savefig(os.path.join(resultPath, 'evalReward6v1WithKillProbAndDistSensitive_killNum_regroup_allBiteReward'))
    plt.savefig(os.path.join(resultPath, 'evalReward6v1WithKillProbAndDistSensitive_killNum_allBiteReward.pdf'), dpi=600)

    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
