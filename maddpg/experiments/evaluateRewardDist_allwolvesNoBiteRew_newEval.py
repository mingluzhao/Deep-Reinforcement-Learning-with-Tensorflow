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

def calcWolfTrajKillAmount(traj, wolvesID, singleReward = 10):
    rewardIDinTraj = 2
    getWolfReward = lambda allAgentsReward: np.sum([allAgentsReward[wolfID] for wolfID in wolvesID])
    rewardList = [getWolfReward(timeStepInfo[rewardIDinTraj]) for timeStepInfo in traj]
    trajReward = np.sum(rewardList)
    killNumber = trajReward/ singleReward

    return killNumber

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

'''
'numWolvesLevels': 			        [2, 3, 4, 5, 6],
'sheepSpeedMultiplierLevels':       [0.5, .75, 1],
'costActionRatioList':         		[0, 0.01, 0.02, 0.03],
'rewardSensitivityToDistance':      [0, 1, 2, 10000],
'biteRewardLevels':            		[0]

'''

maxRunningStepsToSample = 75

class EvaluateWolfSheepTrain:
    def __init__(self, getSheepModelPaths):
        self.getSheepModelPaths = getSheepModelPaths
        self.getSampledSheepPath = lambda sheepPaths: sheepPaths[random.randint(0, len(sheepPaths) - 1)]

    def __call__(self, df):
        sheepSpeedMultiplier = df.index.get_level_values('sheepSpeedMultiplier')[0]
        costActionRatio = df.index.get_level_values('costActionRatio')[0]
        rewardSensitivityToDistance = df.index.get_level_values('rewardSensitivityToDistance')[0]
        numWolves = df.index.get_level_values('numWolves')[0]

        numSheeps = 1
        numBlocks = 2
        maxTimeStep = 75
        killReward = 10
        killProportion = 0.2
        biteReward = 0.0
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
        getCollisionWolfReward = GetCollisionWolfReward(biteReward, killReward, killProportion, sampleFromDistribution, terminalCheck)
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

        isTerminal = lambda state: terminalCheck.terminal
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
        meanAgentActionList = []

        numTrajToSample = 2#300
        calcWolfDistance = CalcWolfDistance(reshapeAction)

        for i in range(numTrajToSample):
            sheepPath = self.getSampledSheepPath(sheepPaths)
            restoreVariables(sheepModel, sheepPath)
            actOneStepOneModel = ActOneStep(actByPolicyTrainNoisy)
            policy = lambda allAgentsStates: [actOneStepOneModel(model, observe(allAgentsStates)) for model in modelsList]

            traj = sampleTrajectory(policy)
            killNumber = calcWolfTrajKillAmount(traj, wolvesID)
            actionMagnitude = calcWolfDistance(traj, wolvesID)

            killNumberList.append(killNumber)
            actionMagnSumList.append(actionMagnitude)
            meanAgentActionList.append(actionMagnitude/numWolves)

        meanTrajKill = np.mean(killNumberList)
        seTrajKill = np.std(killNumberList) / np.sqrt(len(killNumberList) - 1)
        print('meanTrajKill', meanTrajKill, 'seTrajKill ', seTrajKill)

        meanTrajAction = np.mean(actionMagnSumList)
        seTrajAction = np.std(actionMagnSumList) / np.sqrt(len(actionMagnSumList) - 1)

        meanAgentAction = np.mean(meanAgentActionList)
        seAgentAction = np.std(meanAgentActionList) / np.sqrt(len(meanAgentActionList) - 1)

        return pd.Series({'meanKill': meanTrajKill, 'seKill': seTrajKill,
                          'meanTrajAction': meanTrajAction, 'seTrajAction': seTrajAction,
                          'meanAgentAction': meanAgentAction, 'seAgentAction': seAgentAction})


class GetSheepModelPaths:
    def __init__(self, sheepSpeedList, costActionRatioList, rewardSensitivityList, biteRewardList):
        self.sheepSpeedList = [0.5, 0.625, 0.75, 0.875, 1.0]
        self.costActionRatioList = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
        self.rewardSensitivityList = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 10000.0]
        self.biteRewardList = [0.0]

    def __call__(self, numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, killProportion):
        dirName = os.path.dirname(__file__)
        fileNameList = ["maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}WolfActCost{}sensitive{}biteReward{}killPercent{}_agent{}".format(
            numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, costActionRatio,
            rewardSensitivityToDistance, biteReward, killProportion, numWolves)
            for sheepSpeedMultiplier in self.sheepSpeedList for rewardSensitivityToDistance in self.rewardSensitivityList
            for biteReward in self.biteRewardList for costActionRatio in self.costActionRatioList]

        folderName = 'maddpg_rewardSensitiveToDist'
        sheepPaths = [os.path.join(dirName, '..', 'trainedModels', folderName, fileName) for fileName in fileNameList]

        return sheepPaths



def main():
    sheepMaxSpeedOriginal = 1.3

    independentVariables = OrderedDict()
    independentVariables['numWolves'] = [2]
    independentVariables['sheepSpeedMultiplier'] = [0.5, 0.625, 0.75, 0.875, 1.0]
    independentVariables['costActionRatio'] = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
    independentVariables['rewardSensitivityToDistance'] = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 10000.0]

    getSheepModelPaths = GetSheepModelPaths(independentVariables['sheepSpeedMultiplier'], independentVariables['costActionRatio'],
                                            independentVariables['rewardSensitivityToDistance'], biteRewardList=[0.0])
    evaluateWolfSheepTrain = EvaluateWolfSheepTrain(getSheepModelPaths)

    levelNames = list(independentVariables.keys())
    levelValues = list(independentVariables.values())
    levelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=levelIndex)
    #resultDF = toSplitFrame.groupby(levelNames).apply(evaluateWolfSheepTrain)

    resultPath = os.path.join(dirName, '..', 'evalResults')
    if not os.path.exists(resultPath):
        os.makedirs(resultPath)

    # resultLoc21 = os.path.join(resultPath, 'evalRewardWithKillProbAndDistSensitiveNoBiteRewKillInfo_allConditions_2_1.pkl')
    # resultLoc22 = os.path.join(resultPath, 'evalRewardWithKillProbAndDistSensitiveNoBiteRewKillInfo_allConditions_2_2.pkl')
    # resultLoc31 = os.path.join(resultPath, 'evalRewardWithKillProbAndDistSensitiveNoBiteRewKillInfo_allConditions_3_1.pkl')
    # resultLoc32 = os.path.join(resultPath, 'evalRewardWithKillProbAndDistSensitiveNoBiteRewKillInfo_allConditions_3_2.pkl')
    # resultLoc41 = os.path.join(resultPath, 'evalRewardWithKillProbAndDistSensitiveNoBiteRewKillInfo_allConditions_4_1.pkl')
    # resultLoc42 = os.path.join(resultPath, 'evalRewardWithKillProbAndDistSensitiveNoBiteRewKillInfo_allConditions_4_2.pkl')
    #
    # df21 = loadFromPickle(resultLoc21)
    # df22 = loadFromPickle(resultLoc22)
    # df2 = df21.combine_first(df22)
    #
    # df31 = loadFromPickle(resultLoc31)
    # df32 = loadFromPickle(resultLoc32)
    # df3 = df31.combine_first(df32)
    #
    # df41 = loadFromPickle(resultLoc41)
    # df42 = loadFromPickle(resultLoc42)
    # df4 = df41.combine_first(df42)
    #
    # df23 = df2.combine_first(df3)
    # df234 = df4.combine_first(df23)
    #
    # resultLoc56 = os.path.join(resultPath, 'evalRewardWithKillProbAndDistSensitiveNoBiteRewKillInfo_allConditions_56.pkl')
    # df56 = loadFromPickle(resultLoc56)
    #
    # resultDF = df234.combine_first(df56)
    allResultLoc = os.path.join(resultPath, 'evalRewardWithKillProbAndDistSensitiveNoBiteRewKillInfo_allConditions.pkl')
    # saveToPickle(resultDF, allResultLoc)
    resultDF = loadFromPickle(allResultLoc)



    figure = plt.figure(figsize=(20, 20))
    plotCounter = 1

    numRows = len(independentVariables['rewardSensitivityToDistance'])#
    numColumns = len(independentVariables['sheepSpeedMultiplier'])

    for key, outmostSubDf in resultDF.groupby('rewardSensitivityToDistance'):#
        outmostSubDf.index = outmostSubDf.index.droplevel('rewardSensitivityToDistance')#
        for keyCol, outterSubDf in outmostSubDf.groupby('sheepSpeedMultiplier'):
            outterSubDf.index = outterSubDf.index.droplevel('sheepSpeedMultiplier')
            axForDraw = figure.add_subplot(numRows, numColumns, plotCounter)
            for keyRow, innerSubDf in outterSubDf.groupby('costActionRatio'):
                innerSubDf.index = innerSubDf.index.droplevel('costActionRatio')
                plt.ylim([0, 25])

                innerSubDf.plot.line(ax = axForDraw, y='meanKill', yerr='seKill', label = keyRow, uplims=True, lolims=True, capsize=3)
                if plotCounter <= numColumns:
                    axForDraw.title.set_text('Prey Speed = ' + str(np.round(keyCol* sheepMaxSpeedOriginal, 1)) + 'x')
                if plotCounter% numColumns == 1:
                    axForDraw.set_ylabel('Predators Selfish Level = ' + str(key))
                axForDraw.set_xlabel('Number of Predators')

            plotCounter += 1
            plt.xticks(independentVariables['numWolves'])
            plt.legend(title='action cost', title_fontsize = 8, prop={'size': 8})

    figure.text(x=0.03, y=0.5, s='Mean Episode Kill', ha='center', va='center', rotation=90)
    plt.suptitle('MADDPG Evaluate predatorSelfishness/ preySpeed/ actionCost')
    plt.savefig(os.path.join(resultPath, 'evalRewardWithKillProbAndDistSensitiveNoBiteRewKillInfo_killNum_allcond_regroup'))
    plt.show()
    plt.close()

    # # --------
    # figure = plt.figure(figsize=(9, 11))
    # plotCounter = 1
    #
    # numRows = len(independentVariables['rewardSensitivityToDistance'])
    # numColumns = len(independentVariables['sheepSpeedMultiplier'])
    #
    # for key, outmostSubDf in resultDF.groupby('rewardSensitivityToDistance'):#
    #     outmostSubDf.index = outmostSubDf.index.droplevel('rewardSensitivityToDistance')#
    #     for keyCol, outterSubDf in outmostSubDf.groupby('sheepSpeedMultiplier'):
    #         outterSubDf.index = outterSubDf.index.droplevel('sheepSpeedMultiplier')
    #         axForDraw = figure.add_subplot(numRows, numColumns, plotCounter)
    #         for keyRow, innerSubDf in outterSubDf.groupby('costActionRatio'):
    #             innerSubDf.index = innerSubDf.index.droplevel('costActionRatio')
    #             plt.ylim([0, 2500])
    #
    #             innerSubDf.plot.line(ax = axForDraw, y='meanTrajAction', yerr='seTrajAction', label = keyRow, uplims=True, lolims=True, capsize=3)
    #             if plotCounter <= numColumns:
    #                 axForDraw.title.set_text('Prey Speed = ' + str(np.round(keyCol* sheepMaxSpeedOriginal, 1)) + 'x')
    #             if plotCounter% numColumns == 1:
    #                 axForDraw.set_ylabel('Predator Selfish Level = ' + str(key))
    #             axForDraw.set_xlabel('Number of Predators')
    #
    #         plotCounter += 1
    #         plt.xticks(independentVariables['numWolves'])
    #
    #         plt.legend(title='action cost')
    #
    # figure.text(x=0.03, y=0.5, s='Mean Episode Moving Distance', ha='center', va='center', rotation=90)
    # plt.suptitle('MADDPG Evaluate predatorSelfishness/ preySpeed/ actionCost')
    # plt.savefig(os.path.join(resultPath, 'evalRewardWithKillProbAndDistSensitiveNoBiteRewKillInfo_allcond_MoveDistance'))
    # plt.show()
    # plt.close()
    #
    # # ----
    #
    # figure = plt.figure(figsize=(9, 11))
    # plotCounter = 1
    #
    # numRows = len(independentVariables['rewardSensitivityToDistance'])
    # numColumns = len(independentVariables['sheepSpeedMultiplier'])
    #
    # for key, outmostSubDf in resultDF.groupby('rewardSensitivityToDistance'):#
    #     outmostSubDf.index = outmostSubDf.index.droplevel('rewardSensitivityToDistance')#
    #     for keyCol, outterSubDf in outmostSubDf.groupby('sheepSpeedMultiplier'):
    #         outterSubDf.index = outterSubDf.index.droplevel('sheepSpeedMultiplier')
    #         axForDraw = figure.add_subplot(numRows, numColumns, plotCounter)
    #         for keyRow, innerSubDf in outterSubDf.groupby('costActionRatio'):
    #             innerSubDf.index = innerSubDf.index.droplevel('costActionRatio')
    #             plt.ylim([0, 400])
    #
    #             innerSubDf.plot.line(ax = axForDraw, y='meanAgentAction', yerr='seAgentAction', label = keyRow, uplims=True, lolims=True, capsize=3)
    #             if plotCounter <= numColumns:
    #                 axForDraw.title.set_text('Prey Speed = ' + str(np.round(keyCol* sheepMaxSpeedOriginal, 1)) + 'x')
    #             if plotCounter% numColumns == 1:
    #                 axForDraw.set_ylabel('Predator Selfish Level = ' + str(key))
    #             axForDraw.set_xlabel('Number of Predators')
    #
    #         plotCounter += 1
    #         plt.xticks(independentVariables['numWolves'])
    #
    #         plt.legend(title='action cost')
    #
    # figure.text(x=0.03, y=0.5, s='Mean Episode Moving Distance', ha='center', va='center', rotation=90)
    # plt.suptitle('MADDPG Evaluate predatorSelfishness/ preySpeed/ actionCost')
    # plt.savefig(os.path.join(resultPath, 'evalRewardWithKillProbAndDistSensitiveNoBiteRewKillInfo_allcond_SingleAgentMoveDistance'))
    # plt.show()

if __name__ == '__main__':
    main()
