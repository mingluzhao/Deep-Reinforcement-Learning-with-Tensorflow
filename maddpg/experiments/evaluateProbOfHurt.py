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
from environment.chasingEnv.rewardWithProbablisticKill import *

from maddpg.maddpgAlgor.trainer.myMADDPG import *
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
import random

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
        sensitiveZoneRadius = df.index.get_level_values('sensitiveZoneRadius')[0]
        oneWolfSelfHurtProb = df.index.get_level_values('oneWolfSelfHurtProb')[0]# [1, 1.25]
        wolfIndividual = df.index.get_level_values('wolfIndividual')[0] #[shared, individ]

        numWolves = 3
        numSheeps = 1
        numBlocks = 2
        maxTimeStep = 75
        maxRunningStepsToSample = 75
        maxEpisode = 60000
        sheepSpeedMultiplier = 1.0
        costActionRatio = 0.0
        saveTraj = True

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
        getHurtProbOfCatching = GetHurtProbOfCatchingByDeterministicZone(getPosFromAgentState, computeVectorNorm, sensitiveZoneRadius,
                                                                         oneWolfSelfHurtProb)
        hurtReward = -5
        collisionReward = 10
        individualRewardWolf = (wolfIndividual == 'individ')
        rewardWolf = RewardWolfWithHurtProb(wolvesID, sheepsID, entitiesSizeList, isCollision, getHurtProbOfCatching,
                                            sampleFromDistribution, individualRewardWolf, hurtReward, collisionReward)
        reshapeAction = ReshapeAction()
        getActionCost = GetActionCost(costActionRatio, reshapeAction, individualCost=True)
        getWolvesAction = lambda action: [action[wolfID] for wolfID in wolvesID]
        rewardWolfWithActionCost = lambda state, action, nextState: np.array(
            rewardWolf(state, action, nextState)) - np.array(getActionCost(getWolvesAction(action)))

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
        transit = TransitMultiAgentChasing(numEntities, reshapeAction, applyActionForce, applyEnvironForce,
                                           integrateState)

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
        fileName = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}WolfActCost{}HurtProb{}Radius{}{}_agent".format(
            numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, costActionRatio,
            oneWolfSelfHurtProb, sensitiveZoneRadius, wolfIndividual)

        folderName = '3wolvesMaddpgWithProbOfHurtBySheep'
        wolfModelPaths = [os.path.join(dirName, '..', 'trainedModels', folderName, fileName + str(i)) for i in wolvesID]

        [restoreVariables(model, path) for model, path in zip(wolvesModels, wolfModelPaths)]

        sheepPaths = self.getSheepModelPaths(numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, costActionRatio,
            oneWolfSelfHurtProb, sensitiveZoneRadius)

        rewardList = []
        trajList = []
        numTrajToSample = 10
        for i in range(numTrajToSample):
            # sheep
            sheepPath = self.getSampledSheepPath(sheepPaths)
            restoreVariables(sheepModel, sheepPath)

            actOneStepOneModel = ActOneStep(actByPolicyTrainNoisy)
            policy = lambda allAgentsStates: [actOneStepOneModel(model, observe(allAgentsStates)) for model in modelsList]

            traj = sampleTrajectory(policy)
            rew = calcTrajRewardWithSharedWolfReward(traj) if wolfIndividual == 'shared' else calcTrajRewardWithIndividualWolfReward(traj, wolvesID)
            rewardList.append(rew)
            trajList.append(list(traj))

        if saveTraj:
            trajectoryDirectory = os.path.join(dirName, '..', 'trajectories', folderName)
            if not os.path.exists(trajectoryDirectory):
                os.makedirs(trajectoryDirectory)
            trajFileName = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}WolfActCost{}HurtProb{}Radius{}{}_traj".format(
                numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, costActionRatio,
                oneWolfSelfHurtProb, sensitiveZoneRadius, wolfIndividual)
            trajSavePath = os.path.join(trajectoryDirectory, trajFileName)
            saveToPickle(trajList, trajSavePath)

        meanTrajReward = np.mean(rewardList)
        seTrajReward = np.std(rewardList) / np.sqrt(len(rewardList) - 1)
        print('meanTrajRewardSharedWolf', meanTrajReward, 'se ', seTrajReward)

        return pd.Series({'mean': meanTrajReward, 'se': seTrajReward})


class GetSheepModelPaths:
    def __init__(self, wolfTypeList):
        self.wolfTypeList = wolfTypeList

    def __call__(self, numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, costActionRatio,
            oneWolfSelfHurtProb, sensitiveZoneRadius):
        dirName = os.path.dirname(__file__)
        folderName = '3wolvesMaddpgWithProbOfHurtBySheep'
        fileNameList = ["maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}WolfActCost{}HurtProb{}Radius{}{}_agent{}".format(
            numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, costActionRatio,
            oneWolfSelfHurtProb, sensitiveZoneRadius, wolfIndividual, numWolves) for wolfIndividual in self.wolfTypeList]
        sheepPaths = [os.path.join(dirName, '..', 'trainedModels', folderName, fileName) for fileName in fileNameList]

        return sheepPaths

def main():
    independentVariables = OrderedDict()
    independentVariables['wolfIndividual'] = ['shared', 'individ']
    independentVariables['sensitiveZoneRadius'] = [0.25, 0.5, 0.75]
    independentVariables['oneWolfSelfHurtProb'] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    getSheepModelPaths = GetSheepModelPaths(independentVariables['wolfIndividual'])
    evaluateWolfSheepTrain = EvaluateWolfSheepTrain(getSheepModelPaths)

    levelNames = list(independentVariables.keys())
    levelValues = list(independentVariables.values())
    levelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=levelIndex)
    resultDF = toSplitFrame.groupby(levelNames).apply(evaluateWolfSheepTrain)

    resultPath = os.path.join(dirName, '..', 'evalResults')
    resultLoc = os.path.join(resultPath, 'evalProbOfHuntWithSharedRiskedWolvesVSIndivid.pkl')

    # saveToPickle(resultDF, resultLoc)
    #
    # resultDF = loadFromPickle(resultLoc)
    print(resultDF)
    #
    # figure = plt.figure(figsize=(10, 4))
    # plotCounter = 1
    # numRows = 1
    # numColumns = len(independentVariables['sensitiveZoneRadius'])
    # for keyCol, outterSubDf in resultDF.groupby('sensitiveZoneRadius'):
    #     outterSubDf.index = outterSubDf.index.droplevel('sensitiveZoneRadius')
    #     axForDraw = figure.add_subplot(numRows, numColumns, plotCounter)
    #     for keyRow, innerSubDf in outterSubDf.groupby('wolfIndividual'):
    #         innerSubDf.index = innerSubDf.index.droplevel('wolfIndividual')
    #         plt.ylim([0, 250])
    #         innerSubDf.plot.line(ax = axForDraw, y='mean', yerr='se', label = keyRow, uplims=True, lolims=True, capsize=3)
    #
    #     axForDraw.title.set_text('Sensitive Zone Radius = ' + str(keyCol))
    #     if plotCounter == 1:
    #         axForDraw.set_ylabel('Mean Eps Reward')
    #     axForDraw.set_xlabel('Fight Back Prob when 1 Wolf')
    #     plotCounter += 1
    #     axForDraw.set_aspect(0.005, adjustable='box')
    #
    #     plt.legend(title='Wolf type')
    #
    # plt.suptitle('3v1 wolf with 1.0x speed sheep, 0 action cost, train 75 sample 75, shared wolves have shared costs')
    #
    #
    # plt.savefig(os.path.join(resultPath, 'evalProbOfHuntWithSharedRiskedWolvesVSIndivid'))
    # plt.show()


if __name__ == '__main__':
    main()
