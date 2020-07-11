import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
import logging

logging.getLogger('tensorflow').setLevel(logging.ERROR)
import matplotlib.pyplot as plt
from environment.chasingEnv.multiAgentEnv import *
from environment.chasingEnv.multiAgentEnvWithIndividReward import RewardWolfIndividual

from functionTools.loadSaveModel import saveToPickle, restoreVariables
from functionTools.trajectory import SampleTrajectory
import pandas as pd
from maddpg.maddpgAlgor.trainer.myMADDPG import *
from collections import OrderedDict

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



def evaluateWolfSheepIterativeTrain(df):
    maxTimeStep = df.index.get_level_values('maxTimeStep')[0]
    sheepSpeedMultiplier = df.index.get_level_values('sheepSpeedMultiplier')[0]
    individualRewardWolf = df.index.get_level_values('individualRewardWolf')[0]

    numWolves = 3
    numSheeps = 1
    numBlocks = 2
    maxEpisode = 60000
    maxRunningStepsToSample = 50

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
    # sheepMaxSpeed = sheepMaxSpeedOriginal * sheepSpeedMultiplier
    sheepMaxSpeed = sheepMaxSpeedOriginal

    entityMaxSpeedList = [wolfMaxSpeed] * numWolves + [sheepMaxSpeed] * numSheeps + [blockMaxSpeed] * numBlocks
    entitiesMovableList = [True] * numAgents + [False] * numBlocks
    massList = [1.0] * numEntities

    isCollision = IsCollision(getPosFromAgentState)
    punishForOutOfBound = PunishForOutOfBound()
    rewardSheep = RewardSheep(wolvesID, sheepsID, entitiesSizeList, getPosFromAgentState, isCollision,
                              punishForOutOfBound)

    if individualRewardWolf:
        rewardWolf = RewardWolfIndividual(wolvesID, sheepsID, entitiesSizeList, isCollision)
    else:
        rewardWolf = RewardWolf(wolvesID, sheepsID, entitiesSizeList, isCollision)

    rewardFunc = lambda state, action, nextState: \
        list(rewardWolf(state, action, nextState)) + list(rewardSheep(state, action, nextState))


    reset = ResetMultiAgentChasing(numAgents, numBlocks)
    observeOneAgent = lambda agentID: Observe(agentID, wolvesID, sheepsID, blocksID, getPosFromAgentState,
                                              getVelFromAgentState)
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
    sampleTrajectory = SampleTrajectory(maxRunningStepsToSample, transit, isTerminal, rewardFunc, reset)

    initObsForParams = observe(reset())
    obsShape = [initObsForParams[obsID].shape[0] for obsID in range(len(initObsForParams))]

    worldDim = 2
    actionDim = worldDim * 2 + 1

    layerWidth = [128, 128]
    buildMADDPGModels = BuildMADDPGModels(actionDim, numAgents, obsShape)
    modelsList = [buildMADDPGModels(layerWidth, agentID) for agentID in range(numAgents)]

    dirName = os.path.dirname(__file__)
    individStr = 'individ' if individualRewardWolf else 'shared'
    fileName = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}{}_agent".format(
        numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, individStr)

    modelPaths = [os.path.join(dirName, '..', 'trainedModels', '3wolvesMaddpg_ExpEpsLengthAndSheepSpeed', fileName + str(i)) for i in
                  range(numAgents)]

    [restoreVariables(model, path) for model, path in zip(modelsList, modelPaths)]

    actOneStepOneModel = ActOneStep(actByPolicyTrainNoisy)
    policy = lambda allAgentsStates: [actOneStepOneModel(model, observe(allAgentsStates)) for model in modelsList]

    trajRewardList = []
    numTrajToSample = 500
    for i in range(numTrajToSample):
        traj = sampleTrajectory(policy)
        trajReward = calcTrajRewardWithIndividualWolfReward(traj, wolvesID) if individualRewardWolf else calcTrajRewardWithSharedWolfReward(traj)
        trajRewardList.append(trajReward)

    meanTrajReward = np.mean(trajRewardList)
    seTrajReward = np.std(trajRewardList) / np.sqrt(len(trajRewardList) - 1)

    return pd.Series({'mean': meanTrajReward, 'se': seTrajReward})


def main():

    independentVariables = OrderedDict()
    independentVariables['maxTimeStep'] = [25, 50, 75]
    independentVariables['sheepSpeedMultiplier'] = [1, 1.25, 1.5]
    independentVariables['individualRewardWolf'] = [0, 1]

    levelNames = list(independentVariables.keys())
    levelValues = list(independentVariables.values())
    levelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=levelIndex)
    resultDF = toSplitFrame.groupby(levelNames).apply(evaluateWolfSheepIterativeTrain)
    print(resultDF)
    resultLoc = os.path.join(dirName, 'evalMaddpgSheepSpeed_epsLength_individTrainWolfresult.pkl')
    saveToPickle(resultDF, resultLoc)

    figure = plt.figure(figsize=(16, 5))
    plotCounter = 1
    numRows = 1
    numColumns = len(independentVariables['sheepSpeedMultiplier'])
    for keyCol, outterSubDf in resultDF.groupby('sheepSpeedMultiplier'):
        outterSubDf.index = outterSubDf.index.droplevel('sheepSpeedMultiplier')
        axForDraw = figure.add_subplot(numRows, numColumns, plotCounter)
        for keyRow, innerSubDf in outterSubDf.groupby('maxTimeStep'):
            innerSubDf.index = innerSubDf.index.droplevel('maxTimeStep')
            plt.ylim([0, 150])

            innerSubDf.plot.line(ax = axForDraw, y='mean', yerr='se', label = keyRow, uplims=True, lolims=True, capsize=3)
            axForDraw.title.set_text('sheepSpeed' + str(keyCol))
            if plotCounter == 1:
                axForDraw.set_ylabel('Reward')
            axForDraw.set_xlabel('Shared     Individ')
            axForDraw.tick_params(axis='both', which='major')
            axForDraw.tick_params(axis='both', which='minor')
        plotCounter += 1
        axForDraw.set_aspect(0.01, adjustable='box')

        plt.legend(title='train steps')

    plt.suptitle('3v1 maddpg sampling 50 steps/eps')
    plt.show()


if __name__ == '__main__':
    main()
