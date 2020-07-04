import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import matplotlib.pyplot as plt

from environment.chasingEnv.multiAgentEnv import *
from functionTools.loadSaveModel import saveToPickle, restoreVariables
from functionTools.trajectory import SampleTrajectory
from maddpg.maddpgAlgor.trainer.myMADDPG import *
from collections import OrderedDict
from environment.chasingEnv.multiAgentEnvWithIndividReward import RewardWolfIndividual

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



def main():
    numTrajToSample = 500
    maxRunningStepsToSample = 50

    wolfSize = 0.075
    sheepSize = 0.05
    blockSize = 0.2

    sheepMaxSpeed = 1.3
    wolfMaxSpeed = 1.0
    blockMaxSpeed = None

    numWolves = 3
    numBlocks = 2
    maxEpisode = 60000

    numSheepsList = [1, 2, 4, 8]
    meanRewardList = []
    seList = []

    for numSheeps in numSheepsList:
        numAgents = numWolves + numSheeps
        numEntities = numAgents + numBlocks
        wolvesID = list(range(numWolves))
        sheepsID = list(range(numWolves, numAgents))
        blocksID = list(range(numAgents, numEntities))

        entitiesSizeList = [wolfSize] * numWolves + [sheepSize] * numSheeps + [blockSize] * numBlocks
        entityMaxSpeedList = [wolfMaxSpeed] * numWolves + [sheepMaxSpeed] * numSheeps + [blockMaxSpeed] * numBlocks
        entitiesMovableList = [True] * numAgents + [False] * numBlocks
        massList = [1.0] * numEntities

        isCollision = IsCollision(getPosFromAgentState)
        rewardWolf = RewardWolf(wolvesID, sheepsID, entitiesSizeList, isCollision)
        punishForOutOfBound = PunishForOutOfBound()
        rewardSheep = RewardSheep(wolvesID, sheepsID, entitiesSizeList, getPosFromAgentState, isCollision,
                                  punishForOutOfBound)

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
        transit = TransitMultiAgentChasing(numEntities, reshapeAction, applyActionForce, applyEnvironForce,
                                           integrateState)

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
        fileName = "maddpg{}wolves{}sheep{}blocks{}eps_agent".format(numWolves, numSheeps, numBlocks, maxEpisode)
        modelPaths = [os.path.join(dirName, '..', 'trainedModels', '3wolvesMaddpg', fileName + str(i) + '60000eps') for i in range(numAgents)]

        [restoreVariables(model, path) for model, path in zip(modelsList, modelPaths)]

        actOneStepOneModel = ActOneStep(actByPolicyTrainNoisy)
        policy = lambda allAgentsStates: [actOneStepOneModel(model, observe(allAgentsStates)) for model in modelsList]

        trajRewardList = []
        for i in range(numTrajToSample):
            traj = sampleTrajectory(policy)
            trajReward = calcTrajRewardWithSharedWolfReward(traj)
            trajRewardList.append(trajReward)

        meanTrajReward = np.mean(trajRewardList)
        seTrajReward = np.std(trajRewardList) / np.sqrt(len(trajRewardList) - 1)
        print('meanTrajReward: ', meanTrajReward)
        meanRewardList.append(meanTrajReward)
        seList.append(seTrajReward)


# ---- individually rewarded wolves ------

    meanRewardListIndividWolf = []
    seListIndividWolf = []
    for numSheeps in numSheepsList:
        numAgents = numWolves + numSheeps
        numEntities = numAgents + numBlocks
        wolvesID = list(range(numWolves))
        sheepsID = list(range(numWolves, numAgents))
        blocksID = list(range(numAgents, numEntities))

        entitiesSizeList = [wolfSize] * numWolves + [sheepSize] * numSheeps + [blockSize] * numBlocks
        entityMaxSpeedList = [wolfMaxSpeed] * numWolves + [sheepMaxSpeed] * numSheeps + [blockMaxSpeed] * numBlocks
        entitiesMovableList = [True] * numAgents + [False] * numBlocks
        massList = [1.0] * numEntities

        isCollision = IsCollision(getPosFromAgentState)
        rewardWolf = RewardWolfIndividual(wolvesID, sheepsID, entitiesSizeList, isCollision)
        punishForOutOfBound = PunishForOutOfBound()
        rewardSheep = RewardSheep(wolvesID, sheepsID, entitiesSizeList, getPosFromAgentState, isCollision,
                                  punishForOutOfBound)

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
        transit = TransitMultiAgentChasing(numEntities, reshapeAction, applyActionForce, applyEnvironForce,
                                           integrateState)

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
        fileName = "maddpgIndividWolf{}wolves{}sheep{}blocks{}eps_agent".format(numWolves, numSheeps, numBlocks, maxEpisode)
        modelPaths = [os.path.join(dirName, '..', 'trainedModels', '3wolvesMaddpg', fileName + str(i) + '60000eps') for i in range(numAgents)]

        [restoreVariables(model, path) for model, path in zip(modelsList, modelPaths)]

        actOneStepOneModel = ActOneStep(actByPolicyTrainNoisy)
        policy = lambda allAgentsStates: [actOneStepOneModel(model, observe(allAgentsStates)) for model in modelsList]

        trajRewardList = []
        for i in range(numTrajToSample):
            traj = sampleTrajectory(policy)
            trajReward = calcTrajRewardWithIndividualWolfReward(traj, wolvesID)
            trajRewardList.append(trajReward)

        meanTrajReward = np.mean(trajRewardList)
        seTrajReward = np.std(trajRewardList) / np.sqrt(len(trajRewardList) - 1)

        print('meanTrajReward: ', meanTrajReward)

        meanRewardListIndividWolf.append(meanTrajReward)
        seListIndividWolf.append(seTrajReward)

    print('meanRewardList: ', meanRewardList)
    print('std error: ', seList)
    print('meanRewardListIndividWolf: ', meanRewardListIndividWolf)
    print('std error: ', seListIndividWolf)

    plotResult = True
    if plotResult:
        fig = plt.figure()

        plt.errorbar(numSheepsList, meanRewardList, seList)
        plt.errorbar(numSheepsList, meanRewardListIndividWolf, seListIndividWolf)

        fig.suptitle('MADDPG Performance With 3 Wolves')
        plt.xlabel('Number of Sheep')
        plt.ylabel('Mean Episode Reward')
        plt.xticks(numSheepsList)

        plt.legend(['Shared reward', 'Individual reward'])
        plt.show()

if __name__ == '__main__':
    main()
