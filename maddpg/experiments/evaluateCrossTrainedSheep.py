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
from functionTools.loadSaveModel import saveToPickle, restoreVariables
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
    wolfIndividual = df.index.get_level_values('wolfIndividual')[0] # [shared, individ, mixShared, mixIndivid]
    sheepIndividual = df.index.get_level_values('sheepIndividual')[0] #[shared, individ, mixed]

    numWolves = 3
    numSheeps = 1
    numBlocks = 2
    maxTimeStep = 25
    sheepSpeedMultiplier = 1

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
    sampleTrajectory = sampleTrajectoryIndivid if wolfIndividual == 'individ' else sampleTrajectoryShared

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
    if sheepIndividual == 'mixed':
        maxEpisode = 120000
        fileName = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}individ_agent".format(
            numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier)
        sheepModelPath = os.path.join(dirName, '..', 'trainedModels', '3wolvesMaddpg_iterTrainSheep',
                                      fileName + str(sheepsID[0]))
    else:
        individStr = sheepIndividual
        fileName = "maddpg{}wolves{}sheep{}blocks60000episodes25stepSheepSpeed1.0{}_agent".format(
            numWolves, numSheeps, numBlocks, individStr)
        sheepModelPath = os.path.join(dirName, '..', 'trainedModels', '3wolvesMaddpg_ExpEpsLengthAndSheepSpeed',
                                      fileName + str(sheepsID[0]))

# --- wolf ----
    if wolfIndividual == 'mixShared':
        maxEpisode = 120000
        wolfIndividStr = 'shared'
        fileName = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}{}_agent".format(
            numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, wolfIndividStr)
        wolvesModelPaths = [os.path.join(dirName, '..', 'trainedModels', '3wolvesMaddpg_iterTrainSheep',
                                      fileName + str(i)) for i in wolvesID]
    elif wolfIndividual == 'mixIndivid':
        maxEpisode = 120000
        wolfIndividStr = 'individ'
        fileName = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}{}_agent".format(
            numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, wolfIndividStr)
        wolvesModelPaths = [os.path.join(dirName, '..', 'trainedModels', '3wolvesMaddpg_iterTrainSheep',
                                      fileName + str(i)) for i in wolvesID]
    else:
        wolfIndividStr = wolfIndividual
        fileName = "maddpg{}wolves{}sheep{}blocks60000episodes25stepSheepSpeed1.0{}_agent".format(numWolves, numSheeps, numBlocks, wolfIndividStr)
        wolvesModelPaths = [os.path.join(dirName, '..', 'trainedModels', '3wolvesMaddpg_ExpEpsLengthAndSheepSpeed',
                                             fileName + str(i)) for i in wolvesID]

    [restoreVariables(model, path) for model, path in zip(wolvesModel, wolvesModelPaths)]
    restoreVariables(sheepModel, sheepModelPath)

    actOneStepOneModel = ActOneStep(actByPolicyTrainNoisy)
    policy = lambda allAgentsStates: [actOneStepOneModel(model, observe(allAgentsStates)) for model in modelsList]

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
    independentVariables['wolfIndividual'] = ['shared', 'individ', 'mixShared', 'mixIndivid']
    independentVariables['sheepIndividual'] = ['shared', 'individ', 'mixed']

    levelNames = list(independentVariables.keys())
    levelValues = list(independentVariables.values())
    levelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=levelIndex)
    resultDF = toSplitFrame.groupby(levelNames).apply(evaluateWolfSheepTrain)

    resultLoc = os.path.join(dirName, 'result.pkl')

    saveToPickle(resultDF, resultLoc)

    fig, ax = plt.subplots()
    for key, grp in resultDF.groupby('wolfIndividual'):
        grp.index = grp.index.droplevel('wolfIndividual')
        ax = grp.plot(ax=ax, y='mean', yerr='se', uplims=True, label = key, lolims=True, capsize=5)

    plt.legend(title='Wolf type')
    plt.xlabel('Sheep type')
    plt.ylabel('Mean Episode Reward')
    plt.suptitle('3 wolf maddpg with shared/individual/mix sheep')
    plt.show()


if __name__ == '__main__':
    main()
