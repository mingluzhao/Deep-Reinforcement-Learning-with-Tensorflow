import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
import logging

logging.getLogger('tensorflow').setLevel(logging.ERROR)


from environment.chasingEnv.rewardWithKillProbSensitiveToDist import *

from environment.chasingEnv.multiAgentEnv import *
from functionTools.loadSaveModel import saveToPickle, restoreVariables, loadFromPickle
from maddpg.maddpgAlgor.trainer.myMADDPG import *
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from functionTools.trajectory import SampleTrajectoryResetAtTerminal

wolfColor = np.array([0.85, 0.35, 0.35])
sheepColor = np.array([0.35, 0.85, 0.35])
blockColor = np.array([0.25, 0.25, 0.25])

maxEpisode = 60000
maxRunningStepsToSample = 75  # num of timesteps in one eps

def randomSampleActions(numSamples):
    dim = 5
    samplesRaw = [np.random.uniform(0, +1, dim) for _ in range(numSamples)]
    samples = [sample/np.sum(sample) for sample in samplesRaw]
    return samples

def randomSampleStates(xlower, xupper, ylower, yupper, numSamplesEachEdge):
    xlist = np.linspace(xlower, xupper, numSamplesEachEdge)
    ylist = np.linspace(ylower, yupper, numSamplesEachEdge)
    stateList = [(x, y, -0.199, -0.9106) for x in xlist for y in ylist]
    return stateList

def main():
    # numWolves = 4
    # sheepSpeedMultiplier = 1.0
    # costActionRatio = 0.0
    # rewardSensitivityToDistance = 10000.0
    # biteReward = 0.0
    #
    # #
    # numSheeps = 1
    # numBlocks = 2
    # maxTimeStep = 75
    # killReward = 10
    # killProportion = 0.2
    #
    # numAgents = numWolves + numSheeps
    # numEntities = numAgents + numBlocks
    # wolvesID = list(range(numWolves))
    # sheepsID = list(range(numWolves, numAgents))
    # blocksID = list(range(numAgents, numEntities))
    #
    # wolfSize = 0.075
    # sheepSize = 0.05
    # blockSize = 0.2
    # entitiesSizeList = [wolfSize] * numWolves + [sheepSize] * numSheeps + [blockSize] * numBlocks
    #
    # wolfMaxSpeed = 1.0
    # blockMaxSpeed = None
    # sheepMaxSpeedOriginal = 1.3
    # sheepMaxSpeed = sheepMaxSpeedOriginal * sheepSpeedMultiplier
    #
    # entityMaxSpeedList = [wolfMaxSpeed] * numWolves + [sheepMaxSpeed] * numSheeps + [blockMaxSpeed] * numBlocks
    # entitiesMovableList = [True] * numAgents + [False] * numBlocks
    # massList = [1.0] * numEntities
    #
    # collisionReward = 10 # originalPaper = 10*3
    # isCollision = IsCollision(getPosFromAgentState)
    # punishForOutOfBound = PunishForOutOfBound()
    # rewardSheep = RewardSheep(wolvesID, sheepsID, entitiesSizeList, getPosFromAgentState, isCollision,
    #                           punishForOutOfBound, collisionPunishment = collisionReward)
    #
    # collisionDist = wolfSize + sheepSize
    # getAgentsPercentageOfRewards = GetAgentsPercentageOfRewards(rewardSensitivityToDistance, collisionDist)
    # terminalCheck = TerminalCheck()
    #
    # getCollisionWolfReward = GetCollisionWolfReward(biteReward, killReward, killProportion, sampleFromDistribution, terminalCheck)
    # getWolfSheepDistance = GetWolfSheepDistance(computeVectorNorm, getPosFromAgentState)
    # rewardWolf = RewardWolvesWithKillProb(wolvesID, sheepsID, entitiesSizeList, isCollision, terminalCheck, getWolfSheepDistance,
    #              getAgentsPercentageOfRewards, getCollisionWolfReward)
    #
    # reshapeAction = ReshapeAction()
    # getActionCost = GetActionCost(costActionRatio, reshapeAction, individualCost=True)
    # getWolvesAction = lambda action: [action[wolfID] for wolfID in wolvesID]
    # rewardWolfWithActionCost = lambda state, action, nextState: np.array(rewardWolf(state, action, nextState)) - np.array(getActionCost(getWolvesAction(action)))
    #
    # rewardFunc = lambda state, action, nextState: \
    #     list(rewardWolfWithActionCost(state, action, nextState)) + list(rewardSheep(state, action, nextState))
    #
    # reset = ResetMultiAgentChasing(numAgents, numBlocks)
    # observeOneAgent = lambda agentID: Observe(agentID, wolvesID, sheepsID, blocksID, getPosFromAgentState,
    #                                           getVelFromAgentState)
    # observe = lambda state: [observeOneAgent(agentID)(state) for agentID in range(numAgents)]
    #
    # getCollisionForce = GetCollisionForce()
    # applyActionForce = ApplyActionForce(wolvesID, sheepsID, entitiesMovableList)
    # applyEnvironForce = ApplyEnvironForce(numEntities, entitiesMovableList, entitiesSizeList,
    #                                       getCollisionForce, getPosFromAgentState)
    # integrateState = IntegrateState(numEntities, entitiesMovableList, massList,
    #                                 entityMaxSpeedList, getVelFromAgentState, getPosFromAgentState)
    # transit = TransitMultiAgentChasing(numEntities, reshapeAction, applyActionForce, applyEnvironForce, integrateState)
    #
    # # reset = ResetMultiAgentChasing(numAgents, numBlocks)
    # state = [[ 0.3929,  0.2188, -0.0043, -1.    ], [-0.3523,  0.3504, -0.3566, -0.5418],[-0.3358,  0.1583, -0.199,  -0.9106],
    #                  [-0.5682, -0.0665, -0.9999,  0.0165], [-0.7892,  0.1301, -0.5169, -1.1928], [-0.5682, -0.3681,  0. ,     0.    ],
    #                  [ 0.7602, -0.537,   0.,      0.    ]]
    #
    # reset = lambda: state
    #
    # actions = [[2.280e-02, 1.400e-03, 3.000e-04, 1.000e-04, 9.754e-01], [3.200e-03, 1.000e-04, 9.902e-01, 5.000e-04, 6.000e-03],
    #            [4.900e-02, 9.100e-03, 5.180e-02, 3.800e-03, 8.863e-01], [4.000e-04, 1.700e-03, 9.900e-01, 1.000e-04, 7.800e-03],
    #            [1.000e-04, 0.000e+00, 3.660e-02, 1.990e-02, 9.433e-01]]
    # observeOneAgent = lambda agentID: Observe(agentID, wolvesID, sheepsID, blocksID, getPosFromAgentState,
    #                                           getVelFromAgentState)
    # observe = lambda state: [observeOneAgent(agentID)(state) for agentID in range(numAgents)]
    #
    # initObsForParams = observe(reset())
    # obsShape = [initObsForParams[obsID].shape[0] for obsID in range(len(initObsForParams))]
    #
    #
    # worldDim = 2
    # actionDim = worldDim * 2 + 1
    #
    # layerWidth = [128, 128]
    #
    # # ------------ model ------------------------
    # buildMADDPGModels = BuildMADDPGModels(actionDim, numAgents, obsShape)
    # modelsList = [buildMADDPGModels(layerWidth, agentID) for agentID in range(numAgents)]
    #
    # dirName = os.path.dirname(__file__)
    # fileName = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}WolfActCost{}sensitive{}biteReward{}killPercent{}_agent".format(
    #     numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, costActionRatio, rewardSensitivityToDistance, biteReward, killProportion)
    # folderName = 'maddpg_rewardSensitiveToDist_23456wolves'
    # modelPaths = [os.path.join(dirName, '..', 'trainedModels', folderName, str(numWolves)+'tocopy', fileName + str(i) ) for i in range(numAgents)]
    # [restoreVariables(model, path) for model, path in zip(modelsList, modelPaths)]
    #
    # agentID = 2
    # numActionSamples = 1000
    # getAgent2QValue = GetAgentQValue(agentID, modelsList, state, actions, observe)
    # getAgent2StateV = lambda state: np.max([getAgent2QValue(state, action) for action in randomSampleActions(numActionSamples)])
    #
    # sampleRadius = 1
    # # xlower = state[2][0] - sampleRadius
    # # xupper = state[2][0] + sampleRadius
    # # ylower = state[2][1] - sampleRadius
    # # yupper = state[2][1] + sampleRadius
    # xlower = -sampleRadius
    # xupper = sampleRadius
    # ylower = -sampleRadius
    # yupper = sampleRadius
    # numStateSampleEdge = 101
    # stateSamples = randomSampleStates(xlower, xupper, ylower, yupper, numStateSampleEdge)
    #
    # resultsDF = pd.DataFrame()
    # resultsDF['x'] = [state[0] for state in stateSamples]
    # resultsDF['y'] = [state[1] for state in stateSamples]
    # resultsDF['value'] = [getAgent2StateV(state) for state in stateSamples]
    # resultsDF = resultsDF.pivot('x', 'y', 'value')
    # saveToPickle(resultsDF, os.path.join(dirName, '..', 'trainedModels', 'heatmapResult_moreStates.pickle'))

    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import numpy as np
    import seaborn as sns

    def get_alpha_blend_cmap(cmap, alpha):
        cls = plt.get_cmap(cmap)(np.linspace(0, 1, 256))
        cls = (1 - alpha) + alpha * cls
        return ListedColormap(cls)

    resultsDF = loadFromPickle(os.path.join(dirName, '..', 'trainedModels', 'heatmapResult_moreStates.pickle'))
    # ax = sns.heatmap(resultsDF, cmap=get_alpha_blend_cmap("rocket_r", 0.5), linewidths=0.0)
    ax = sns.heatmap(resultsDF, linewidths=0)#, annot= True)
    plt.savefig('heatmapResult_moreStates1.pdf', dpi = 1000)
    plt.show()

# getAgent2StateV([-0.3358, -0.2, -0.199, -0.9106])
# Out[25]: 1.1672657
# getAgent2StateV([-0.3358, 0, -0.199, -0.9106])
# Out[26]: 1.5151532

# def main():
#     numWolves = 4
#     sheepSpeedMultiplier = 1.0
#     costActionRatio = 0.0
#     rewardSensitivityToDistance = 10000.0
#     biteReward = 0.0
#
#     #
#     numSheeps = 1
#     numBlocks = 2
#     maxTimeStep = 75
#     killReward = 10
#     killProportion = 0.2
#
#     numAgents = numWolves + numSheeps
#     numEntities = numAgents + numBlocks
#     wolvesID = list(range(numWolves))
#     sheepsID = list(range(numWolves, numAgents))
#     blocksID = list(range(numAgents, numEntities))
#
#     wolfSize = 0.075
#     sheepSize = 0.05
#     blockSize = 0.2
#     entitiesSizeList = [wolfSize] * numWolves + [sheepSize] * numSheeps + [blockSize] * numBlocks
#
#     wolfMaxSpeed = 1.0
#     blockMaxSpeed = None
#     sheepMaxSpeedOriginal = 1.3
#     sheepMaxSpeed = sheepMaxSpeedOriginal * sheepSpeedMultiplier
#
#     entityMaxSpeedList = [wolfMaxSpeed] * numWolves + [sheepMaxSpeed] * numSheeps + [blockMaxSpeed] * numBlocks
#     entitiesMovableList = [True] * numAgents + [False] * numBlocks
#     massList = [1.0] * numEntities
#
#     collisionReward = 10 # originalPaper = 10*3
#     isCollision = IsCollision(getPosFromAgentState)
#     punishForOutOfBound = PunishForOutOfBound()
#     rewardSheep = RewardSheep(wolvesID, sheepsID, entitiesSizeList, getPosFromAgentState, isCollision,
#                               punishForOutOfBound, collisionPunishment = collisionReward)
#
#     collisionDist = wolfSize + sheepSize
#     getAgentsPercentageOfRewards = GetAgentsPercentageOfRewards(rewardSensitivityToDistance, collisionDist)
#     terminalCheck = TerminalCheck()
#
#     getCollisionWolfReward = GetCollisionWolfReward(biteReward, killReward, killProportion, sampleFromDistribution, terminalCheck)
#     getWolfSheepDistance = GetWolfSheepDistance(computeVectorNorm, getPosFromAgentState)
#     rewardWolf = RewardWolvesWithKillProb(wolvesID, sheepsID, entitiesSizeList, isCollision, terminalCheck, getWolfSheepDistance,
#                  getAgentsPercentageOfRewards, getCollisionWolfReward)
#
#     reshapeAction = ReshapeAction()
#     getActionCost = GetActionCost(costActionRatio, reshapeAction, individualCost=True)
#     getWolvesAction = lambda action: [action[wolfID] for wolfID in wolvesID]
#     rewardWolfWithActionCost = lambda state, action, nextState: np.array(rewardWolf(state, action, nextState)) - np.array(getActionCost(getWolvesAction(action)))
#
#     rewardFunc = lambda state, action, nextState: \
#         list(rewardWolfWithActionCost(state, action, nextState)) + list(rewardSheep(state, action, nextState))
#
#     reset = ResetMultiAgentChasing(numAgents, numBlocks)
#     observeOneAgent = lambda agentID: Observe(agentID, wolvesID, sheepsID, blocksID, getPosFromAgentState,
#                                               getVelFromAgentState)
#     observe = lambda state: [observeOneAgent(agentID)(state) for agentID in range(numAgents)]
#
#     getCollisionForce = GetCollisionForce()
#     applyActionForce = ApplyActionForce(wolvesID, sheepsID, entitiesMovableList)
#     applyEnvironForce = ApplyEnvironForce(numEntities, entitiesMovableList, entitiesSizeList,
#                                           getCollisionForce, getPosFromAgentState)
#     integrateState = IntegrateState(numEntities, entitiesMovableList, massList,
#                                     entityMaxSpeedList, getVelFromAgentState, getPosFromAgentState)
#     transit = TransitMultiAgentChasing(numEntities, reshapeAction, applyActionForce, applyEnvironForce, integrateState)
#
#     isTerminal = lambda state: False #lambda state: terminalCheck.terminal
#     initObsForParams = observe(reset())
#     obsShape = [initObsForParams[obsID].shape[0] for obsID in range(len(initObsForParams))]
#
#     sampleTrajectory = SampleTrajectoryResetAtTerminal(maxRunningStepsToSample, transit, isTerminal, rewardFunc, reset)
#
#     worldDim = 2
#     actionDim = worldDim * 2 + 1
#
#     layerWidth = [128, 128]
#
#     # ------------ model ------------------------
#     buildMADDPGModels = BuildMADDPGModels(actionDim, numAgents, obsShape)
#     modelsList = [buildMADDPGModels(layerWidth, agentID) for agentID in range(numAgents)]
#
#     dirName = os.path.dirname(__file__)
#     fileName = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}WolfActCost{}sensitive{}biteReward{}killPercent{}_agent".format(
#         numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, costActionRatio, rewardSensitivityToDistance, biteReward, killProportion)
#     folderName = 'maddpg_rewardSensitiveToDist_23456wolves'
#     modelPaths = [os.path.join(dirName, '..', 'trainedModels', folderName, str(numWolves)+'tocopy', fileName + str(i) ) for i in range(numAgents)]
#     [restoreVariables(model, path) for model, path in zip(modelsList, modelPaths)]
#
#     actOneStepOneModel = ActOneStep(actByPolicyTrainNoisy)
#     policy = lambda allAgentsStates: [actOneStepOneModel(model, observe(allAgentsStates)) for model in modelsList]
#
#     rewardList = []
#     numTrajToSample = 10 #300
#     trajToRender = []
#     trajList = []
#
#     for i in range(numTrajToSample):
#         traj = sampleTrajectory(policy)
#         rewardList.append(rew)
#         trajToRender = trajToRender + list(traj)
#         trajList.append(traj)
#
#     meanTrajReward = np.mean(rewardList)
#     seTrajReward = np.std(rewardList) / np.sqrt(len(rewardList) - 1)
#     print('meanTrajReward', meanTrajReward, 'se ', seTrajReward)

if __name__ == '__main__':
    main()




