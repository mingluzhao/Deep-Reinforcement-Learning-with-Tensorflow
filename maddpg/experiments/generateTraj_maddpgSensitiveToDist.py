import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
import logging

logging.getLogger('tensorflow').setLevel(logging.ERROR)

import time

from environment.chasingEnv.multiAgentEnv import *
from functionTools.loadSaveModel import saveToPickle, restoreVariables, loadFromPickle
from functionTools.trajectory import SampleTrajectoryResetAtTerminal
from visualize.visualizeMultiAgent import *
from visualize.drawDemo import *
from environment.chasingEnv.rewardWithKillProbSensitiveToDist import *

from pygame.color import THECOLORS
from maddpg.maddpgAlgor.trainer.myMADDPG import *
import json

wolfColor = np.array([0.85, 0.35, 0.35])
sheepColor = np.array([0.35, 0.85, 0.35])
blockColor = np.array([0.25, 0.25, 0.25])

maxEpisode = 60000
maxRunningStepsToSample = 75  # num of timesteps in one eps

def calcWolvesTrajReward(traj, wolvesID):
    rewardIDinTraj = 2
    getWolfReward = lambda allAgentsReward: np.sum([allAgentsReward[wolfID] for wolfID in wolvesID])
    rewardList = [getWolfReward(timeStepInfo[rewardIDinTraj]) for timeStepInfo in traj]
    trajReward = np.sum(rewardList)
    trajReward = trajReward/10
    print(trajReward)

    return trajReward

def main():
    numWolves = 3
    sheepSpeedMultiplier = 0.75
    costActionRatio = 0.0
    rewardSensitivityToDistance = 10000.0
    biteReward = 0.0

    #
    numSheeps = 1
    numBlocks = 2
    maxTimeStep = 75
    killReward = 10
    killProportion = 0.2

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

    collisionReward = 10 # originalPaper = 10*3
    isCollision = IsCollision(getPosFromAgentState)
    punishForOutOfBound = PunishForOutOfBound()
    rewardSheep = RewardSheep(wolvesID, sheepsID, entitiesSizeList, getPosFromAgentState, isCollision,
                              punishForOutOfBound, collisionPunishment = collisionReward)

    collisionDist = wolfSize + sheepSize
    getAgentsPercentageOfRewards = GetAgentsPercentageOfRewards(rewardSensitivityToDistance, collisionDist)
    terminalCheck = TerminalCheck()

    getCollisionWolfReward = GetCollisionWolfReward(biteReward, killReward, killProportion, sampleFromDistribution, terminalCheck)
    getWolfSheepDistance = GetWolfSheepDistance(computeVectorNorm, getPosFromAgentState)
    rewardWolf = RewardWolvesWithKillProb(wolvesID, sheepsID, entitiesSizeList, isCollision, terminalCheck, getWolfSheepDistance,
                 getAgentsPercentageOfRewards, getCollisionWolfReward)

    reshapeAction = ReshapeAction()
    getActionCost = GetActionCost(costActionRatio, reshapeAction, individualCost=True)
    getWolvesAction = lambda action: [action[wolfID] for wolfID in wolvesID]
    rewardWolfWithActionCost = lambda state, action, nextState: np.array(rewardWolf(state, action, nextState)) - np.array(getActionCost(getWolvesAction(action)))

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
    transit = TransitMultiAgentChasing(numEntities, reshapeAction, applyActionForce, applyEnvironForce, integrateState)

    isTerminal = lambda state: False #lambda state: terminalCheck.terminal
    initObsForParams = observe(reset())
    obsShape = [initObsForParams[obsID].shape[0] for obsID in range(len(initObsForParams))]

    sampleTrajectory = SampleTrajectoryResetAtTerminal(maxRunningStepsToSample, transit, isTerminal, rewardFunc, reset)

    worldDim = 2
    actionDim = worldDim * 2 + 1

    layerWidth = [128, 128]

    # ------------ model ------------------------
    buildMADDPGModels = BuildMADDPGModels(actionDim, numAgents, obsShape)
    modelsList = [buildMADDPGModels(layerWidth, agentID) for agentID in range(numAgents)]

    dirName = os.path.dirname(__file__)
    fileName = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}WolfActCost{}sensitive{}biteReward{}killPercent{}_agent".format(
        numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, costActionRatio, rewardSensitivityToDistance, biteReward, killProportion)
    folderName = 'maddpg_rewardSensitiveToDist_23456wolves'
    modelPaths = [os.path.join(dirName, '..', 'trainedModels', folderName, str(numWolves)+'tocopy', fileName + str(i) ) for i in range(numAgents)]
    [restoreVariables(model, path) for model, path in zip(modelsList, modelPaths)]

    actOneStepOneModel = ActOneStep(actByPolicyTrainNoisy)
    policy = lambda allAgentsStates: [actOneStepOneModel(model, observe(allAgentsStates)) for model in modelsList]

    wolfColor = np.array([0.85, 0.35, 0.35])
    sheepColor = np.array([0.35, 0.85, 0.35])
    blockColor = np.array([0.25, 0.25, 0.25])
    entitiesColorList = [wolfColor] * numWolves + [sheepColor] * numSheeps + [blockColor] * numBlocks
    render = Render(entitiesSizeList, entitiesColorList, numAgents, getPosFromAgentState)

    rewardList = []
    numTrajToSample = 10 #300
    trajToRender = []
    trajList = []

    for i in range(numTrajToSample):
        traj = sampleTrajectory(policy)
        rew = calcWolvesTrajReward(traj, wolvesID)
        rewardList.append(rew)
        trajToRender = trajToRender + list(traj)
        trajList.append(traj)

    meanTrajReward = np.mean(rewardList)
    seTrajReward = np.std(rewardList) / np.sqrt(len(rewardList) - 1)
    print('meanTrajReward', meanTrajReward, 'se ', seTrajReward)
    # render(trajToRender)

    # trajectoryDirectory = os.path.join(dirName, '..', 'trajectories', folderName)
    # if not os.path.exists(trajectoryDirectory):
    #     os.makedirs(trajectoryDirectory)
    # trajFileName = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}WolfActCost{}sensitive{}biteReward{}killPercent{}_Traj".format(
    #     numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, costActionRatio,
    #     rewardSensitivityToDistance, biteReward, killProportion)
    #
    # trajSavePath = os.path.join(trajectoryDirectory, trajFileName)
    # saveToPickle(trajList, trajSavePath)

    # trajList = loadFromPickle(trajSavePath)

    # visualize ------------

    screenWidth = 700
    screenHeight = 700
    screen = pg.display.set_mode((screenWidth, screenHeight))
    screenColor = THECOLORS['black']
    xBoundary = [0, 700]
    yBoundary = [0, 700]
    lineColor = THECOLORS['white']
    lineWidth = 4
    drawBackground = DrawBackground(screen, screenColor, xBoundary, yBoundary, lineColor, lineWidth)

    FPS = 10
    numBlocks = 2
    wolfColor = [255, 255, 255]
    sheepColor = [0, 250, 0]
    blockColor = [200, 200, 200]
    circleColorSpace = [wolfColor] * numWolves + [sheepColor] * numSheeps + [blockColor] * numBlocks
    viewRatio = 1
    sheepSize = int(0.05 * screenWidth / (2 * viewRatio))
    wolfSize = int(0.075 * screenWidth / (3 * viewRatio))
    blockSize = int(0.2 * screenWidth / (2 * viewRatio))
    circleSizeSpace = [wolfSize] * numWolves + [sheepSize] * numSheeps + [blockSize] * numBlocks
    positionIndex = [0, 1]
    agentIdsToDraw = list(range(numWolves + numSheeps + numBlocks))

    conditionName = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}WolfActCost{}sensitive{}biteReward{}killPercent{}".format(
        numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, costActionRatio,
        rewardSensitivityToDistance, biteReward, killProportion)
    imageSavePath = os.path.join(dirName, '..', 'trajectories', folderName, conditionName)
    if not os.path.exists(imageSavePath):
        os.makedirs(imageSavePath)
    imageFolderName = str('forDemo')
    saveImageDir = os.path.join(os.path.join(imageSavePath, imageFolderName))
    if not os.path.exists(saveImageDir):
        os.makedirs(saveImageDir)
    imaginedWeIdsForInferenceSubject = list(range(numWolves))

    updateColorSpaceByPosterior = None
    outsideCircleAgentIds = imaginedWeIdsForInferenceSubject
    outsideCircleColor = np.array([[255, 0, 0]] * numWolves)
    outsideCircleSize = int(wolfSize * 1.5)
    drawCircleOutside = DrawCircleOutsideEnvMADDPG(screen, viewRatio, outsideCircleAgentIds, positionIndex,
                                                   outsideCircleColor, outsideCircleSize)

    saveImage = False
    numAgents = numWolves + numSheeps
    sheepsID = list(range(numWolves, numAgents))

    drawState = DrawStateEnvMADDPG(FPS, screen, viewRatio, circleColorSpace, circleSizeSpace, agentIdsToDraw,
                                   positionIndex, saveImage, saveImageDir, sheepsID, wolvesID,
                                   drawBackground, sensitiveZoneSize=None, updateColorByPosterior=updateColorSpaceByPosterior, drawCircleOutside=drawCircleOutside)

    # MDP Env
    interpolateState = None
    stateIndexInTimeStep = 0
    actionIndexInTimeStep = 1
    posteriorIndexInTimeStep = None

    stateID = 0
    nextStateID = 3
    wolfSizeForCheck = 0.075
    sheepSizeForCheck = 0.05
    checkStatus = CheckStatus(wolvesID, sheepsID, isCollision, wolfSizeForCheck, sheepSizeForCheck, stateID, nextStateID)
    chaseTrial = ChaseTrialWithTrajWithKillNotation(stateIndexInTimeStep, drawState, checkStatus, interpolateState, actionIndexInTimeStep, posteriorIndexInTimeStep)

    [chaseTrial(trajectory) for trajectory in np.array(trajList[:20])]

    # screenWidth = 700
    # screenHeight = 700
    # screen = pg.display.set_mode((screenWidth, screenHeight))
    # screenColor = THECOLORS['black']
    # xBoundary = [0, 700]
    # yBoundary = [0, 700]
    # lineColor = THECOLORS['white']
    # lineWidth = 4
    # drawBackground = DrawBackground(screen, screenColor, xBoundary, yBoundary, lineColor, lineWidth)
    #
    # FPS = 10
    # numBlocks = 2
    # predatorColor = THECOLORS['white']# [255, 255, 255]
    # preyColor = THECOLORS['green'] #[0, 250, 0]
    # blockColor = THECOLORS['grey']
    # circleColorSpace = [predatorColor] * numWolves + [preyColor] * numSheeps + [blockColor] * numBlocks
    # viewRatio = 1
    # preySize = int(0.05 * screenWidth / (2 * viewRatio))
    # predatorSize = int(0.075 * screenWidth / (3 * viewRatio))
    # blockSize = int(0.2 * screenWidth / (2 * viewRatio))
    # circleSizeSpace = [predatorSize] * numWolves + [preySize] * numSheeps + [blockSize] * numBlocks
    # positionIndex = [0, 1]
    # agentIdsToDraw = list(range(numWolves + numSheeps + numBlocks))
    #
    # conditionName = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}WolfActCost{}sensitive{}biteReward{}killPercent{}".format(
    #     numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, costActionRatio,
    #     rewardSensitivityToDistance, biteReward, killProportion)
    # imageSavePath = os.path.join(dirName, '..', 'trajectories', conditionName)
    # if not os.path.exists(imageSavePath):
    #     os.makedirs(imageSavePath)
    # imageFolderName = str('forDemo')
    # saveImageDir = os.path.join(os.path.join(imageSavePath, imageFolderName))
    # if not os.path.exists(saveImageDir):
    #     os.makedirs(saveImageDir)
    #
    # outsideCircleColor = [THECOLORS['red']] * numWolves
    # outsideCircleSize = int(predatorSize * 1.5)
    # drawCircleOutside = DrawCircleOutside(screen, wolvesID, positionIndex,
    #                                       outsideCircleColor, outsideCircleSize, viewRatio=viewRatio)
    # saveImage = False
    # drawState = DrawState(FPS, screen, circleColorSpace, circleSizeSpace, agentIdsToDraw,
    #                       positionIndex, saveImage, saveImageDir, sheepsID, wolvesID,
    #                       drawBackground, drawCircleOutside=drawCircleOutside, viewRatio=viewRatio)
    #
    # # MDP Env
    # stateID = 0
    # nextStateID = 3
    # predatorSizeForCheck = 0.075
    # preySizeForCheck = 0.05
    # checkStatus = CheckStatus(wolvesID, sheepsID, isCollision, predatorSizeForCheck, preySizeForCheck, stateID,
    #                           nextStateID)
    # chaseTrial = ChaseTrialWithTrajWithKillNotation(stateID, drawState, checkStatus)
    # [chaseTrial(trajectory) for trajectory in np.array(trajList[:20])]

if __name__ == '__main__':
    main()