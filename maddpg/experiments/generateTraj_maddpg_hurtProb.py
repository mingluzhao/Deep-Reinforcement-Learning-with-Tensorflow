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
from visualize.visualizeMultiAgent import *
from environment.chasingEnv.multiAgentEnvWithIndividReward import RewardWolfIndividual
from environment.chasingEnv.rewardWithProbablisticKill import *
from maddpg.maddpgAlgor.trainer.myMADDPG import *
import json
import pygame as pg
from pygame.color import THECOLORS

from visualize.drawDemo import DrawBackground,  DrawCircleOutsideEnvMADDPG, DrawStateEnvMADDPG, ChaseTrialWithTraj

maxEpisode = 60000
maxRunningStepsToSample = 75 # num of timesteps in one eps

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
    numWolves = 3
    numSheeps = 1
    numBlocks = 2
    maxTimeStep = 75
    sheepSpeedMultiplier = 1.0
    costActionRatio = 0.0

    individualRewardWolf = 0
    oneWolfSelfHurtProb = 0.0 #
    sensitiveZoneRadius = 0.25 #

    saveTraj = False
    saveImage = False
    visualizeTraj = True
    useOriginalEnvDemo = True
    
    print("maddpg: {} wolves, {} sheep, {} blocks, saveTraj: {}, visualize: {}".format(numWolves, numSheeps, numBlocks, str(saveTraj), str(visualizeTraj)))

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
                              punishForOutOfBound, collisionPunishment=10)
    getHurtProbOfCatching = GetHurtProbOfCatchingByDeterministicZone(getPosFromAgentState, computeVectorNorm, sensitiveZoneRadius,
                                                                     oneWolfSelfHurtProb)
    hurtReward = -5
    collisionReward = 10
    rewardWolf = RewardWolfWithHurtProb(wolvesID, sheepsID, entitiesSizeList, isCollision, getHurtProbOfCatching,
                                        sampleFromDistribution, individualRewardWolf, hurtReward, collisionReward)
    reshapeAction = ReshapeAction()
    getActionCost = GetActionCost(costActionRatio, reshapeAction, individualCost=True)
    getWolvesAction = lambda action: [action[wolfID] for wolfID in wolvesID]
    rewardWolfWithActionCost = lambda state, action, nextState: np.array(rewardWolf(state, action, nextState)) - np.array(getActionCost(getWolvesAction(action)))

    rewardFunc = lambda state, action, nextState: \
        list(rewardWolfWithActionCost(state, action, nextState)) + list(rewardSheep(state, action, nextState))

    reset = ResetMultiAgentChasing(numAgents, numBlocks)
    observeOneAgent = lambda agentID: Observe(agentID, wolvesID, sheepsID, blocksID, getPosFromAgentState, getVelFromAgentState)
    observe = lambda state: [observeOneAgent(agentID)(state) for agentID in range(numAgents)]

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

    # ------------ model ------------------------
    buildMADDPGModels = BuildMADDPGModels(actionDim, numAgents, obsShape)
    modelsList = [buildMADDPGModels(layerWidth, agentID) for agentID in range(numAgents)]

    dirName = os.path.dirname(__file__)
    individStr = 'individ' if individualRewardWolf else 'shared'
    fileName = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}WolfActCost{}HurtProb{}Radius{}{}_agent".format(
        numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, costActionRatio, oneWolfSelfHurtProb, sensitiveZoneRadius, individStr)

    folderName = '3wolvesMaddpgWithProbOfHurtBySheep0729'
    modelPaths = [os.path.join(dirName, '..', 'trainedModels', folderName, fileName + str(i)) for i in range(numAgents)]

    # dirName = os.path.dirname(__file__)
    # individStr = 'individ' if individualRewardWolf else 'shared'
    # fileName = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}WolfActCost{}HurtProb{}Radius{}{}_agent".format(
    #     numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, costActionRatio, oneWolfSelfHurtProb, sensitiveZoneRadius, individStr)
    #
    # folderName = '3wolvesMaddpgWithProbOfHurtBySheep'
    # wolfModelPaths = [os.path.join(dirName, '..', 'trainedModels', folderName, fileName + str(i)) for i in wolvesID]
    #
    # sheepFileName =  "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}WolfActCost{}HurtProb{}Radius{}{}_agent".format(
    #     numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, costActionRatio, oneWolfSelfHurtProb, 0.25, individStr)
    # sheepModelPaths = [os.path.join(dirName, '..', 'trainedModels', '3wolvesMaddpgWithProbOfHurtBySheep', sheepFileName + str(i)) for i in sheepsID]
    #
    # modelPaths = wolfModelPaths + sheepModelPaths


    [restoreVariables(model, path) for model, path in zip(modelsList, modelPaths)]

    actOneStepOneModel = ActOneStep(actByPolicyTrainNoisy)
    policy = lambda allAgentsStates: [actOneStepOneModel(model, observe(allAgentsStates)) for model in modelsList]

    trajList = []
    rewardList = []
    numTrajToSample = 10
    for i in range(numTrajToSample):
        traj = sampleTrajectory(policy)
        rew = calcTrajRewardWithSharedWolfReward(traj) if not individualRewardWolf else calcTrajRewardWithIndividualWolfReward(traj, wolvesID)
        rewardList.append(rew)
        trajList.append(list(traj))

    meanTrajReward = np.mean(rewardList)
    seTrajReward = np.std(rewardList) / np.sqrt(len(rewardList) - 1)
    print('meanTrajRewardSharedWolf', meanTrajReward, 'se ', seTrajReward)

    trajectoryDirectory = os.path.join(dirName, '..', 'trajectories', folderName)
    # if not os.path.exists(trajectoryDirectory):
    #     os.makedirs(trajectoryDirectory)
    # trajFileName = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}WolfActCost{}{}".format(
    #     numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, costActionRatio, individStr)
    # trajSavePath = os.path.join(trajectoryDirectory, trajFileName)
    # saveToPickle(trajList, trajSavePath)

    # visualize
    # visualizeTraj = False
    if visualizeTraj:
        if useOriginalEnvDemo:
            wolfColor = np.array([0.85, 0.35, 0.35])
            sheepColor = np.array([0.35, 0.85, 0.35])
            blockColor = np.array([0.25, 0.25, 0.25])
            entitiesColorList = [wolfColor] * numWolves + [sheepColor] * numSheeps + [blockColor] * numBlocks
            render = Render(entitiesSizeList, entitiesColorList, numAgents, getPosFromAgentState)
            trajToRender = np.concatenate(trajList)
            render(trajToRender)
        else:
            # generate demo image
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
            viewRatio = 1.5
            sheepSize = int(0.05 * screenWidth / (2 * viewRatio))
            wolfSize = int(0.075 * screenWidth / (3 * viewRatio))
            blockSize = int(0.2 * screenWidth / (3 * viewRatio))
            circleSizeSpace = [wolfSize] * numWolves + [sheepSize] * numSheeps + [blockSize] * numBlocks
            positionIndex = [0, 1]
            agentIdsToDraw = list(range(numWolves + numSheeps + numBlocks))


            imageSavePath = os.path.join(trajectoryDirectory, 'picMovingSheep')
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

            sensitiveZoneSize = int(sensitiveZoneRadius * screenWidth / (2 * viewRatio))
            drawState = DrawStateEnvMADDPG(FPS, screen, viewRatio, circleColorSpace, circleSizeSpace, agentIdsToDraw,
                                           positionIndex,
                                           saveImage, saveImageDir, sheepsID, sensitiveZoneSize,
                                           drawBackground, updateColorSpaceByPosterior,
                                           drawCircleOutside)
            # MDP Env
            interpolateState = None
    
            stateIndexInTimeStep = 0
            actionIndexInTimeStep = 1
            posteriorIndexInTimeStep = None
            chaseTrial = ChaseTrialWithTraj(stateIndexInTimeStep, drawState, interpolateState, actionIndexInTimeStep,
                                            posteriorIndexInTimeStep)
    
            # print(len(trajectories))
            lens = [len(trajectory) for trajectory in trajList]
            maxWolfPositions = np.array([max([max([max(abs(timeStep[0][wolfId][0]), abs(timeStep[0][wolfId][1]))
                                                   for wolfId in range(numWolves)])
                                              for timeStep in trajectory])
                                         for trajectory in trajList])
            flags = maxWolfPositions < 1.3 * viewRatio
            index = flags.nonzero()[0]
            print(trajList[0][1])
            # [chaseTrial(trajectory) for trajectory in np.array(trajList)[index[[0, 2, 3]]]]
            [chaseTrial(trajectory) for trajectory in np.array(trajList)[index]]


if __name__ == '__main__':
    main()
