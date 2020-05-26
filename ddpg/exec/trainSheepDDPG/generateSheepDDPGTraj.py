import os
import numpy as np
import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))
sys.path.append(os.path.join(dirName, '..'))


from src.ddpg import actByPolicyTrain, BuildActorModel
from environment.chasingEnv.reward import RewardFunctionCompete, GetBoundaryPunishment, RewardSheepWithBoundaryHeuristics
from environment.chasingEnv.chasingPolicy import HeatSeekingContinuousDeterministicPolicy
from src.policy import ActDDPGOneStep
from environment.chasingEnv.envNoPhysics import Reset, TransitForNoPhysics, getIntendedNextState, StayWithinBoundary, \
    TransitWithSingleWolf, GetAgentPosFromState, IsTerminal
from functionTools.trajectory import SampleTrajectory
from environment.chasingEnv.continuousChasingVisualization import initializeScreen, Observe, ScaleTrajectory,\
    AdjustDfFPStoTraj, DrawBackground, DrawState, ChaseTrialWithTraj
from pygame.color import THECOLORS
from functionTools.loadSaveModel import restoreVariables



def main():
    numAgents = 2
    stateDim = numAgents * 2
    actionLow = -1
    actionHigh = 1
    actionBound = (actionHigh - actionLow)/2
    actionDim = 2

    buildActorModel = BuildActorModel(stateDim, actionDim, actionBound)
    actorLayerWidths = [64]
    actorWriter, actorModel = buildActorModel(actorLayerWidths)

    dirName = os.path.dirname(__file__)
    actorModelPath = os.path.join(dirName, '..', 'trainedDDPGModels', 'actorModel=0_dimension=2_gamma=0.95_learningRateActor=0.01_learningRateCritic=0.01_maxEpisode=1000_maxTimeStep=50_minibatchSize=32_wolfSpeed=3.ckpt')
                                  # 'actorModel=0_dimension=2_gamma=0.95_learningRateActor=0.01_learningRateCritic=0.01_maxEpisode=5000_maxTimeStep=100_minibatchSize=32_wolfSpeed=3.ckpt')

    restoreVariables(actorModel, actorModelPath)
    sheepPolicy =  ActDDPGOneStep(actionLow, actionHigh, actByPolicyTrain, actorModel, getNoise = None)


    sheepId = 0
    wolfId = 1
    getSheepPos = GetAgentPosFromState(sheepId)
    getWolfPos = GetAgentPosFromState(wolfId)

    wolfSpeed = 3
    wolfPolicy = HeatSeekingContinuousDeterministicPolicy(getWolfPos, getSheepPos, wolfSpeed)
    xBoundary = (0, 20)
    yBoundary = (0, 20)
    stayWithinBoundary = StayWithinBoundary(xBoundary, yBoundary)
    transit = TransitForNoPhysics(getIntendedNextState, stayWithinBoundary)
    # transit = TransitWithSingleWolf(physicalTransition, wolfPolicy)

    maxTimeStep = 50
    sheepAliveBonus = 1 / maxTimeStep
    sheepTerminalPenalty = 10
    killzoneRadius = 1
    isTerminal = IsTerminal(getWolfPos, getSheepPos, killzoneRadius)
    getBoundaryPunishment = GetBoundaryPunishment(xBoundary, yBoundary, sheepIndex=0, punishmentVal= 10)
    rewardSheep = RewardFunctionCompete(sheepAliveBonus, sheepTerminalPenalty, isTerminal)
    rewardSheepWithBoundaryHeuristics = RewardSheepWithBoundaryHeuristics(rewardSheep, getIntendedNextState, getBoundaryPunishment, getSheepPos)

    getSheepAction = lambda actions: [actions[sheepId* actionDim], actions[sheepId* actionDim+ 1]]
    getReward = lambda state, action, nextState: rewardSheepWithBoundaryHeuristics(state, getSheepAction(action), nextState)

    policy = lambda state: list(sheepPolicy(state)) + list(wolfPolicy(state))
    reset = Reset(xBoundary, yBoundary, numAgents)


    for i in range(20):
        maxRunningSteps = 50
        sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, getReward, reset)
        trajectory = sampleTrajectory(policy)

        # plots& plot
        showDemo = True
        if showDemo:
            observe = Observe(trajectory, numAgents)

            fullScreen = False
            screenWidth = 800
            screenHeight = 800
            screen = initializeScreen(fullScreen, screenWidth, screenHeight)

            leaveEdgeSpace = 200
            lineWidth = 3
            xBoundary = [leaveEdgeSpace, screenWidth - leaveEdgeSpace * 2]
            yBoundary = [leaveEdgeSpace, screenHeight - leaveEdgeSpace * 2]
            screenColor = THECOLORS['black']
            lineColor = THECOLORS['white']

            drawBackground = DrawBackground(screen, screenColor, xBoundary, yBoundary, lineColor, lineWidth)
            circleSize = 10
            positionIndex = [0, 1]
            drawState = DrawState(screen, circleSize, positionIndex, drawBackground)

            numberOfAgents = 2
            chasingColors = [THECOLORS['green'], THECOLORS['red']]
            colorSpace = chasingColors[: numberOfAgents]

            FPS = 60
            chaseTrial = ChaseTrialWithTraj(FPS, colorSpace, drawState, saveImage=True)

            rawXRange = [0, 20]
            rawYRange = [0, 20]
            scaledXRange = [210, 590]
            scaledYRange = [210, 590]
            scaleTrajectory = ScaleTrajectory(positionIndex, rawXRange, rawYRange, scaledXRange, scaledYRange)

            oldFPS = 5
            adjustFPS = AdjustDfFPStoTraj(oldFPS, FPS)

            getTrajectory = lambda rawTrajectory: scaleTrajectory(adjustFPS(rawTrajectory))
            positionList = [observe(index) for index in range(len(trajectory))]
            positionListToDraw = getTrajectory(positionList)

            currentDir = os.getcwd()
            parentDir = os.path.abspath(os.path.join(currentDir, os.pardir))
            imageFolderName = 'Demo'
            saveImageDir = os.path.join(os.path.join(parentDir, 'chasingDemo'), imageFolderName)
            if not os.path.exists(saveImageDir):
                os.makedirs(saveImageDir)

            chaseTrial(numberOfAgents, positionListToDraw, saveImageDir)

if __name__ == '__main__':
    main()