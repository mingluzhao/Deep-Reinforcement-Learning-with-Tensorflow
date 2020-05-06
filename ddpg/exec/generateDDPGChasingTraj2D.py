import os

DIRNAME = os.path.dirname(__file__)
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from environment.chasingEnv.envNoPhysics import *
from src.ddpg import *
from src.policy import *
from functionTools.trajectory import *
from environment.chasingEnv.continuousChasingVisualization import *
from pygame.color import THECOLORS
from functionTools.loadSaveModel import *
from functionTools.trajectory import *

import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

np.random.seed(1)
tf.set_random_seed(1)

def main():
    np.random.seed(1)
    tf.set_random_seed(1)

    dirName = os.path.dirname(__file__)
    path = os.path.join(dirName, '..', 'trajectory', 'expModelTraj200Steps.pickle')
    trajOfExpModel = loadFromPickle(path)
    initState = trajOfExpModel[0] # [10.30991692 14.76448735  3.88667778 14.11653945]
    
    numAgents = 2
    numStateSpace = numAgents * 2
    actionRange = 1
    actionDim = 2
    
    buildActorModel = BuildActorModel(numStateSpace, actionDim, actionRange)
    actorLayerWidths = [64, 64, 64]
    actorWriter, actorModel = buildActorModel(actorLayerWidths)

    # sheep NN Policy
    sheepActModelPath = os.path.join(dirName, '..', 'trainedDDPGModels', 'actorModel=0_dimension=2_gamma=0.95_learningRateActor=0.01_learningRateCritic=0.01_maxEpisode=5000_maxTimeStep=25_minibatchSize=1024_wolfSpeed=3.ckpt')
    restoreVariables(actorModel, sheepActModelPath)
    sheepPolicy = ActByDDPG2D(actByPolicyTrain, actorModel)

###physics
    xBoundary = (0, 20)
    yBoundary = (0, 20)
    stayWithinBoundary = StayWithinBoundary(xBoundary, yBoundary)
    transit = TransitForNoPhysics(stayWithinBoundary)

    sheepId = 0
    wolfId = 1
    getSheepXPos = GetAgentPosFromState(sheepId)
    getWolfXPos = GetAgentPosFromState(wolfId)

    actionMagnitude = 3
    wolfPolicy = HeatSeekingContinuousDeterministicPolicy(getWolfXPos, getSheepXPos, actionMagnitude)

    killzoneRadius = 1
    isTerminal = IsTerminal(getWolfXPos, getSheepXPos, killzoneRadius)
    numAgents = 2
    # reset = Reset(xBoundary, yBoundary, numAgents)
    reset = lambda: initState


    policy = lambda state: list(sheepPolicy(state)) + list(wolfPolicy(state))
    maxRunningSteps = 20        # max possible length of the trajectory/episode
    sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset)
    trajectory = sampleTrajectory(policy)
    print(trajectory)

    dataPath = os.path.join(dirName, '..', 'trajectory', 'traj5000stepsNewParam' + '.pickle')
    saveToPickle(trajectory, dataPath)

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
    saveImageDir = os.path.join(os.path.join(parentDir, 'demo'), imageFolderName)
    if not os.path.exists(saveImageDir):
        os.makedirs(saveImageDir)

    chaseTrial(numberOfAgents, positionListToDraw, saveImageDir)


if __name__ == '__main__':
    main()

