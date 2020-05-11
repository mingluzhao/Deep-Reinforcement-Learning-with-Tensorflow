import sys
import os

DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..'))
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pickle

from environment.chasingEnv.continuousChasingVisualization import *
import pandas as pd
from pygame.color import THECOLORS
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def loadFromPickle(path):
    pickleIn = open(path, 'rb')
    object = pickle.load(pickleIn)
    pickleIn.close()
    return object

def main():
    dirName = os.path.dirname(__file__)
    dataPath = os.path.join(dirName, '..', 'trajectory', 'traj200steps' + '.pickle')
    trajectory = loadFromPickle(dataPath)

    numAgents = 2
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
    imageFolderName = 'DemoAfterChange'
    saveImageDir = os.path.join(os.path.join(parentDir, 'plots'), imageFolderName)
    if not os.path.exists(saveImageDir):
        os.makedirs(saveImageDir)

    chaseTrial(numberOfAgents, positionListToDraw, saveImageDir)


if __name__ == '__main__':
    main()