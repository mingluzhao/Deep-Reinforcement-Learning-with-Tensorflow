import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import random
import numpy as np
import pickle
from collections import OrderedDict
import pandas as pd
from matplotlib import pyplot as plt
import itertools as it
import pygame as pg
from pygame.color import THECOLORS

from visualize.drawDemo import DrawBackground,  DrawCircleOutsideEnvMADDPG, DrawState, \
    DrawStateEnvMADDPG, ChaseTrialWithTraj
from functionTools.trajectoriesSaveLoad import loadFromPickle


def main():
    numWolves = 3
    numSheep = 1
    numBlocks = 2
    maxTimeStep = 25
    sheepSpeedMultiplier = 1.0
    individualRewardWolf = int(False)
    costActionRatio = 0.1
    maxEpisode = 60000
    
    dirName = os.path.dirname(__file__)
    individStr = 'individ' if individualRewardWolf else 'shared'
    folderName = '2and3wolvesMaddpgWithActionCost'
    trajectoryDirectory = os.path.join(dirName, '..', 'trajectories', folderName)
    trajFileName = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}{}".format(
        numWolves, numSheep, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, individStr)

    if costActionRatio > 0:
        trajFileName = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}WolfActCost{}{}".format(
            numWolves, numSheep, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, costActionRatio, individStr)
    trajSavePath = os.path.join(trajectoryDirectory, trajFileName)
    trajectories = loadFromPickle(trajSavePath)

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
    circleColorSpace = [wolfColor] * numWolves + [sheepColor] * numSheep + [blockColor] * numBlocks
    viewRatio = 1.5
    sheepSize = int(0.05 * screenWidth / (2 * viewRatio))
    wolfSize = int(0.075 * screenWidth / (3 * viewRatio))
    blockSize = int(0.2 * screenWidth / (3 * viewRatio))
    circleSizeSpace = [wolfSize] * numWolves + [sheepSize] * numSheep + [blockSize] * numBlocks
    positionIndex = [0, 1]
    agentIdsToDraw = list(range(numWolves + numSheep + numBlocks))
    # saveImage = True
    saveImage = False
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
    drawState = DrawStateEnvMADDPG(FPS, screen, viewRatio, circleColorSpace, circleSizeSpace, agentIdsToDraw,
                                   positionIndex,
                                   saveImage, saveImageDir, drawBackground, updateColorSpaceByPosterior,
                                   drawCircleOutside)

    # MDP Env
    interpolateState = None

    stateIndexInTimeStep = 0
    actionIndexInTimeStep = 1
    posteriorIndexInTimeStep = None
    chaseTrial = ChaseTrialWithTraj(stateIndexInTimeStep, drawState, interpolateState, actionIndexInTimeStep,
                                    posteriorIndexInTimeStep)

    # print(len(trajectories))
    lens = [len(trajectory) for trajectory in trajectories]
    maxWolfPositions = np.array([max([max([max(abs(timeStep[0][wolfId][0]), abs(timeStep[0][wolfId][1]))
                                           for wolfId in range(numWolves)])
                                      for timeStep in trajectory])
                                 for trajectory in trajectories])
    flags = maxWolfPositions < 1.3 * viewRatio
    index = flags.nonzero()[0]
    print(trajectories[0][1])
    [chaseTrial(trajectory) for trajectory in np.array(trajectories)[index[[0, 2, 3]]]]


if __name__ == '__main__':
    main()
