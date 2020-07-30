import sys
import os
dirName = os.path.dirname(__file__)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.append(os.path.join(dirName, '.. .. ..'))

import subprocess
import numpy as np 
import pandas as pd
import pygame as pg
from pygame.color import THECOLORS
from visualize.drawDemo import DrawBackground,  DrawCircleOutsideEnvMADDPG, DrawStateEnvMADDPG, ChaseTrialWithTraj
from functionTools.loadSaveModel import loadFromPickle


class CompileDemosFromPics:
    def __init__(self, conditionFolderName, getConditionName, videoSavePath):
        self.conditionFolderName = conditionFolderName
        self.getConditionName = getConditionName
        self.videoSavePath = videoSavePath

    def __call__(self, df):
        sensitiveZoneRadius = df.index.get_level_values('sensitiveZoneRadius')[0]
        oneWolfSelfHurtProb = df.index.get_level_values('oneWolfSelfHurtProb')[0]
        wolfIndividual = df.index.get_level_values('wolfIndividual')[0]

        numWolves = 3
        numSheeps = 1
        numBlocks = 2
        maxTimeStep = 75
        maxEpisode = 60000
        sheepSpeedMultiplier = 1.0
        costActionRatio = 0.0

        conditionName = self.getConditionName(numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier,
                                              costActionRatio, oneWolfSelfHurtProb, sensitiveZoneRadius, wolfIndividual)
        picsFolderPath = os.path.join(dirName, '..', 'trajectories', self.conditionFolderName, conditionName, 'forDemo')
        picsPath = os.path.join(picsFolderPath, '%d.png')
        outputVideoPath = os.path.join(self.videoSavePath, conditionName + '.mp4')
        createVid = 'ffmpeg -r 10 -f image2 -s 1920x1080 -i ' + picsPath + ' -vcodec libx264 -crf 25 -pix_fmt yuv420p ' + outputVideoPath
        subprocess.run(createVid, shell=True)


class DrawDemos:
    def __init__(self, conditionFolderName, getConditionName):
        self.conditionFolderName = conditionFolderName
        self.getConditionName = getConditionName

    def __call__(self, df):
        sensitiveZoneRadius = df.index.get_level_values('sensitiveZoneRadius')[0]
        oneWolfSelfHurtProb = df.index.get_level_values('oneWolfSelfHurtProb')[0]
        wolfIndividual = df.index.get_level_values('wolfIndividual')[0]

        numWolves = 3
        numSheeps = 1
        numBlocks = 2
        maxTimeStep = 75
        maxEpisode = 60000
        sheepSpeedMultiplier = 1.0
        costActionRatio = 0.0

        conditionName = self.getConditionName(numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier,
                                              costActionRatio, oneWolfSelfHurtProb, sensitiveZoneRadius, wolfIndividual)
        imageSavePath = os.path.join(dirName, '..', 'trajectories', self.conditionFolderName, conditionName)
        if not os.path.exists(imageSavePath):
            os.makedirs(imageSavePath)
        trajFileName = conditionName + '_traj'
        trajectoryDir = os.path.join(dirName, '..', 'trajectories', self.conditionFolderName, trajFileName)
        trajList = loadFromPickle(trajectoryDir)

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

        saveImage = True
        sensitiveZoneSize = int(sensitiveZoneRadius * screenWidth / (2 * viewRatio))
        numAgents = numWolves + numSheeps
        sheepsID = list(range(numWolves, numAgents))
        drawState = DrawStateEnvMADDPG(FPS, screen, viewRatio, circleColorSpace, circleSizeSpace, agentIdsToDraw,
                                       positionIndex, saveImage, saveImageDir, sheepsID, sensitiveZoneSize,
                                       drawBackground, updateColorSpaceByPosterior, drawCircleOutside)
        # MDP Env
        interpolateState = None
        stateIndexInTimeStep = 0
        actionIndexInTimeStep = 1
        posteriorIndexInTimeStep = None
        chaseTrial = ChaseTrialWithTraj(stateIndexInTimeStep, drawState, interpolateState, actionIndexInTimeStep,
                                        posteriorIndexInTimeStep)
        maxWolfPositions = np.array([max([max([max(abs(timeStep[0][wolfId][0]), abs(timeStep[0][wolfId][1]))
                                               for wolfId in range(numWolves)])
                                          for timeStep in trajectory])
                                     for trajectory in trajList])
        flags = maxWolfPositions < 1.3 * viewRatio
        index = flags.nonzero()[0]
        [chaseTrial(trajectory) for trajectory in np.array(trajList)[index]]


def main():
    getConditionName = lambda numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, \
        costActionRatio, oneWolfSelfHurtProb, sensitiveZoneRadius, wolfIndividual: \
        "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}WolfActCost{}HurtProb{}Radius{}{}".format(
            numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, costActionRatio, 
            oneWolfSelfHurtProb, sensitiveZoneRadius, wolfIndividual)

    # draw Demos, generate pictures
    conditionFolderName = '3wolvesMaddpgWithProbOfHurtBySheep'
    drawDemos = DrawDemos(conditionFolderName, getConditionName)

    independentVariables = dict()
    independentVariables['wolfIndividual'] = ['shared', 'individ']
    independentVariables['sensitiveZoneRadius'] = [0.5, 0.75]
    independentVariables['oneWolfSelfHurtProb'] = [0.0, 0.2]

    levelNames = list(independentVariables.keys())
    levelValues = list(independentVariables.values())
    levelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=levelIndex)
    toSplitFrame.groupby(levelNames).apply(drawDemos)

    # compile pictures to videos
    videoSavePath = os.path.join(dirName, '..', 'trajectories', 'videos')
    if not os.path.exists(videoSavePath):
        os.makedirs(videoSavePath)
    compileDemosFromPics = CompileDemosFromPics(conditionFolderName, getConditionName, videoSavePath)
    toSplitFrame.groupby(levelNames).apply(compileDemosFromPics)

if __name__ == '__main__':
    main()
