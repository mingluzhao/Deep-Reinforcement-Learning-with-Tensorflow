import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))
sys.path.append(os.path.join(dirName, '..'))

import random
import numpy as np
import pickle
from collections import OrderedDict
import pandas as pd
from matplotlib import pyplot as plt
import itertools as it
import math
import pygame as pg
from pygame.color import THECOLORS

#from src.visualization.drawDemo import DrawBackground, DrawCircleOutside, DrawState, ChaseTrialWithTraj, InterpolateState
from environment.dontGetCaught.analyticGeometryFunctions import transCartesianToPolar, transPolarToCartesian
from environment.dontGetCaught.MDPChasing.env import IsTerminal, IsLegalInitPositions, ResetState, PrepareSheepVelocity, PrepareWolfVelocity, PrepareDistractorVelocity, \
PrepareAllAgentsVelocities, StayInBoundaryByReflectVelocity, TransitWithInterpolation
from environment.dontGetCaught.MDPChasing.reward import RewardFunctionTerminalPenalty
from environment.dontGetCaught.MDPChasing.policies import RandomPolicy
from environment.dontGetCaught.chooseFromDistribution import sampleFromDistribution, maxFromDistribution
from environment.dontGetCaught.trajectory import ForwardOneStep, SampleTrajectory
from environment.dontGetCaught.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle

def composeFowardOneTimeStepWithRandomSubtlety(numOfAgent):
    # one time step used in different algorithms; here evaluate number of agent
    # MDP 
    
    # experiment parameter for env
    numMDPTimeStepPerSecond = 5 #  change direction every 200ms 
    distanceToVisualDegreeRatio = 20

    minSheepSpeed = int(17.4 * distanceToVisualDegreeRatio/numMDPTimeStepPerSecond)
    maxSheepSpeed = int(23.2 * distanceToVisualDegreeRatio/numMDPTimeStepPerSecond)
    warmUpTimeSteps = 10 * numMDPTimeStepPerSecond # 10s to warm up
    prepareSheepVelocity = PrepareSheepVelocity(minSheepSpeed, maxSheepSpeed, warmUpTimeSteps)
    
    minWolfSpeed = int(8.7 * distanceToVisualDegreeRatio/numMDPTimeStepPerSecond)
    maxWolfSpeed = int(14.5 * distanceToVisualDegreeRatio/numMDPTimeStepPerSecond)
    wolfSubtleties = [500, 11, 3.3, 1.83, 0.92, 0.31, 0.001] # 0, 30, 60, .. 180
    initWolfSubtlety = np.random.choice(wolfSubtleties)
    prepareWolfVelocity = PrepareWolfVelocity(minWolfSpeed, maxWolfSpeed, warmUpTimeSteps, initWolfSubtlety, transCartesianToPolar, transPolarToCartesian)
    
    minDistractorSpeed = int(8.7 * distanceToVisualDegreeRatio/numMDPTimeStepPerSecond)
    maxDistractorSpeed = int(14.5 * distanceToVisualDegreeRatio/numMDPTimeStepPerSecond)
    prepareDistractorVelocity = PrepareDistractorVelocity(maxDistractorSpeed, maxDistractorSpeed, warmUpTimeSteps, transCartesianToPolar, transPolarToCartesian)
    
    sheepId = 0
    wolfId = 1
    distractorsIds = list(range(2, numOfAgent))
    prepareAllAgentsVelocities = PrepareAllAgentsVelocities(sheepId, wolfId, distractorsIds, prepareSheepVelocity, prepareWolfVelocity, prepareDistractorVelocity)

    xBoundary = [0, 600]
    yBoundary = [0, 600]
    stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
    
    killzoneRadius = 2.5 * distanceToVisualDegreeRatio
    isTerminal = IsTerminal(sheepId, wolfId, killzoneRadius)
 
    numFramePerSecond = 30 # visual display fps
    numFramesToInterpolate = int(numFramePerSecond / numMDPTimeStepPerSecond - 1) # interpolate each MDP timestep to multiple frames; check terminal for each frame

    transitFunction = TransitWithInterpolation(initWolfSubtlety, numFramesToInterpolate, prepareAllAgentsVelocities, stayInBoundaryByReflectVelocity, isTerminal)
    
    aliveBonus = 0.01
    deathPenalty = -1
    rewardFunction = RewardFunctionTerminalPenalty(aliveBonus, deathPenalty, isTerminal)

    forwardOneStep = ForwardOneStep(transitFunction, rewardFunction)

    return forwardOneStep


class SampleTrajectoriesForCoditions:
    # how to run episode/trajectory is different in algorithms, here is a simple example to run episodes with fixed policy
    def __init__(self, numTrajectories, composeFowardOneTimeStepWithRandomSubtlety):
        self.numTrajectories = numTrajectories
        self.composeFowardOneTimeStepWithRandomSubtlety = composeFowardOneTimeStepWithRandomSubtlety
    
    def __call__(self, parameters):
        numOfAgent = parameters['numOfAgent']
        trajectories = []
        for trajectoryId in range(self.numTrajectories):
            
            forwardOneStep = self.composeFowardOneTimeStepWithRandomSubtlety(numOfAgent)
            
            sheepId = 0
            wolfId = 1
            distractorsIds = list(range(2, numOfAgent))
            distanceToVisualDegreeRatio = 20
            minInitSheepWolfDistance = 9 * distanceToVisualDegreeRatio
            minInitSheepDistractorDistance = 2.5 * distanceToVisualDegreeRatio  # no distractor in killzone when init
            isLegalInitPositions = IsLegalInitPositions(sheepId, wolfId, distractorsIds, minInitSheepWolfDistance, minInitSheepDistractorDistance)
            xBoundary = [0, 600]
            yBoundary = [0, 600]
            resetState = ResetState(xBoundary, yBoundary, numOfAgent, isLegalInitPositions, transPolarToCartesian)
            
            killzoneRadius = 2.5 * distanceToVisualDegreeRatio
            isTerminal = IsTerminal(sheepId, wolfId, killzoneRadius)
            
            numMDPTimeStepPerSecond = 5  
            maxRunningSteps = 25 * numMDPTimeStepPerSecond
            sampleTrajecoty = SampleTrajectory(maxRunningSteps, isTerminal, resetState, forwardOneStep)
            
            numActionDirections = 8
            actionSpace = [(np.cos(directionId * 2 * math.pi / numActionDirections), np.sin(directionId * 2 * math.pi / numActionDirections)) for directionId in range(numActionDirections)]
            randomPolicy = RandomPolicy(actionSpace)
            sampleAction = lambda state: sampleFromDistribution(randomPolicy(state))
             
            trajectory = sampleTrajecoty(sampleAction)

            trajectories.append(trajectory)
        return trajectories


def main():
    manipulatedVariables = OrderedDict()
    manipulatedVariables['numOfAgent'] = [15, 25]
    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)
    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]
 
    numTrajectories = 2
    sampleTrajectoriesForConditions = SampleTrajectoriesForCoditions(numTrajectories, composeFowardOneTimeStepWithRandomSubtlety)
    trajectoriesMultipleConditions = [sampleTrajectoriesForConditions(para) for para in parametersAllCondtion]

if __name__ == '__main__':
    main()


