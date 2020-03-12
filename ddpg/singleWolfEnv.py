import numpy as np
import pandas as pd
import random

import pygame
from pygame.color import THECOLORS
from pygame.locals import *


def l2Norm(s0, s1, rho=1):
    diff = (np.asarray(s0) - np.asarray(s1)) * rho
    return np.linalg.norm(diff)

def isTerminal(state):
    agentState = state[:4]
    wolfState = state[4:8]

    agentCoordinates = agentState[:2]
    wolfCoordinates = wolfState[:2]

    if l2Norm(agentCoordinates, wolfCoordinates) <= 30:
        return True
    return False

def beliefArrayToDataFrame(numberObjects, oldBelief, assumeWolfPrecisionList):
    multiIndex = pd.MultiIndex.from_product([assumeWolfPrecisionList, range(
        1, numberObjects)], names=['assumeChasingPrecision', 'Identity'])
    oldBeliefList = list(oldBelief)
    beliefDF = pd.DataFrame(
        oldBeliefList, index=multiIndex, columns=['p'])
    return beliefDF

def positionArraytToDataFrame(numberObjects, oldPosition):
    oldPositionList = oldPosition.tolist()
    oldPositionList = [oldPositionList[i:i+4] for i in range(0,len(oldPositionList),4)]
    oldPositionDF = pd.DataFrame(oldPositionList,index=list(range(numberObjects)),
    columns=['positionX','positionY','velocityX','velocityY'])
    return oldPositionDF

def attentionArraytToDataFrame(oldAttentionStatus, oldBelief, attentionLimitation):
    attentionStatusList = oldAttentionStatus.tolist()
    attentionStatusDF = pd.DataFrame(
        attentionStatusList, index=oldBelief.index, columns=['attentionStatus'])
    return attentionStatusDF

def renormalVector(rawVector, targetLength):
    rawLength = np.power(np.power(rawVector, 2).sum(), 0.5)
    changeRate = np.divide(targetLength, rawLength)
    return np.multiply(rawVector, changeRate)

class Reset():
    def __init__(self, numberObjects, initialPosition, movingRange, speedList):
        self.initialPosition = initialPosition
        self.movingRange = movingRange
        self.speedList = speedList 
        self.numberObjects = numberObjects

    def __call__(self):
        initPositionList = self.initialPosition(self.numberObjects)
        statesList = []
        initVelocity = [0,0]
        for initPosition in initPositionList:
            statesList.append(initPosition + initVelocity)

        initState = pd.DataFrame(statesList,index=list(range(self.numberObjects)),
            columns=['positionX','positionY','velocityX','velocityY'])

        initPhysicalState = np.asarray(initState).flatten()

        return initPhysicalState

class Transition():
    def __init__(self, movingRange, speedList, numberObjects, wolfPolicy, wolfPrecision, renderOn=False):
        self.movingRange = movingRange
        self.speedList = speedList
        self.renderOn = renderOn
        self.numberObjects = numberObjects
        self.wolfPolicy = wolfPolicy
        self.wolfPrecision = wolfPrecision

    def __call__(self, oldState, action):

        oldState = positionArraytToDataFrame(self.numberObjects, oldState)

        wolfAction = self.wolfPolicy(oldState, self.wolfPrecision)
        actionForTransition = [action, wolfAction]
        
        newState = self.physicalTransition(oldState, actionForTransition)

        newState = np.asarray(newState).flatten()

        return newState

    def physicalTransition(self, currentStates, currentActions):
        currentPositions = currentStates.loc[:][[
            'positionX', 'positionY']].values
        currentVelocities = currentStates.loc[:][[
            'velocityX', 'velocityY']].values
        numberObjects = len(currentStates.index)

        newVelocities = [renormalVector(np.add(currentVelocities[i], np.divide(
            currentActions[i], 2.0)), self.speedList[i]) for i in range(numberObjects)]

        # sheep no renormal
        # newVelocities[0] = currentActions[0]
        # print(newVelocities)
        newPositions = [np.add(currentPositions[i], newVelocities[i])
                        for i in range(numberObjects)]

        for i in range(numberObjects):
            if newPositions[i][0] > self.movingRange[2]:
                newPositions[i][0] = 2 * \
                    self.movingRange[2] - newPositions[i][0]
            if newPositions[i][0] < self.movingRange[0]:
                newPositions[i][0] = 2 * \
                    self.movingRange[0] - newPositions[i][0]
            if newPositions[i][1] > self.movingRange[3]:
                newPositions[i][1] = 2 * \
                    self.movingRange[3] - newPositions[i][1]
            if newPositions[i][1] < self.movingRange[1]:
                newPositions[i][1] = 2 * \
                    self.movingRange[1] - newPositions[i][1]

        newVelocities = [newPositions[i] - currentPositions[i]
                         for i in range(numberObjects)]
        newPhysicalStatesList = [list(newPositions[i]) + list(newVelocities[i])
                         for i in range(numberObjects)]
        newPhysicalStates = pd.DataFrame(
            newPhysicalStatesList, index=currentStates.index, columns=currentStates.columns)

        if self.renderOn:
            currentPositions = [list(currentPositions[i]) for i in range(numberObjects)]
            agentCoordinates = list(map(int, currentPositions[0]))
            wolfCoordinates = list(map(int, currentPositions[1]))

            pygame.init()
            screenSize = [self.movingRange[2],self.movingRange[3]]
            screen = pygame.display.set_mode(screenSize)
            circleR = 10
            screen.fill([0,0,0])
            color = [THECOLORS['green'],THECOLORS['red']] + [THECOLORS['blue']] * (numberObjects-2)
            positionList = [agentCoordinates, wolfCoordinates]

            for drawposition in positionList:
                pygame.draw.circle(screen,color[int(positionList.index(drawposition))],drawposition,circleR)
            pygame.display.flip()
            # pygame.time.wait(0.2)

            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()

        return newPhysicalStates


if __name__ == '__main__':
	main()