import numpy as np
import math

class IsTerminal():
    def __init__(self, sheepId, wolfId, minDistance):
        self.sheepId = sheepId
        self.wolfId = wolfId
        self.minDistance = minDistance

    def __call__(self, state):
        currentAllAgentsPositions, currentAllVelocities = state
        sheepPosition = currentAllAgentsPositions[self.sheepId]
        wolfPosition = currentAllAgentsPositions[self.wolfId]
        sheepWolfDistance = np.linalg.norm((np.array(sheepPosition) - np.array(wolfPosition)), ord=2)
        terminal = (sheepWolfDistance <= self.minDistance)
        return terminal

class IsLegalInitPositions():
    def __init__(self, sheepId, wolfId, distractorsIds, minSheepWolfDistance, minSheepDistractorDistance):
        self.sheepId = sheepId
        self.wolfId = wolfId
        self.distractorsIds = distractorsIds
        self.minSheepWolfDistance = minSheepWolfDistance
        self.minSheepDistractorDistance = minSheepDistractorDistance

    def __call__(self, initPositions):
        sheepPosition = initPositions[self.sheepId]
        wolfPosition = initPositions[self.wolfId]
        distractorsPositions = [initPositions[id] for id in self.distractorsIds]
        sheepWolfDistance = np.linalg.norm((np.array(sheepPosition) - np.array(wolfPosition)), ord=2)
        sheepDistractorsDistances = [np.linalg.norm((np.array(sheepPosition) - np.array(distractorPosition)), ord=2) 
                for distractorPosition in distractorsPositions]
        legalSheepWolf = (sheepWolfDistance > self.minSheepWolfDistance)
        legalSheepDistractors = np.all([(sheepDistractorDistance > self.minSheepDistractorDistance) for sheepDistractorDistance in sheepDistractorsDistances])  
        legal = legalSheepWolf and legalSheepDistractors
        return legal

class ResetState():
    def __init__(self, xBoundary, yBoundary, numOfAgent, isLegalInitPositions, transPolarToCartesian):
        self.xMin, self.xMax = xBoundary
        self.yMin, self.yMax = yBoundary
        self.numOfAgnet = numOfAgent
        self.isLegalInitPositions = isLegalInitPositions
        self.transPolarToCartesian = transPolarToCartesian

    def __call__(self):
        initAllAgentsVelocities = [self.transPolarToCartesian(np.random.uniform(-math.pi, math.pi)) for agentId in range(self.numOfAgnet)]
        initAllAgentsPositions = [[np.random.uniform(self.xMin, self.xMax),
                      np.random.uniform(self.yMin, self.yMax)]
                     for _ in range(self.numOfAgnet)]
        while not self.isLegalInitPositions(initAllAgentsPositions):
            initAllAgentsPositions = [[np.random.uniform(self.xMin, self.xMax),
                          np.random.uniform(self.yMin, self.yMax)]
                         for _ in range(self.numOfAgnet)] 
        
        initState = np.array([initAllAgentsPositions, initAllAgentsVelocities])
        return initState

class PrepareSheepVelocity():
    def __init__(self, minSheepSpeed, maxSheepSpeed, warmUpTimeSteps):
        self.minSheepSpeed = minSheepSpeed
        self.maxSheepSpeed = maxSheepSpeed
        self.warmUpTimeSteps = warmUpTimeSteps

    def __call__(self, sheepAction, timeStep):
        warmUpRate = min(1, timeStep/self.warmUpTimeSteps)
        sheepSpeed = self.minSheepSpeed + (self.maxSheepSpeed - self.minSheepSpeed) * warmUpRate
        sheepVelocity = np.array(sheepAction) * sheepSpeed 
        return sheepVelocity

class PrepareWolfVelocity():
    def __init__(self, minWolfSpeed, maxWolfSpeed, warmUpTimeSteps, wolfSubtlety, transCartesianToPolar, transPolarToCartesian):
        self.minWolfSpeed = minWolfSpeed
        self.maxWolfSpeed = maxWolfSpeed
        self.warmUpTimeSteps = warmUpTimeSteps
        self.wolfSubtlety = wolfSubtlety
        self.transCartesianToPolar = transCartesianToPolar
        self.transPolarToCartesian = transPolarToCartesian

    def __call__(self, sheepPosition, wolfPosition, timeStep):
        heatSeekingDirectionPolar = self.transCartesianToPolar(sheepPosition - wolfPosition)
        wolfDirectionPolar = np.random.vonmises(heatSeekingDirectionPolar, self.wolfSubtlety) 
        wolfDirection = self.transPolarToCartesian(wolfDirectionPolar)
        
        warmUpRate = min(1, timeStep/self.warmUpTimeSteps)
        wolfSpeed = self.minWolfSpeed + (self.maxWolfSpeed - self.minWolfSpeed) * warmUpRate
        wolfVelocity = wolfSpeed * wolfDirection

        return wolfVelocity

class PrepareDistractorVelocity():
    def __init__(self, minDistractorSpeed, maxDistractorSpeed, warmUpTimeSteps, transCartesianToPolar, transPolarToCartesian):
        self.minDistractorSpeed = minDistractorSpeed
        self.maxDistractorSpeed = maxDistractorSpeed
        self.warmUpTimeSteps = warmUpTimeSteps
        self.transCartesianToPolar = transCartesianToPolar
        self.transPolarToCartesian = transPolarToCartesian

    def __call__(self, lastDistractorVelocity, timeStep):
        oldDistractorDirectionPolar = self.transCartesianToPolar(lastDistractorVelocity)
        distractorDirectionPolar = np.random.uniform(-math.pi*1/3, math.pi*1/3) + oldDistractorDirectionPolar 
        distractorDirection = self.transPolarToCartesian(distractorDirectionPolar)
        
        warmUpRate = min(1, timeStep/self.warmUpTimeSteps)
        distractorSpeed = self.minDistractorSpeed + (self.maxDistractorSpeed - self.minDistractorSpeed) * warmUpRate
        distractorVelocity = distractorSpeed * distractorDirection
        return distractorVelocity

class PrepareAllAgentsVelocities():
    def __init__(self, sheepId, wolfId, distractorsIds, prepareSheepVelocity, prepareWolfVelocity, prepareDistractorVelocity):
        self.sheepId = sheepId
        self.wolfId = wolfId
        self.distractorsIds = distractorsIds
        self.prepareSheepVelocity = prepareSheepVelocity
        self.prepareWolfVelocity = prepareWolfVelocity
        self.prepareDistractorVelocity = prepareDistractorVelocity

    def __call__(self, state, action, timeStep):

        currentAllAgentsPositions, lastAllAgentsVelocities = state
        sheepPosition = currentAllAgentsPositions[self.sheepId]
        wolfPosition = currentAllAgentsPositions[self.wolfId]
        lastDistractorsVelocities = [lastAllAgentsVelocities[id] for id in self.distractorsIds]

        sheepVelocity = self.prepareSheepVelocity(action, timeStep)
        wolfVelocity = self.prepareWolfVelocity(sheepPosition, wolfPosition, timeStep)
        currentAllAgentsVelocities = [self.prepareDistractorVelocity(lastDistractorVelocity, timeStep) for lastDistractorVelocity in lastDistractorsVelocities]
        
        currentAllAgentsVelocities.insert(self.sheepId, sheepVelocity)
        currentAllAgentsVelocities.insert(self.wolfId, wolfVelocity)
        return currentAllAgentsVelocities

class StayInBoundaryByReflectVelocity():
    def __init__(self, xBoundary, yBoundary):
        self.xMin, self.xMax = xBoundary
        self.yMin, self.yMax = yBoundary

    def __call__(self, position, velocity):
        adjustedX, adjustedY = position
        adjustedVelX, adjustedVelY = velocity
        if position[0] >= self.xMax:
            adjustedX = 2 * self.xMax - position[0]
            adjustedVelX = -velocity[0]
        if position[0] <= self.xMin:
            adjustedX = 2 * self.xMin - position[0]
            adjustedVelX = -velocity[0]
        if position[1] >= self.yMax:
            adjustedY = 2 * self.yMax - position[1]
            adjustedVelY = -velocity[1]
        if position[1] <= self.yMin:
            adjustedY = 2 * self.yMin - position[1]
            adjustedVelY = -velocity[1]
        checkedPosition = np.array([adjustedX, adjustedY])
        checkedVelocity = np.array([adjustedVelX, adjustedVelY])
        return checkedPosition, checkedVelocity


class TransitWithInterpolation:
    def __init__(self, wolfSubtlety, numFramesToInterpolate, prepareAllAgentsVelocities, stayInBoundaryByReflectVelocity, isTerminal):
        self.timeStep = 0
        self.numFramesToInterpolate = numFramesToInterpolate
        self.prepareAllAgentsVelocities = prepareAllAgentsVelocities
        self.stayInBoundaryByReflectVelocity = stayInBoundaryByReflectVelocity
        self.isTerminal = isTerminal

    def __call__(self, state, action):

        currentAllPositions, lastAllVelocities = state
        currentAllVelocities = self.prepareAllAgentsVelocities(state, action, self.timeStep)
        currentAllVelocitiesForInterpolation = np.array(currentAllVelocities) / (self.numFramesToInterpolate + 1)

        for frameIndex in range(self.numFramesToInterpolate + 1):
            noBoundaryNextPositions = np.array(currentAllPositions) + np.array(currentAllVelocitiesForInterpolation)

            checkedNextPositionsAndVelocities = [self.stayInBoundaryByReflectVelocity(
                position, velocity) for position, velocity in zip(noBoundaryNextPositions, currentAllVelocitiesForInterpolation)]
            nextAllPositions, nextAllVelocitiesForInterpolation = list(zip(*checkedNextPositionsAndVelocities))

            nextState = np.array([nextAllPositions, currentAllVelocitiesForInterpolation]) 
            if self.isTerminal(nextState):
                break

            currentAllPositions = nextAllPositions
            currentAllVelocitiesForInterpolation = nextAllVelocitiesForInterpolation
        
        self.timeStep = self.timeStep + 1
        return nextState
