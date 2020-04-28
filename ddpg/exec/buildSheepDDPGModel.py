import os
import numpy as np
import tensorflow as tf

from src.ddpg import BuildActorModel, BuildCriticModel, actByPolicyTrain, \
    actByPolicyTarget, evaluateCriticTarget, TrainCritic, getActionGradients, TrainActor, \
    AddToMemory, initializeMemory, UpdateModelsByMiniBatch, RunDDPGTimeStep, RunEpisode, DDPG,\
    UpdateParameters, TrainCriticBySASRQ, TrainActorFromGradients, TrainActorOneStep
from src.reward import RewardFunctionCompete
from src.policy import AddActionNoise, HeatSeekingContinuousDeterministicPolicy, \
    ActByDDPG2D, ActOneStepWithNoise2D
from src.envNoPhysics import Reset, TransitForNoPhysics, StayWithinBoundary, \
    TransitWithSingleWolf, GetAgentPosFromState, IsTerminal

from src.continuousVisualization import *
from src.traj import *
from pygame.color import THECOLORS

np.random.seed(1)
tf.set_random_seed(1)

visualize = True

class GetSavePath:
    def __init__(self, dataDirectory, extension, fixedParameters={}):
        self.dataDirectory = dataDirectory
        self.extension = extension
        self.fixedParameters = fixedParameters

    def __call__(self, parameters):
        allParameters = dict(list(parameters.items()) + list(self.fixedParameters.items()))
        sortedParameters = sorted(allParameters.items())
        nameValueStringPairs = [parameter[0] + '=' + str(parameter[1]) for parameter in sortedParameters]

        fileName = '_'.join(nameValueStringPairs) + self.extension
        fileName = fileName.replace(" ", "")

        path = os.path.join(self.dataDirectory, fileName)
        return path


def saveVariables(model, path):
    graph = model.graph
    saver = graph.get_collection_ref("saver")[0]
    saver.save(model, path)
    print("Model saved in {}".format(path))


def main():
    numAgents = 2
    numStateSpace = numAgents * 2
    actionLow = -1
    actionHigh = 1
    actionRange = 1
    actionDim = 2

    buildActorModel = BuildActorModel(numStateSpace, actionDim, actionRange)
    actorTrainingLayerWidths = [64, 64, 64]
    # actorTrainingLayerWidths = [30]
    actorTargetLayerWidths = actorTrainingLayerWidths
    actorWriter, actorModel = buildActorModel(actorTrainingLayerWidths, actorTargetLayerWidths)

    buildCriticModel = BuildCriticModel(numStateSpace, actionDim)
    criticTrainingLayerWidths = [64, 64, 64]
    # criticTrainingLayerWidths = [30]
    criticTargetLayerWidths = criticTrainingLayerWidths
    criticWriter, criticModel = buildCriticModel(criticTrainingLayerWidths, criticTargetLayerWidths)

    sheepId = 0
    wolfId = 1
    getSheepXPos = GetAgentPosFromState(sheepId)
    getWolfXPos = GetAgentPosFromState(wolfId)
    killzoneRadius = 1
    isTerminal = IsTerminal(getWolfXPos, getSheepXPos, killzoneRadius)

    maxRunningSteps = 20
    sheepAliveBonus = 1 / maxRunningSteps

    tau = 0.01 #
    updateParameters = UpdateParameters(tau)

    learningRateCritic = 0.01 #
    gamma = 0.95 #
    trainCriticBySASRQ = TrainCriticBySASRQ(learningRateCritic, gamma, criticWriter)
    trainCritic = TrainCritic(actByPolicyTarget, evaluateCriticTarget, trainCriticBySASRQ)

    learningRateActor = 0.01#
    trainActorFromGradients = TrainActorFromGradients(learningRateActor, actorWriter)
    trainActorOneStep = TrainActorOneStep(actByPolicyTrain, trainActorFromGradients, getActionGradients)
    trainActor = TrainActor(trainActorOneStep)

    updateModelsByMiniBatch = UpdateModelsByMiniBatch(trainActor, trainCritic)
    wolfSpeed = 3
    wolfPolicy = HeatSeekingContinuousDeterministicPolicy(getWolfXPos, getSheepXPos, wolfSpeed)

    xBoundary = (0, 20)
    yBoundary = (0, 20)
    stayWithinBoundary = StayWithinBoundary(xBoundary, yBoundary)
    physicalTransition = TransitForNoPhysics(stayWithinBoundary)
    transitionFunction = TransitWithSingleWolf(physicalTransition, wolfPolicy)

    actionNoise = 0.1
    noiseDecay = 0.999
    memoryCapacity = 1000000 #
    addActionNoise = AddActionNoise(actionNoise, noiseDecay, actionLow, actionHigh, memoryCapacity)
    sheepTerminalPenalty = -10
    rewardSheep = RewardFunctionCompete(sheepAliveBonus, sheepTerminalPenalty, isTerminal)

    actOneStepWithNoise = ActOneStepWithNoise2D(actByPolicyTrain, addActionNoise, transitionFunction, rewardSheep, isTerminal)
    minibatchSize = 1024 #
    rewardScalingFactor = 1
    addToMemory = AddToMemory(rewardScalingFactor)

    paramUpdateInterval = 100
    runDDPGTimeStep = RunDDPGTimeStep(actOneStepWithNoise, addToMemory, updateModelsByMiniBatch,
                                      updateParameters, minibatchSize, memoryCapacity, paramUpdateInterval)

    maxTimeStep = 25 #
    reset = Reset(xBoundary, yBoundary, numAgents)
    runEpisode = RunEpisode(reset, runDDPGTimeStep, maxTimeStep)

    maxEpisode = 5000#
    ddpg = DDPG(initializeMemory, runEpisode, memoryCapacity, maxEpisode)
    trainedActorModel, trainedCriticModel = ddpg(actorModel, criticModel)

    modelIndex = 0
    actorFixedParam = {'actorModel': modelIndex}
    criticFixedParam = {'criticModel': modelIndex}
    parameters = {'wolfSpeed':wolfSpeed, 'dimension': actionDim, 'maxEpisode': maxEpisode, 'maxTimeStep': maxTimeStep, 'minibatchSize': minibatchSize, 'gamma': gamma,
                                 'learningRateActor': learningRateActor, 'learningRateCritic': learningRateCritic}

    modelSaveDirectory = "../trainedDDPGModels"
    modelSaveExtension = '.ckpt'
    getActorSavePath = GetSavePath(modelSaveDirectory, modelSaveExtension, actorFixedParam)
    getCriticSavePath = GetSavePath(modelSaveDirectory, modelSaveExtension, criticFixedParam)
    savePathActor = getActorSavePath(parameters)
    savePathCritic = getCriticSavePath(parameters)

    with actorModel.as_default():
        saveVariables(trainedActorModel, savePathActor)
    with criticModel.as_default():
        saveVariables(trainedCriticModel, savePathCritic)

    if visualize:
        sheepPolicy = ActByDDPG2D(actByPolicyTrain, actorModel)
        transit = TransitForNoPhysics(stayWithinBoundary)
        policy = lambda state: list(sheepPolicy(state)) + list(wolfPolicy(state))

        # reset = lambda: [14.71807246,  2.11855376,  9.30940919,  6.51058129]
        sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset)
        trajectory = sampleTrajectory(policy)

        dirName = os.path.dirname(__file__)
        dataPath = os.path.join(dirName, '..', 'trajectory', 'newParams' + '.pickle')
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
        imageFolderName = 'Demo5000Steps'
        saveImageDir = os.path.join(os.path.join(parentDir, 'demo'), imageFolderName)
        if not os.path.exists(saveImageDir):
            os.makedirs(saveImageDir)

        chaseTrial(numberOfAgents, positionListToDraw, saveImageDir)


if __name__ == '__main__':
    main()

