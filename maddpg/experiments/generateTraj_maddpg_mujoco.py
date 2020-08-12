import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import xmltodict
import mujoco_py as mujoco

from environment.chasingEnv.multiAgentEnv import *
from functionTools.loadSaveModel import saveToPickle, restoreVariables
from functionTools.trajectory import SampleTrajectory
from visualize.visualizeMultiAgent import *

from maddpg.maddpgAlgor.trainer.myMADDPG import *
import json
from environment.mujocoEnv.multiAgentMujocoEnv import TransitionFunction,ResetUniformWithoutXPos,SampleBlockState,IsOverlap

dirName = os.path.dirname(__file__)

maxEpisode = 60000
maxRunningStepsToSample = 75  # num of timesteps in one eps

def calcWolvesTrajReward(traj, wolvesID):
    rewardIDinTraj = 2
    getWolfReward = lambda allAgentsReward: np.sum([allAgentsReward[wolfID] for wolfID in wolvesID])
    rewardList = [getWolfReward(timeStepInfo[rewardIDinTraj]) for timeStepInfo in traj]
    trajReward = np.sum(rewardList)
    return trajReward

def main():
    debug = 1
    if debug:
        numWolves = 3
        numSheeps = 1
        numBlocks = 2
        hasWalls= 0.0

        dt=0.02
        maxTimeStep = 75
        sheepSpeedMultiplier = 1.0
        individualRewardWolf = 0

        mujocoVisualize=True
        saveAllmodels = True
        costActionRatio = 0.0

    else:
        print(sys.argv)
        condition = json.loads(sys.argv[1])
        numWolves = int(condition['numWolves'])
        numSheeps = int(condition['numSheeps'])
        numBlocks = int(condition['numBlocks'])
        hasWalls=float(condition['hasWalls'])

        dt =float(condition['dt'])
        maxTimeStep = int(condition['maxTimeStep'])
        sheepSpeedMultiplier = float(condition['sheepSpeedMultiplier'])
        individualRewardWolf = float(condition['individualRewardWolf'])
        costActionRatio = float(condition['costActionRatio'])
        mujocoVisualize=False

    print("maddpg: {} wolves, {} sheep, {} blocks, {} episodes with {} steps each eps, sheepSpeed: {}x, wolfIndividualReward: {}".format(numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, individualRewardWolf))

    dataMainFolder=os.path.join(dirName, '..', 'trainedModels', 'mujocoMADDPG')

    numAgents = numWolves + numSheeps
    numEntities = numAgents + numBlocks
    wolvesID = list(range(numWolves))
    sheepsID = list(range(numWolves, numAgents))
    blocksID = list(range(numAgents, numEntities))

    wolfSize = 0.075
    sheepSize = 0.05
    blockSize = 0.2
    entitiesSizeList = [wolfSize] * numWolves + [sheepSize] * numSheeps + [blockSize] * numBlocks

    collisionReward = 30 # originalPaper = 10*3
    isCollision = IsCollision(getPosFromAgentState)
    punishForOutOfBound =lambda state :0
    rewardSheep = RewardSheep(wolvesID, sheepsID, entitiesSizeList, getPosFromAgentState, isCollision, punishForOutOfBound, collisionReward)
    rewardWolf = RewardWolf(wolvesID, sheepsID, entitiesSizeList, isCollision, collisionReward, individualRewardWolf)
    reshapeAction = ReshapeAction()
    getActionCost = GetActionCost(costActionRatio, reshapeAction, individualCost=True)
    getWolvesAction = lambda action: [action[wolfID] for wolfID in wolvesID]
    rewardWolfWithActionCost = lambda state, action, nextState: np.array(rewardWolf(state, action, nextState)) - np.array(getActionCost(getWolvesAction(action)))

    rewardFunc = lambda state, action, nextState: \
        list(rewardWolfWithActionCost(state, action, nextState)) + list(rewardSheep(state, action, nextState))

    # ------------ mujocoEnv ------------------------

    physicsDynamicsPath = os.path.join(dirName, '..', '..', 'environment', 'mujocoEnv', 'dt={}'.format(dt),
                                       'hasWalls={}_numBlocks={}_numSheeps={}_numWolves={}.xml'.format(hasWalls, numBlocks, numSheeps, numWolves))

    with open(physicsDynamicsPath) as f:
        xml_string = f.read()
    envXmlDict = xmltodict.parse(xml_string.strip())
    envXml = xmltodict.unparse(envXmlDict)
    physicsModel = mujoco.load_model_from_xml(envXml)
    physicsSimulation = mujoco.MjSim(physicsModel)

    qPosInit = [0, 0] * numAgents
    qVelInit = [0, 0] * numAgents
    qVelInitNoise = 0
    qPosInitNoise = 0.8
    getBlockRandomPos = lambda: np.random.uniform(-0.7, +0.7, 2)

    # qVelInitNoise = 0*hasWalls
    # qPosInitNoise = 0.8*hasWalls
    # getBlockRandomPos = lambda: np.random.uniform(-0.7*hasWalls, +0.7*hasWalls, 2)
    getBlockSpeed = lambda: np.zeros(2)

    numQPos = len(physicsSimulation.data.qpos)
    numQVel = len(physicsSimulation.data.qvel)

    sampleAgentsQPos = lambda: np.asarray(qPosInit) + np.random.uniform(low=-qPosInitNoise, high=qPosInitNoise, size=numQPos)
    sampleAgentsQVel = lambda: np.asarray(qVelInit) + np.random.uniform(low=-qVelInitNoise, high=qVelInitNoise, size=numQVel)

    minDistance = 0.2 + 2 * blockSize  # >2*wolfSize+2*blockSize
    isOverlap = IsOverlap(minDistance)
    sampleBlockState = SampleBlockState(numBlocks, getBlockRandomPos, getBlockSpeed, isOverlap)

    reset = ResetUniformWithoutXPos(physicsSimulation, numAgents, numBlocks, sampleAgentsQPos, sampleAgentsQVel, sampleBlockState)

    transitTimePerStep = 0.1
    numSimulationFrames = int(transitTimePerStep / dt)

    isTerminal = lambda state: False
    transit = TransitionFunction(physicsSimulation, numAgents, numSimulationFrames, mujocoVisualize, isTerminal, reshapeAction)
    observeOneAgent = lambda agentID: Observe(agentID, wolvesID, sheepsID, blocksID, getPosFromAgentState, getVelFromAgentState)
    observe = lambda state: [observeOneAgent(agentID)(state) for agentID in range(numAgents)]
    initObsForParams = observe(reset())
    obsShape = [initObsForParams[obsID].shape[0] for obsID in range(len(initObsForParams))]
    sampleTrajectory = SampleTrajectory(maxRunningStepsToSample, transit, isTerminal, rewardFunc, reset)

    worldDim = 2
    actionDim = worldDim * 2 + 1
    layerWidth = [128, 128]

    # ------------ model ------------------------
    buildMADDPGModels = BuildMADDPGModels(actionDim, numAgents, obsShape)
    modelsList = [buildMADDPGModels(layerWidth, agentID) for agentID in range(numAgents)]

    fileName = "maddpg{}wall{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}WolfActCost{}individ{}_agent".format(
        hasWalls, numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, costActionRatio, individualRewardWolf)
    modelPaths = [os.path.join(dataMainFolder, fileName + str(i) + '100eps') for i in range(numAgents)]

    [restoreVariables(model, path) for model, path in zip(modelsList, modelPaths)]

    actOneStepOneModel = ActOneStep(actByPolicyTrainNoisy)
    policy = lambda allAgentsStates: [actOneStepOneModel(model, observe(allAgentsStates)) for model in modelsList]

    rewardList = []
    numTrajToSample = 5
    trajList = []
    for i in range(numTrajToSample):
        traj = sampleTrajectory(policy)
        rew = calcWolvesTrajReward(traj, wolvesID)
        rewardList.append(rew)
        trajList.append(list(traj))

    meanTrajReward = np.mean(rewardList)
    seTrajReward = np.std(rewardList) / np.sqrt(len(rewardList) - 1)
    print('meanTrajReward', meanTrajReward, 'se ', seTrajReward) #69/5

    # trajSavePath = os.path.join(dirName, '..', 'trajectory', fileName)
    # saveToPickle(trajList, trajSavePath)

    # trajectoryDirectory = os.path.join(dirName, '..', 'trajectories', folderName)
    # if not os.path.exists(trajectoryDirectory):
    #     os.makedirs(trajectoryDirectory)
    # trajFileName = "maddpg{}wolves{}sheep{}blocks{}eps{}stepSheepSpeed{}{}Traj".format(
    #     numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, individStr)
    # trajSavePath = os.path.join(trajectoryDirectory, trajFileName)
    # saveToPickle(trajList, trajSavePath)


if __name__ == '__main__':
    main()