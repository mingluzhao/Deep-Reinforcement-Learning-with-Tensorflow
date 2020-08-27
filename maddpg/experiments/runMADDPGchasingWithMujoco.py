import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import numpy as np
import json
import xmltodict
import mujoco_py as mujoco

from maddpg.maddpgAlgor.trainer.myMADDPG import BuildMADDPGModels, TrainCritic, TrainActor, TrainCriticBySASR, \
    TrainActorFromSA, TrainMADDPGModelsWithBuffer, ActOneStep, actByPolicyTrainNoisy, actByPolicyTargetNoisyForNextState
from RLframework.RLrun import UpdateParameters, SampleOneStep, SampleFromMemory,\
    RunTimeStep, RunEpisode, RunAlgorithm, getBuffer, SaveModel, StartLearn
from functionTools.loadSaveModel import saveVariables
from environment.chasingEnv.multiAgentEnv import  ReshapeAction, RewardSheep, RewardWolf, Observe, GetCollisionForce, IntegrateState, \
    IsCollision, PunishForOutOfBound, getPosFromAgentState, getVelFromAgentState, GetActionCost
from environment.mujocoEnv.multiAgentMujocoEnv import TransitionFunction,ResetUniformWithoutXPos,SampleBlockState,IsOverlap

# fixed training parameters
maxEpisode = 60000
learningRateActor = 0.01#
learningRateCritic = 0.01#
gamma = 0.95 #
tau=0.01 #
bufferSize = 1e6#
minibatchSize = 1024#

def main():
    debug = 1
    if debug:
        numWolves = 3
        numSheeps = 1
        numBlocks = 2
        hasWalls= 2

        dt=0.02
        maxTimeStep = 75
        sheepSpeedMultiplier = 1.0
        individualRewardWolf = 0

        mujocoVisualize=False
        saveAllmodels = True
        costActionRatio = 0.0

    else:
        print(sys.argv)
        condition = json.loads(sys.argv[1])
        numWolves = int(condition['numWolves'])
        numSheeps = int(condition['numSheeps'])
        numBlocks = int(condition['numBlocks'])
        hasWalls= 0.0 #float(condition['hasWalls'])

        dt = 0.02 #float(condition['dt'])
        maxTimeStep = int(condition['maxTimeStep'])
        sheepSpeedMultiplier = float(condition['sheepSpeedMultiplier'])
        individualRewardWolf = float(condition['individualRewardWolf'])
        costActionRatio = float(condition['costActionRatio'])
        saveAllmodels = 0
        mujocoVisualize=False

    print("maddpg: {} wolves, {} sheep, {} blocks, {} episodes with {} steps each eps, sheepSpeed: {}x, wolfIndividualReward: {}, save all models: {}".format(numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, individualRewardWolf, str(saveAllmodels)))

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
    punishForOutOfBound = PunishForOutOfBound() #lambda state :0
    rewardSheep = RewardSheep(wolvesID, sheepsID, entitiesSizeList, getPosFromAgentState, isCollision, punishForOutOfBound, collisionReward)
    rewardWolf = RewardWolf(wolvesID, sheepsID, entitiesSizeList, isCollision, collisionReward, individualRewardWolf)
    reshapeAction = ReshapeAction()
    getActionCost = GetActionCost(costActionRatio, reshapeAction, individualCost=True)
    getWolvesAction = lambda action: [action[wolfID] for wolfID in wolvesID]
    rewardWolfWithActionCost = lambda state, action, nextState: np.array(rewardWolf(state, action, nextState)) - np.array(getActionCost(getWolvesAction(action)))

    rewardFunc = lambda state, action, nextState: \
        list(rewardWolfWithActionCost(state, action, nextState)) + list(rewardSheep(state, action, nextState))

#------------ mujocoEnv ------------------------

    # physicsDynamicsPath=os.path.join(dirName,'..','..','environment','mujocoEnv','dt={}'.format(dt),
    #                                  'hasWalls={}_numBlocks={}_numSheeps={}_numWolves={}.xml'.format(hasWalls,numBlocks,numSheeps,numWolves))
    physicsDynamicsPath=os.path.join(dirName,'..','..','environment','mujocoEnv','newMujoco',
                                     'hasWalls={}_numBlocks={}_numSheeps={}_numWolves={}.xml'.format(hasWalls,numBlocks,numSheeps,numWolves))

    with open(physicsDynamicsPath) as f:
        xml_string = f.read()
    envXmlDict = xmltodict.parse(xml_string.strip())
    envXml=xmltodict.unparse(envXmlDict)
    physicsModel = mujoco.load_model_from_xml(envXml)
    physicsSimulation = mujoco.MjSim(physicsModel)

    qPosInit = [0, 0]*numAgents
    qVelInit = [0, 0]*numAgents

    qVelInitNoise = 0
    qPosInitNoise = 0.8
    getBlockRandomPos = lambda: np.random.uniform(-0.7, +0.7, 2)

    # qVelInitNoise = 0*hasWalls
    # qPosInitNoise = 0.8*hasWalls
    # getBlockRandomPos = lambda: np.random.uniform(-0.7*hasWalls, +0.7*hasWalls, 2)
    getBlockSpeed = lambda: np.zeros(2)

    numQPos = len(physicsSimulation.data.qpos)
    numQVel = len(physicsSimulation.data.qvel)

    sampleAgentsQPos=lambda: np.asarray(qPosInit)+np.random.uniform(low=-qPosInitNoise, high=qPosInitNoise, size=numQPos)
    sampleAgentsQVel=lambda: np.asarray(qVelInit) + np.random.uniform(low=-qVelInitNoise, high=qVelInitNoise, size=numQVel)

    minDistance=0.2+2*blockSize#>2*wolfSize+2*blockSize
    isOverlap=IsOverlap(minDistance)
    sampleBlockState=SampleBlockState(numBlocks,getBlockRandomPos,getBlockSpeed,isOverlap)

    reset= ResetUniformWithoutXPos(physicsSimulation,  numAgents, numBlocks,sampleAgentsQPos, sampleAgentsQVel,sampleBlockState)

    transitTimePerStep=0.1
    numSimulationFrames = int(transitTimePerStep/dt)

    isTerminal = lambda state: [False]* numAgents
    transit = TransitionFunction(physicsSimulation,numAgents , numSimulationFrames,mujocoVisualize,isTerminal,reshapeAction)

    observeOneAgent = lambda agentID: Observe(agentID, wolvesID, sheepsID, blocksID, getPosFromAgentState,getVelFromAgentState)
    observe = lambda state: [observeOneAgent(agentID)(state) for agentID in range(numAgents)]
    initObsForParams = observe(reset())
    obsShape = [initObsForParams[obsID].shape[0] for obsID in range(len(initObsForParams))]

    worldDim = 2
    actionDim = worldDim * 2 + 1

    layerWidth = [128, 128]

#------------ models ------------------------

    buildMADDPGModels = BuildMADDPGModels(actionDim, numAgents, obsShape)
    modelsList = [buildMADDPGModels(layerWidth, agentID) for agentID in range(numAgents)]

    trainCriticBySASR = TrainCriticBySASR(actByPolicyTargetNoisyForNextState, learningRateCritic, gamma)
    trainCritic = TrainCritic(trainCriticBySASR)
    trainActorFromSA = TrainActorFromSA(learningRateActor)
    trainActor = TrainActor(trainActorFromSA)

    paramUpdateInterval = 1 #
    updateParameters = UpdateParameters(paramUpdateInterval, tau)
    sampleBatchFromMemory = SampleFromMemory(minibatchSize)

    learnInterval = 100
    learningStartBufferSize = minibatchSize * maxTimeStep
    startLearn = StartLearn(learningStartBufferSize, learnInterval)

    trainMADDPGModels = TrainMADDPGModelsWithBuffer(updateParameters, trainActor, trainCritic, sampleBatchFromMemory, startLearn, modelsList)

    actOneStepOneModel = ActOneStep(actByPolicyTrainNoisy)
    actOneStep = lambda allAgentsStates, runTime: [actOneStepOneModel(model, allAgentsStates) for model in modelsList]

    sampleOneStep = SampleOneStep(transit, rewardFunc)
    runTimeStep = RunTimeStep(actOneStep, sampleOneStep, trainMADDPGModels, observe = observe)

    runEpisode = RunEpisode(reset, runTimeStep, maxTimeStep, isTerminal)

    getAgentModel = lambda agentId: lambda: trainMADDPGModels.getTrainedModels()[agentId]
    getModelList = [getAgentModel(i) for i in range(numAgents)]
    modelSaveRate = 5000
    fileName = "maddpg{}wall{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}WolfActCost{}individ{}_agent".format(
        hasWalls, numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, costActionRatio, individualRewardWolf)

    modelPath = os.path.join(dataMainFolder, fileName)
    saveModels = [SaveModel(modelSaveRate, saveVariables, getTrainedModel, modelPath+ str(i), saveAllmodels) for i, getTrainedModel in enumerate(getModelList)]

    maddpg = RunAlgorithm(runEpisode, maxEpisode, saveModels, numAgents)
    replayBuffer = getBuffer(bufferSize)
    meanRewardList = maddpg(replayBuffer)


if __name__ == '__main__':
    main()


