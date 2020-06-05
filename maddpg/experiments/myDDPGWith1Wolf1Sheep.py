import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)


import matplotlib.pyplot as plt
import gym
from collections import deque

from src.ddpg_maddpg import *
from RLframework.RLrun import resetTargetParamToTrainParam, UpdateParameters, SampleOneStep, SampleFromMemory,\
    LearnFromBuffer, RunMultiAgentTimeStep, RunEpisode, RunAlgorithm
from src.policy import ActDDPGOneStep
from functionTools.loadSaveModel import GetSavePath, saveVariables, saveToPickle
from environment.gymEnv.multiAgentEnv import *
from functionTools.trajectory import SampleTrajectory
from visualize.visualizeMultiAgent import *

import maddpg.maddpgAlgor.common.tf_util as U
from maddpg.maddpgAlgor.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers
from gym import spaces
import pickle
import argparse

maxEpisode = 10000
maxTimeStep = 25

learningRateActor = 0.01#
learningRateCritic = 0.01#
gamma = 0.95 #
tau=0.01 #
bufferSize = 1e6#
minibatchSize = 64#
learningStartBufferSize = minibatchSize * maxTimeStep#
########


wolfSize = 0.075
sheepSize = 0.05
blockSize = 0.2

sheepMaxSpeed = 1.3
wolfMaxSpeed = 1.0
blockMaxSpeed = None

wolfColor = np.array([0.85, 0.35, 0.35])
sheepColor = np.array([0.35, 0.85, 0.35])
blockColor = np.array([0.25, 0.25, 0.25])

policyPath = os.path.join(dirName, '..', 'myDDPGpolicy1WolfDDPG1SheepDDPG')


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")  # 60000
    parser.add_argument("--num-adversaries", type=int, default=3, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpgAlgor", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpgAlgor", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default='exp', help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default= policyPath,
                        help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="",
                        help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=True)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default=os.path.join(dirName, '..', 'benchmark_files'),
                        help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default=os.path.join(dirName, '..', 'learning_curves'),
                        help="directory where plot data is saved")
    return parser.parse_args()


def main():
    wolvesID = [0]
    sheepsID = [1]
    blocksID = []

    numWolves = len(wolvesID)
    numSheeps = len(sheepsID)
    numBlocks = len(blocksID)

    numAgents = numWolves + numSheeps
    numEntities = numAgents + numBlocks

    entitiesSizeList = [wolfSize] * numWolves + [sheepSize] * numSheeps + [blockSize] * numBlocks
    entityMaxSpeedList = [wolfMaxSpeed] * numWolves + [sheepMaxSpeed] * numSheeps + [blockMaxSpeed] * numBlocks
    entitiesMovableList = [True] * numAgents + [False] * numBlocks
    massList = [1.0] * numEntities

    isCollision = IsCollision(getPosFromAgentState)
    rewardWolf = RewardWolf(wolvesID, sheepsID, entitiesSizeList, isCollision)
    punishForOutOfBound = PunishForOutOfBound()
    rewardSheep = RewardSheep(wolvesID, sheepsID, entitiesSizeList, getPosFromAgentState, isCollision,
                              punishForOutOfBound)

    rewardFunc = lambda state, action, nextState: \
        list(rewardWolf(state, action, nextState)) + list(rewardSheep(state, action, nextState))

    reset = ResetMultiAgentChasing(numAgents, numBlocks)
    observeOneAgent = lambda agentID: Observe(agentID, wolvesID, sheepsID, blocksID, getPosFromAgentState,
                                              getVelFromAgentState)
    observe = lambda state: [observeOneAgent(agentID)(state) for agentID in range(numAgents)]

    reshapeAction = ReshapeAction()
    getCollisionForce = GetCollisionForce()
    applyActionForce = ApplyActionForce(wolvesID, sheepsID, entitiesMovableList)
    applyEnvironForce = ApplyEnvironForce(numEntities, entitiesMovableList, entitiesSizeList,
                                          getCollisionForce, getPosFromAgentState)
    integrateState = IntegrateState(numEntities, entitiesMovableList, massList,
                                    entityMaxSpeedList, getVelFromAgentState, getPosFromAgentState)
    transit = TransitMultiAgentChasing(numEntities, reshapeAction, applyActionForce, applyEnvironForce, integrateState)

    isTerminal = lambda state: [False]* numAgents
    initObsForParams = observe(reset())
    obsShape = [initObsForParams[obsID].shape for obsID in range(len(initObsForParams))]

    worldDim = 2
    wolfObsDim = np.array(obsShape)[wolvesID[0]][0]
    sheepObsDim = np.array(obsShape)[sheepsID[0]][0]
    actionDim = worldDim * 2 + 1

    layerWidth = [64, 64]


#------------ wolf ------------------------
    buildWolfActorModel = BuildActorModel(wolfObsDim, actionDim)
    actorWriterWolf, actorModelWolf = buildWolfActorModel(layerWidth)

    buildWolfCriticModel = BuildCriticModel(wolfObsDim, actionDim)
    criticWriterWolf, criticModelWolf = buildWolfCriticModel(layerWidth)

    trainCriticBySASRQWolf = TrainCriticBySASRQ(learningRateCritic, gamma, criticWriterWolf)
    trainCriticWolf = TrainCritic(actByPolicyTarget, evaluateCriticTarget, trainCriticBySASRQWolf)

    trainActorFromGradientsWolf = TrainActorFromGradients(learningRateActor, actorWriterWolf)
    trainActorOneStepWolf = TrainActorOneStep(actByPolicyTrain, trainActorFromGradientsWolf, getActionGradients)
    trainActorWolf = TrainActor(trainActorOneStepWolf)

    paramUpdateInterval = 1 #
    updateParameters = UpdateParameters(paramUpdateInterval, tau)

    modelList = [actorModelWolf, criticModelWolf]
    actorModelWolf, criticModelWolf = resetTargetParamToTrainParam(modelList)
    trainWolfModels = TrainDDPGModels(updateParameters, trainActorWolf, trainCriticWolf, actorModelWolf, criticModelWolf)

#------------ sheep ------------------------
    buildSheepActorModel = BuildActorModel(sheepObsDim, actionDim)
    actorWriterSheep, actorModelSheep = buildSheepActorModel(layerWidth)

    buildSheepCriticModel = BuildCriticModel(sheepObsDim, actionDim)
    criticWriterSheep, criticModelSheep = buildSheepCriticModel(layerWidth)

    trainCriticBySASRQSheep = TrainCriticBySASRQ(learningRateCritic, gamma, criticWriterSheep)
    trainCriticSheep = TrainCritic(actByPolicyTarget, evaluateCriticTarget, trainCriticBySASRQSheep)

    trainActorFromGradientsSheep = TrainActorFromGradients(learningRateActor, actorWriterSheep)
    trainActorOneStepSheep = TrainActorOneStep(actByPolicyTrain, trainActorFromGradientsSheep, getActionGradients)
    trainActorSheep = TrainActor(trainActorOneStepSheep)

    paramUpdateInterval = 100 #
    updateParameters = UpdateParameters(paramUpdateInterval, tau)

    modelList = [actorModelSheep, criticModelSheep]
    actorModelSheep, criticModelSheep = resetTargetParamToTrainParam(modelList)
    trainSheepModels = TrainDDPGModels(updateParameters, trainActorSheep, trainCriticSheep, actorModelSheep, criticModelSheep)

# ------------------------------------
    act1AgentOneStepWithNoiseWolf = ActOneStepMADDPGWithNoise(actorModelWolf, actByPolicyTrain)
    act1AgentOneStepWithNoiseSheep = ActOneStepMADDPGWithNoise(actorModelSheep, actByPolicyTrain)

    sample1AgentFromAgentMemory = SampleFromMemory(minibatchSize)
    learnFromBufferWolf = LearnFromBuffer(learningStartBufferSize, sample1AgentFromAgentMemory, trainWolfModels)
    learnFromBufferSheep = LearnFromBuffer(learningStartBufferSize, sample1AgentFromAgentMemory, trainSheepModels)

    sampleOneStep = SampleOneStep(transit, rewardFunc)

    agentsActOneStep = [act1AgentOneStepWithNoiseWolf, act1AgentOneStepWithNoiseSheep]
    actOneStepWithNoise = lambda observation, runTime: [act1Agent(observation[i][None], runTime) for i, act1Agent in enumerate(agentsActOneStep)] # return all actions
    learnFromBuffer = [learnFromBufferWolf, learnFromBufferSheep]
    runDDPGTimeStep = RunMultiAgentTimeStep(actOneStepWithNoise, sampleOneStep, learnFromBuffer, multiagent= True, observe = observe)

    runEpisode = RunEpisode(reset, runDDPGTimeStep, maxTimeStep, isTerminal)
    ddpg = RunAlgorithm(runEpisode, maxEpisode)
    replayBuffer = deque(maxlen=int(bufferSize))
    meanRewardList, trajectory = ddpg(replayBuffer)

    plotResult = True
    if plotResult:
        plt.plot(list(range(len(meanRewardList))), meanRewardList)
        plt.show()

    trainedActorModelSheep, trainedCriticModelSheep = trainSheepModels.getTrainedModels()
    trainedActorModelWolf, trainedCriticModelWolf = trainWolfModels.getTrainedModels()


    with actorModelWolf.as_default():
        saveVariables(trainedActorModelWolf, os.path.join(dirName, '..', 'myDDPGWolfPolicy'))
    with actorModelSheep.as_default():
        saveVariables(trainedActorModelSheep, os.path.join(dirName, '..', 'myDDPGSheepPolicy'))



if __name__ == '__main__':
    main()

