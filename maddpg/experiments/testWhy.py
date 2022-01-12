import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from environment.chasingEnv.multiAgentEnv import *
from functionTools.loadSaveModel import saveToPickle, restoreVariables, loadFromPickle
from functionTools.trajectory import SampleTrajectoryResetAtTerminal
from environment.chasingEnv.rewardWithKillProbSensitiveToDist import *

from maddpg.maddpgAlgor.trainer.myMADDPG import *
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
import random

wolfColor = np.array([0.85, 0.35, 0.35])
sheepColor = np.array([0.35, 0.85, 0.35])
blockColor = np.array([0.25, 0.25, 0.25])


def calcWolfTrajKillAmount(traj, wolvesID, singleReward = 10):
    rewardIDinTraj = 2
    getWolfReward = lambda allAgentsReward: np.sum([allAgentsReward[wolfID] for wolfID in wolvesID])
    rewardList = [getWolfReward(timeStepInfo[rewardIDinTraj]) for timeStepInfo in traj]
    trajReward = np.sum(rewardList)
    killNumber = trajReward/ singleReward

    return killNumber



class Try:
    def __init__(self):
        self.numWolves = 4
        self.recovered = 10000.0
        self.cost = 0.0
        self.fileID = 8.0
        self.epsID = 60000


    def __call__(self):
        numWolves = self.numWolves
        recoveredSelfishIndex = self.recovered
        costActionRatio = self.cost
        fileID = self.fileID
        epsID = self.epsID

        continueSelfishIndex = 10000.0 if fileID < 10 else 0.0

        numSheeps = 1
        numBlocks = 2
        maxTimeStep = 75
        killReward = 10
        killProportion = 0.2
        biteReward = 0.0
        maxEpisode = 60000
        sheepSpeedMultiplier = 0.75

        numAgents = numWolves + numSheeps
        numEntities = numAgents + numBlocks
        wolvesID = list(range(numWolves))
        sheepsID = list(range(numWolves, numAgents))
        blocksID = list(range(numAgents, numEntities))

        wolfSize = 0.075
        sheepSize = 0.05
        blockSize = 0.2
        entitiesSizeList = [wolfSize] * numWolves + [sheepSize] * numSheeps + [blockSize] * numBlocks

        wolfMaxSpeed = 1.0
        blockMaxSpeed = None
        sheepMaxSpeedOriginal = 1.3
        sheepMaxSpeed = sheepMaxSpeedOriginal * sheepSpeedMultiplier
        entityMaxSpeedList = [wolfMaxSpeed] * numWolves + [sheepMaxSpeed] * numSheeps + [blockMaxSpeed] * numBlocks

        entitiesMovableList = [True] * numAgents + [False] * numBlocks
        massList = [1.0] * numEntities

        collisionReward = 10
        isCollision = IsCollision(getPosFromAgentState)
        punishForOutOfBound = PunishForOutOfBound()
        rewardSheep = RewardSheep(wolvesID, sheepsID, entitiesSizeList, getPosFromAgentState, isCollision,
                                  punishForOutOfBound, collisionPunishment=collisionReward)

        collisionDist = wolfSize + sheepSize
        getAgentsPercentageOfRewards = GetAgentsPercentageOfRewards(sensitivity = 10000, collisionDist = collisionDist)
        terminalCheck = TerminalCheck()
        getCollisionWolfReward = GetCollisionWolfReward(biteReward, killReward, killProportion, sampleFromDistribution, terminalCheck)
        getWolfSheepDistance = GetWolfSheepDistance(computeVectorNorm, getPosFromAgentState)
        rewardWolf = RewardWolvesWithKillProb(wolvesID, sheepsID, entitiesSizeList, isCollision, terminalCheck,
                                              getWolfSheepDistance, getAgentsPercentageOfRewards, getCollisionWolfReward)

        rewardFunc = lambda state, action, nextState: \
            list(rewardWolf(state, action, nextState)) + list(rewardSheep(state, action, nextState))

        reset = ResetMultiAgentChasing(numAgents, numBlocks)
        observeOneAgent = lambda agentID: Observe(agentID, wolvesID, sheepsID, blocksID, getPosFromAgentState, getVelFromAgentState)
        observe = lambda state: [observeOneAgent(agentID)(state) for agentID in range(numAgents)]

        getCollisionForce = GetCollisionForce()
        reshapeAction = ReshapeAction()
        applyActionForce = ApplyActionForce(wolvesID, sheepsID, entitiesMovableList)
        applyEnvironForce = ApplyEnvironForce(numEntities, entitiesMovableList, entitiesSizeList, getCollisionForce, getPosFromAgentState)
        integrateState = IntegrateState(numEntities, entitiesMovableList, massList, entityMaxSpeedList, getVelFromAgentState, getPosFromAgentState)
        transit = TransitMultiAgentChasing(numEntities, reshapeAction, applyActionForce, applyEnvironForce, integrateState)

        isTerminal = lambda state: terminalCheck.terminal
        maxRunningStepsToSample = 75
        sampleTrajectory = SampleTrajectoryResetAtTerminal(maxRunningStepsToSample, transit, isTerminal, rewardFunc, reset)

        initObsForParams = observe(reset())
        obsShape = [initObsForParams[obsID].shape[0] for obsID in range(len(initObsForParams))]
        worldDim = 2
        actionDim = worldDim * 2 + 1
        buildMADDPGModels = BuildMADDPGModels(actionDim, numAgents, obsShape)
        layerWidth = [128, 128]
        modelsList = [buildMADDPGModels(layerWidth, agentID) for agentID in range(numAgents)]

        sheepModel = modelsList[sheepsID[0]]
        wolvesModels = modelsList[:-1]

        dirName = os.path.dirname(__file__)
        if epsID <= 60000:
            rewardSensitivityToDistance = recoveredSelfishIndex
            fileName = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}WolfActCost{}sensitive{}biteReward{}killPercent{}file{}_agent".format(
                numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, costActionRatio,
                rewardSensitivityToDistance, biteReward, killProportion, fileID)
            folderName = 'maddpg_testEqui_new'
            wolfModelPaths = [os.path.join(dirName, '..', 'trainedModels', folderName, fileName + str(i) + str(epsID) + 'eps') for i in wolvesID]
            [restoreVariables(model, path) for model, path in zip(wolvesModels, wolfModelPaths)]

            sheepPath = os.path.join(dirName, '..', 'trainedModels', folderName, fileName + str(numWolves) + str(epsID) + 'eps')
            restoreVariables(sheepModel, sheepPath)

        else:
            epsIDToUse = epsID - 60000
            fileName = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}WolfActCost{}biteReward{}killPercent{}initSensitive{}later{}file{}_agent".format(
                numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, costActionRatio,
                biteReward, killProportion, recoveredSelfishIndex, continueSelfishIndex, fileID)
            folderName = 'maddpg_testEqui_new'
            wolfModelPaths = [os.path.join(dirName, '..', 'trainedModels', folderName, fileName + str(i) + str(epsIDToUse) + 'eps') for i in wolvesID]
            [restoreVariables(model, path) for model, path in zip(wolvesModels, wolfModelPaths)]
            sheepPath = os.path.join(dirName, '..', 'trainedModels', folderName, fileName + str(numWolves) + str(epsID) + 'eps')
            restoreVariables(sheepModel, sheepPath)


        killNumberList = []
        numTrajToSample = 3
        for i in range(numTrajToSample):
            actOneStepOneModel = ActOneStep(actByPolicyTrainNoisy)
            policy = lambda allAgentsStates: [actOneStepOneModel(model, observe(allAgentsStates)) for model in modelsList]

            traj = sampleTrajectory(policy)
            killNumber = calcWolfTrajKillAmount(traj, wolvesID)
            killNumberList.append(killNumber)

        meanTrajKill = np.mean(killNumberList)
        seTrajKill = np.std(killNumberList) / np.sqrt(len(killNumberList) - 1)
        print('meanTrajKill', meanTrajKill, 'seTrajKill ', seTrajKill)


        return pd.Series({'meanKill': meanTrajKill, 'seKill': seTrajKill})



def main():
    tryWhy = Try()
    tryWhy()



if __name__ == '__main__':
    main()
