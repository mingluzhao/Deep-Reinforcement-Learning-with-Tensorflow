import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
from gym import spaces

from environment.chasingEnv.multiAgentEnv import *
from functionTools.loadSaveModel import saveToPickle, restoreVariables
from functionTools.trajectory import SampleTrajectory
from visualize.visualizeMultiAgent import *

from ddpg.src.newddpg_followPaper import *

wolfSize = 0.075
sheepSize = 0.05
blockSize = 0.2

sheepMaxSpeed = 1.3
wolfMaxSpeed = 1.0
blockMaxSpeed = None

wolfColor = np.array([0.85, 0.35, 0.35])
sheepColor = np.array([0.35, 0.85, 0.35])
blockColor = np.array([0.25, 0.25, 0.25])


def main():
    wolvesID = [0]
    sheepsID = [1]
    blocksID = []

    numWolves = len(wolvesID)
    numSheeps = len(sheepsID)
    numBlocks = len(blocksID)
    
    numAgents = numWolves + numSheeps
    numEntities = numAgents + numBlocks
    
    entitiesSizeList = [wolfSize]* numWolves + [sheepSize] * numSheeps + [blockSize]* numBlocks
    entityMaxSpeedList = [wolfMaxSpeed]* numWolves + [sheepMaxSpeed] * numSheeps + [blockMaxSpeed]* numBlocks
    entitiesMovableList = [True]* numAgents + [False] * numBlocks
    massList = [1.0] * numEntities
    
    isCollision = IsCollision(getPosFromAgentState)
    rewardWolf = RewardWolf(wolvesID, sheepsID, entitiesSizeList, isCollision)
    punishForOutOfBound = PunishForOutOfBound()
    rewardSheep = RewardSheep(wolvesID, sheepsID, entitiesSizeList, getPosFromAgentState, isCollision, punishForOutOfBound)

    rewardFunc = lambda state, action, nextState: \
        list(rewardWolf(state, action, nextState)) + list(rewardSheep(state, action, nextState))

    reset = ResetMultiAgentChasing(numAgents, numBlocks)
    observeOneAgent = lambda agentID: Observe(agentID, wolvesID, sheepsID, blocksID, getPosFromAgentState, getVelFromAgentState)
    observe = lambda state: [observeOneAgent(agentID)(state) for agentID in range(numAgents)]

    reshapeAction = ReshapeAction()
    getCollisionForce = GetCollisionForce()
    applyActionForce = ApplyActionForce(wolvesID, sheepsID, entitiesMovableList)
    applyEnvironForce = ApplyEnvironForce(numEntities, entitiesMovableList, entitiesSizeList,
                                          getCollisionForce, getPosFromAgentState)
    integrateState = IntegrateState(numEntities, entitiesMovableList, massList,
                                    entityMaxSpeedList, getVelFromAgentState, getPosFromAgentState)
    transit = TransitMultiAgentChasing(numEntities, reshapeAction, applyActionForce, applyEnvironForce, integrateState)

    isTerminal = lambda state: False
    maxRunningSteps = 25
    sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, rewardFunc, reset)

    initObsForParams = observe(reset())
    obsShape = [initObsForParams[obsID].shape for obsID in range(len(initObsForParams))]

#----------policy -----------------------
    worldDim = 2
    wolfObsDim = np.array(obsShape)[wolvesID[0]][0]
    sheepObsDim = np.array(obsShape)[sheepsID[0]][0]
    actionDim = worldDim * 2 + 1
    layerWidth = [64, 64]

    buildWolfModel = BuildDDPGModels(wolfObsDim, actionDim)
    writerWolf, modelWolf = buildWolfModel(layerWidth, wolvesID[0])

    buildSheepModel = BuildDDPGModels(sheepObsDim, actionDim)
    writerSheep, modelSheep = buildSheepModel(layerWidth, sheepsID[0])

    dirName = os.path.dirname(__file__)
    modelPathWolf = os.path.join(dirName, '..', 'myDDPGWolfPolicy')
    modelPathSheep = os.path.join(dirName, '..', 'myDDPGSheepPolicy')

    # dirName = os.path.dirname(__file__)
    # modelPathWolf = os.path.join(dirName, '..', 'newDDPGWolfPolicy12')
    # modelPathSheep = os.path.join(dirName, '..', 'newDDPGSheepPolicy12')

    restoreVariables(modelWolf, modelPathWolf)
    restoreVariables(modelSheep, modelPathSheep)

    policyWolf = ActOneStepMADDPGWithNoise(modelWolf, actByPolicyTrainNoisy)
    policySheep = ActOneStepMADDPGWithNoise(modelSheep, actByPolicyTrainNoisy)
    agentPolicies = [policyWolf, policySheep]

    trajList = []
    for i in range(20):
        with U.single_threaded_session():
            policy = lambda state: [actAgent(obs[None]) for actAgent, obs in zip(agentPolicies, observe(state))]
            traj = sampleTrajectory(policy)
            trajList = trajList + list(traj)

    # saveTraj
        saveTraj = False
        if saveTraj:
            trajSavePath = os.path.join(dirName, '..', 'trajectory', 'trajectory_newddpg.pickle')
            saveToPickle(traj, trajSavePath)

    # visualize
    visualize = True
    if visualize:
        entitiesColorList = [wolfColor] * numWolves + [sheepColor] * numSheeps + [blockColor] * numBlocks
        render = Render(entitiesSizeList, entitiesColorList, numAgents, getPosFromAgentState)
        render(trajList)


if __name__ == '__main__':
    main()
