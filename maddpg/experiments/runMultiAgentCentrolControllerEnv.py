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

from environment.chasingEnv.multiAgentEnv import TransitMultiAgentChasing, ApplyActionForce, ApplyEnvironForce, \
    ResetMultiAgentChasing, ReshapeAction, RewardSheep, RewardWolf, Observe, GetCollisionForce, IntegrateState, \
    IsCollision, PunishForOutOfBound, getPosFromAgentState, getVelFromAgentState
from visualize.visualizeMultiAgent import *
from functionTools.trajectory import SampleTrajectory

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
    debug = 1
    if debug:
        numWolves = 2
        numSheeps = 1
        numBlocks = 0
        saveAllmodels = True
        maxEpisode = 60000
    else:
        print(sys.argv)
        condition = json.loads(sys.argv[1])
        numWolves = int(condition['numWolves'])
        numSheeps = int(condition['numSheeps'])
        numBlocks = int(condition['numBlocks'])
        saveAllmodels = True
        maxEpisode = 60000

    print("ddpgCentralController: {} wolves, {} sheep, {} blocks, {} total episodes, save all models: {}".format(
        numWolves, numSheeps, numBlocks, maxEpisode, str(saveAllmodels)))

    numAgents = numWolves + numSheeps
    numEntities = numAgents + numBlocks
    wolvesID = list(range(numWolves))
    sheepsID = list(range(numWolves, numAgents))
    blocksID = list(range(numAgents, numEntities))

    entitiesSizeList = [wolfSize] * numWolves + [sheepSize] * numSheeps + [blockSize] * numBlocks
    entityMaxSpeedList = [wolfMaxSpeed] * numWolves + [sheepMaxSpeed] * numSheeps + [blockMaxSpeed] * numBlocks
    entitiesMovableList = [True] * numAgents + [False] * numBlocks
    massList = [1.0] * numEntities

    isCollision = IsCollision(getPosFromAgentState)
    rewardWolf = RewardWolf(wolvesID, sheepsID, entitiesSizeList, isCollision)
    punishForOutOfBound = PunishForOutOfBound()
    rewardSheep = RewardSheep(wolvesID, sheepsID, entitiesSizeList, getPosFromAgentState, isCollision, punishForOutOfBound)
    rewardFunc = lambda state, action, nextState: list(rewardWolf(state, action, nextState)) + list(rewardSheep(state, action, nextState))
    reshapeReward = lambda reward: [np.sum([reward[id] for id in wolvesID])]+ [reward[id] for id in sheepsID]
    rewardCentralControl = lambda state, action, nextState: reshapeReward(rewardFunc(state, action, nextState))

    reset = ResetMultiAgentChasing(numAgents, numBlocks)
    observeOneAgent = lambda agentID: Observe(agentID, wolvesID, sheepsID, blocksID, getPosFromAgentState, getVelFromAgentState)
    observe = lambda state: [observeOneAgent(agentID)(state) for agentID in range(numAgents)]
    reshapeObsForCentralControl = lambda observation: [np.concatenate([observation[id] for id in wolvesID])] + [observation[id] for id in sheepsID]
    observeCentralControl = lambda state: reshapeObsForCentralControl(observe(state))

    reshapeAction = ReshapeAction()
    getCollisionForce = GetCollisionForce()
    applyActionForce = ApplyActionForce(wolvesID, sheepsID, entitiesMovableList)
    applyEnvironForce = ApplyEnvironForce(numEntities, entitiesMovableList, entitiesSizeList, getCollisionForce, getPosFromAgentState)
    integrateState = IntegrateState(numEntities, entitiesMovableList, massList, entityMaxSpeedList, getVelFromAgentState, getPosFromAgentState)
    transit = TransitMultiAgentChasing(numEntities, reshapeAction, applyActionForce, applyEnvironForce, integrateState)
    reshapeActionForCentralControl = lambda actionCentral: list(
        actionCentral[0].reshape(numWolves, actionDim)) + [actionCentral[id] for id in range(1, len(actionCentral))]
    transitCentralControl = lambda state, action: transit(state, reshapeActionForCentralControl(action))

    isTerminal = lambda state: False

    worldDim = 2
    actionDim = worldDim * 2 + 1

    # ------- example random policy ----------------
    centralControllerActionDim = numWolves * actionDim
    actOneStepRandomCentral = lambda controllerObs: np.random.uniform(-1, +1, centralControllerActionDim)
    actOneStepRandomSheep = lambda sheepObs: np.random.uniform(-1, +1, actionDim)
    agentsActOneStep = [actOneStepRandomCentral, actOneStepRandomSheep]
    policy = lambda state: [actAgent(obs) for actAgent, obs in zip(agentsActOneStep, observeCentralControl(state))]

    # ------ sample trajectory & visualize --------------
    maxRunningSteps = 25
    sampleTrajectory = SampleTrajectory(maxRunningSteps, transitCentralControl, isTerminal, rewardCentralControl, reset)

    trajList = []
    numTrajToSample = 100
    for i in range(numTrajToSample):
        traj = sampleTrajectory(policy)
        trajList = trajList + list(traj)

    # visualize
    visualize = True
    if visualize:
        entitiesColorList = [wolfColor] * numWolves + [sheepColor] * numSheeps + [blockColor] * numBlocks
        render = Render(entitiesSizeList, entitiesColorList, numAgents, getPosFromAgentState)
        render(trajList)


if __name__ == '__main__':
    main()
