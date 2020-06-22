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

from ddpg.src.newddpg_centralController import *

wolfSize = 0.075
sheepSize = 0.05
blockSize = 0.2

sheepMaxSpeed = 1.3
wolfMaxSpeed = 1.0
blockMaxSpeed = None

wolfColor = np.array([0.85, 0.35, 0.35])
sheepColor = np.array([0.35, 0.85, 0.35])
blockColor = np.array([0.25, 0.25, 0.25])

class ResetWithStaticSheep:
    def __init__(self, numTotalAgents, numBlocks):
        self.positionDimension = 2
        self.numTotalAgents = numTotalAgents
        self.numBlocks = numBlocks

    def __call__(self):
        getAgentRandomPos = lambda: np.random.uniform(-1, +1, self.positionDimension)
        getAgentRandomVel = lambda: np.zeros(self.positionDimension)
        agentsState = [list(getAgentRandomPos()) + list(getAgentRandomVel()) for ID in range(2)] + [[0, 0, 0, 0]]

        getBlockRandomPos = lambda: np.random.uniform(-0.9, +0.9, self.positionDimension)
        getBlockSpeed = lambda: np.zeros(self.positionDimension)

        blocksState = [list(getBlockRandomPos()) + list(getBlockSpeed()) for blockID in range(self.numBlocks)]
        state = np.array(agentsState + blocksState)
        return state

def main():
    numWolves = 2
    numSheeps = 1
    numBlocks = 0

    numAgents = numWolves + numSheeps
    numEntities = numAgents + numBlocks
    wolvesID = list(range(numWolves))
    sheepsID = list(range(numWolves, numAgents))
    blocksID = list(range(numAgents, numEntities))
    
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
    reshapeReward = lambda reward: [np.sum([reward[id] for id in wolvesID])]+ [reward[id] for id in sheepsID]
    rewardCentralControl = lambda state, action, nextState: reshapeReward(rewardFunc(state, action, nextState))

    # reset = ResetMultiAgentChasing(numAgents, numBlocks)
    reset = ResetWithStaticSheep(numAgents, numBlocks)
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
    sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, rewardCentralControl, reset)

    reshapeObsForCentralControl = lambda observation: [np.concatenate([observation[id] for id in wolvesID])] + [observation[id] for id in sheepsID]
    observeCentralControl = lambda state: reshapeObsForCentralControl(observe(state))
    initObsForParams = observeCentralControl(reset())
    obsShapeList = [initObsForParams[obsID].shape[0] for obsID in range(len(initObsForParams))] # in range(numAgents)
    worldDim = 2
    actionDim = worldDim * 2 + 1
    layerWidth = [64, 64]

    actionDimListCentralController = [numWolves* actionDim, actionDim]
    numModels = 1 + numSheeps

    controllerObsDim = obsShapeList[0]
    sheepObsDim = obsShapeList[1]

    #------------ models ------------------------
    buildControllerModel = BuildDDPGModelsForCentralController(controllerObsDim, actionDim, numWolves)
    buildAgentModels = [BuildDDPGModels(sheepObsDim, actionDim) for id in range(numSheeps)]

    buildModels = [buildControllerModel] + buildAgentModels
    allModels = [buildAgentModel(layerWidth, agentID) for agentID, buildAgentModel in enumerate(buildModels)]# 2 models

    dirName = os.path.dirname(__file__)
    modelPaths = [os.path.join(dirName, '..', 'trainedModels', 'ddpgCentralControllerStaticSheep2wolves1sheep0blocks60000eps_agent' + str(i) + '60000eps') for i in range(numModels)]
    [restoreVariables(model, path) for model, path in zip(allModels, modelPaths)]

    # agentsActOneStep = [ActOneStepWithSoftMaxNoise(model, actByPolicyTrainNoisy) for model in allModels]
    agentsActOneStep = [ActOneStepWithSoftMaxNoise(allModels[0], actByPolicyTrainNoisy), lambda obs: [0, 0, 0, 0, 0]]

    policyCentral = lambda state: [actAgent(obs[None]) for actAgent, obs in zip(agentsActOneStep, observeCentralControl(state))]
    reshapeActionForCentralControl = lambda actionCentral: list(actionCentral[0].reshape(numWolves, actionDim)) + [actionCentral[id] for id in range(1, len(actionCentral))]
    policy = lambda state: reshapeActionForCentralControl(policyCentral(state))

    trajList = []
    for i in range(100):
        traj = sampleTrajectory(policy)
        trajList = trajList + list(traj)

    # saveTraj
    saveTraj = False
    if saveTraj:
        trajSavePath = os.path.join(dirName, '..', 'trajectory', 'trajectory_my2Wolf1Sheep3Block_allddpg.pickle')
        saveToPickle(trajList, trajSavePath)

    # visualize
    visualize = True
    if visualize:
        entitiesColorList = [wolfColor] * numWolves + [sheepColor] * numSheeps + [blockColor] * numBlocks
        render = Render(entitiesSizeList, entitiesColorList, numAgents, getPosFromAgentState)
        render(trajList)


if __name__ == '__main__':
    main()
