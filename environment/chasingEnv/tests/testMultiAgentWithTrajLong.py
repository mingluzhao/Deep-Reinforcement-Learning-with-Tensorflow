import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
sys.path.append(os.path.join(dirName, '..', '..', '..'))

import unittest
from ddt import ddt, data, unpack
from environment.chasingEnv.multiAgentEnv import *
from functionTools.loadSaveModel import loadFromPickle

wolfSize = 0.075
sheepSize = 0.05
blockSize = 0.2

sheepMaxSpeed = 1.3
wolfMaxSpeed = 1.0
blockMaxSpeed = None

@ddt
class TestMultiAgentEnv(unittest.TestCase):
    def setUp(self):
        numWolves = 3
        numSheeps = 1
        numBlocks = 2
        saveAllmodels = False
        maxTimeStep = 75
        sheepSpeedMultiplier = 1
        individualRewardWolf = 0
        costActionRatio = 0.0

        numAgents = numWolves + numSheeps
        numEntities = numAgents + numBlocks
        self.wolvesID = list(range(numWolves))
        self.sheepsID = list(range(numWolves, numAgents))
        self.blocksID = list(range(numAgents, numEntities))

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

        collisionReward = 30  # originalPaper = 10*3
        isCollision = IsCollision(getPosFromAgentState)
        punishForOutOfBound = PunishForOutOfBound()
        self.rewardSheep = RewardSheep(self.wolvesID, self.sheepsID, entitiesSizeList, getPosFromAgentState, isCollision,
                                  punishForOutOfBound, collisionPunishment=10)

        self.rewardWolf = RewardWolf(self.wolvesID, self.sheepsID, entitiesSizeList, isCollision, collisionReward,
                                individualRewardWolf)
        reshapeAction = ReshapeAction()
        getActionCost = GetActionCost(costActionRatio, reshapeAction, individualCost=True)
        getWolvesAction = lambda action: [action[wolfID] for wolfID in self.wolvesID]
        rewardWolfWithActionCost = lambda state, action, nextState: np.array(
            self.rewardWolf(state, action, nextState)) - np.array(getActionCost(getWolvesAction(action)))

        self.rewardFunc = lambda state, action, nextState: \
            list(rewardWolfWithActionCost(state, action, nextState)) + list(self.rewardSheep(state, action, nextState))

        reset = ResetMultiAgentChasing(numAgents, numBlocks)
        observeOneAgent = lambda agentID: Observe(agentID, self.wolvesID, self.sheepsID, self.blocksID, getPosFromAgentState, getVelFromAgentState)
        observe = lambda state: [observeOneAgent(agentID)(state) for agentID in range(numAgents)]
        getCollisionForce = GetCollisionForce()
        applyActionForce = ApplyActionForce(self.wolvesID, self.sheepsID, entitiesMovableList)
        applyEnvironForce = ApplyEnvironForce(numEntities, entitiesMovableList, entitiesSizeList,
                                              getCollisionForce, getPosFromAgentState)
        integrateState = IntegrateState(numEntities, entitiesMovableList, massList,
                                        entityMaxSpeedList, getVelFromAgentState, getPosFromAgentState)
        self.transit = TransitMultiAgentChasing(numEntities, reshapeAction, applyActionForce, applyEnvironForce,
                                           integrateState)

        self.isTerminal = lambda state: [False] * numAgents
        trajectoryPath = os.path.join(dirName, '3v1Traj.pkl')
        self.trajList = loadFromPickle(trajectoryPath)[:3]

        observationPath = os.path.join(dirName, 'obs.pickle')
        self.obs = loadFromPickle(observationPath) # 25* 3* 12



    def testWolfReward(self):
        for traj in self.trajList:
            for timeStep in range(len(traj)):
                state = traj[timeStep][0]
                action = traj[timeStep][1]
                trueReward = traj[timeStep][2]
                nextState = traj[timeStep][3]

                trueWolfReward = np.array(trueReward)[self.wolvesID]
                wolfReward = self.rewardWolf(state, action, nextState)

                rounding = 7
                trueWolfReward = np.round(trueWolfReward, rounding)
                wolfReward = np.round(wolfReward, rounding)
                self.assertEqual(tuple(wolfReward), tuple(trueWolfReward))


    def testSheepReward(self):
        for traj in self.trajList:
            for timeStep in range(len(traj)):
                state = traj[timeStep][0]
                action = traj[timeStep][1]
                trueReward = traj[timeStep][2]
                nextState = traj[timeStep][3]

                trueSheepReward = np.array(trueReward)[self.sheepsID]
                sheepReward = self.rewardSheep(state, action, nextState)

                rounding = 7
                trueSheepReward = np.round(trueSheepReward, rounding)
                sheepReward = np.round(sheepReward, rounding)

                self.assertEqual(sheepReward, trueSheepReward)

    def testReward(self):
        for traj in self.trajList:
            for timeStep in range(len(traj)):
                state = traj[timeStep][0]
                action = traj[timeStep][1]
                trueReward = traj[timeStep][2]
                nextState = traj[timeStep][3]
                reward = self.rewardFunc(state, action, nextState)
                rounding = 7
                trueReward = np.round(trueReward, rounding)
                reward = np.round(reward, rounding)

                self.assertEqual(tuple(reward), tuple(trueReward))

    def testTransition(self):
        for traj in self.trajList:
            for timeStep in range(len(traj)):
                state = traj[timeStep][0]
                action = traj[timeStep][1]
                trueNextState = traj[timeStep][3]
                nextState = self.transit(state, action)

                rounding = 5
                trueNextState = np.round(trueNextState, rounding)
                nextState = np.round(nextState, rounding)

                [self.assertEqual(tuple(trueS), tuple(nextS)) for trueS, nextS in zip(trueNextState, nextState)]


if __name__ == '__main__':
    unittest.main()
