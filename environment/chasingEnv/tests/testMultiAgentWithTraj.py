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
        self.wolvesID = [0, 1]
        self.sheepsID = [2]
        self.blocksID = [3]
        self.isCollision = IsCollision(getPosFromAgentState)
        self.entitiesSizeList = [wolfSize, wolfSize, sheepSize, blockSize]
        self.numEntities = len(self.entitiesSizeList)
        self.entityMaxSpeedList = [wolfMaxSpeed, wolfMaxSpeed, sheepMaxSpeed, blockMaxSpeed]

        self.entitiesMovableList = [True, True, True, False]
        self.massList = [1.0] * self.numEntities

        self.collisionReward = len(self.wolvesID) * 10 # only for this test
        self.collisionPunishment = self.collisionReward
        self.rewardWolf = RewardWolf(self.wolvesID, self.sheepsID, self.entitiesSizeList, self.isCollision, self.collisionReward, individual = 0)
        self.punishForOutOfBound = PunishForOutOfBound()
        self.rewardSheep = RewardSheep(self.wolvesID, self.sheepsID, self.entitiesSizeList, getPosFromAgentState, self.isCollision, self.punishForOutOfBound,
                                       self.collisionPunishment)

        self.rewardFunc = lambda state, action, nextState: \
            list(self.rewardWolf(state, action, nextState)) + list(self.rewardSheep(state, action, nextState))


        self.numTotalAgents = len(self.sheepsID) + len(self.wolvesID)
        self.numBlocks = len(self.blocksID)
        self.reset = ResetMultiAgentChasing(self.numTotalAgents, self.numBlocks)

        observeOneAgent = lambda agentID: Observe(agentID, self.wolvesID, self.sheepsID, self.blocksID, getPosFromAgentState, getVelFromAgentState)
        self.observe = lambda state: [observeOneAgent(agentID)(state) for agentID in range(self.numTotalAgents)]

        trajectoryPath = os.path.join(dirName, 'trajectoryFull.pickle')
        self.traj = loadFromPickle(trajectoryPath)

        observationPath = os.path.join(dirName, 'obs.pickle')
        self.obs = loadFromPickle(observationPath) # 25* 3* 12


    def testObserve(self):
        for timeStep in range(len(self.traj)):
            trueObs = self.obs[timeStep]
            state = self.traj[timeStep][0]
            observation = self.observe(state)
            self.assertTrue([tuple(trueVal) == tuple(calcVal) for trueVal, calcVal in zip(trueObs, observation)])

    def testWolfReward(self):
        for timeStep in range(len(self.traj)):
            state = self.traj[timeStep][0]
            action = self.traj[timeStep][1]
            trueReward = self.traj[timeStep][2]
            nextState = self.traj[timeStep][3]

            trueWolfReward = np.array(trueReward)[self.wolvesID]
            wolfReward = self.rewardWolf(state, action, nextState)

            self.assertEqual(tuple(trueWolfReward), tuple(wolfReward))

    def testSheepReward(self):
        for timeStep in range(len(self.traj)):
            state = self.traj[timeStep][0]
            action = self.traj[timeStep][1]
            trueReward = self.traj[timeStep][2]
            nextState = self.traj[timeStep][3]

            trueSheepReward = np.array(trueReward)[self.sheepsID]

            sheepReward = self.rewardSheep(state, action, nextState)
            self.assertEqual(tuple(trueSheepReward), tuple(sheepReward))

    def testReward(self):
        for timeStep in range(len(self.traj)):
            state = self.traj[timeStep][0]
            action = self.traj[timeStep][1]
            trueReward = self.traj[timeStep][2]
            nextState = self.traj[timeStep][3]
            reward = self.rewardFunc(state, action, nextState)
            self.assertEqual(tuple(trueReward), tuple(reward))


## remember to reshape the actions, action in the trajectory is of dim 3* 5
## trajectory contain action of the block?

    def testTransition(self):
        for timeStep in range(len(self.traj)):
            reshapeAction = ReshapeAction()
            getCollisionForce = GetCollisionForce()
            applyActionForce = ApplyActionForce(self.wolvesID, self.sheepsID, self.entitiesMovableList, actionDim=2)
            applyEnvironForce = ApplyEnvironForce(self.numEntities, self.entitiesMovableList, self.entitiesSizeList,
                                                  getCollisionForce, getPosFromAgentState)
            integrateState = IntegrateState(self.numEntities, self.entitiesMovableList, self.massList,
                                                 self.entityMaxSpeedList, getVelFromAgentState, getPosFromAgentState)
            transit = TransitMultiAgentChasing(self.numEntities, reshapeAction, applyActionForce, applyEnvironForce, integrateState)

            state = self.traj[timeStep][0]
            action = self.traj[timeStep][1]
            action.append(np.zeros(action[0].shape)) # original traj action does not contain block action
            trueNextState = self.traj[timeStep][3]
            nextState = transit(state, action)

            rounding = 7
            trueNextState = np.round(trueNextState, rounding)
            nextState = np.round(nextState, rounding)

            self.assertTrue([tuple(trueS) == tuple(nextS) for trueS, nextS in zip(trueNextState, nextState)])

@ddt
class TestMultiAgentEnvWithLongTrajShared(unittest.TestCase):
    def setUp(self):
        numWolves = 3
        numSheeps = 1
        numBlocks = 2

        self.numAgents = numWolves + numSheeps
        self.numEntities = self.numAgents + numBlocks
        self.wolvesID = list(range(numWolves))
        self.sheepsID = list(range(numWolves, self.numAgents))
        self.blocksID = list(range(self.numAgents, self.numEntities))

        self.entitiesSizeList = [wolfSize] * numWolves + [sheepSize] * numSheeps + [blockSize] * numBlocks
        self.entityMaxSpeedList = [wolfMaxSpeed] * numWolves + [sheepMaxSpeed] * numSheeps + [blockMaxSpeed] * numBlocks
        self.entitiesMovableList = [True] * self.numAgents + [False] * numBlocks
        self.massList = [1.0] * self.numEntities

        self.isCollision = IsCollision(getPosFromAgentState)
        self.collisionReward = 30
        self.collisionPunishment = self.collisionReward
        self.rewardWolf = RewardWolf(self.wolvesID, self.sheepsID, self.entitiesSizeList, self.isCollision, self.collisionReward, individual = 0)
        self.punishForOutOfBound = PunishForOutOfBound()
        self.rewardSheep = RewardSheep(self.wolvesID, self.sheepsID, self.entitiesSizeList, getPosFromAgentState, self.isCollision, self.punishForOutOfBound,
                                       self.collisionPunishment)

        self.rewardFunc = lambda state, action, nextState: \
            list(self.rewardWolf(state, action, nextState)) + list(self.rewardSheep(state, action, nextState))

        self.numTotalAgents = len(self.sheepsID) + len(self.wolvesID)
        self.numBlocks = len(self.blocksID)
        self.reset = ResetMultiAgentChasing(self.numTotalAgents, self.numBlocks)

        observeOneAgent = lambda agentID: Observe(agentID, self.wolvesID, self.sheepsID, self.blocksID, getPosFromAgentState, getVelFromAgentState)
        self.observe = lambda state: [observeOneAgent(agentID)(state) for agentID in range(self.numTotalAgents)]

        trajectoryPath = os.path.join(dirName, 'maddpg3wolves1sheep2blocks60000episodes75stepSheepSpeed1.0WolfActCost0.0shared_mixTraj')
        self.trajList = loadFromPickle(trajectoryPath)


    def testWolfReward(self):
        for traj in self.trajList:
            for timeStep in range(len(traj)):
                state = traj[timeStep][0]
                action = traj[timeStep][1]
                trueReward = traj[timeStep][2]
                nextState = traj[timeStep][3]

                trueWolfReward = np.array(trueReward)[self.wolvesID]
                wolfReward = self.rewardWolf(state, action, nextState)

                self.assertEqual(tuple(trueWolfReward), tuple(wolfReward))

    def testSheepReward(self):
        for traj in self.trajList:

            for timeStep in range(len(traj)):
                state = traj[timeStep][0]
                action = traj[timeStep][1]
                trueReward = traj[timeStep][2]
                nextState = traj[timeStep][3]

                trueSheepReward = np.array(trueReward)[self.sheepsID]

                sheepReward = self.rewardSheep(state, action, nextState)
                self.assertEqual(tuple(trueSheepReward), tuple(sheepReward))

    def testReward(self):
        for traj in self.trajList:
            for timeStep in range(len(traj)):
                state = traj[timeStep][0]
                action = traj[timeStep][1]
                trueReward = traj[timeStep][2]
                nextState = traj[timeStep][3]
                reward = self.rewardFunc(state, action, nextState)
                self.assertEqual(tuple(trueReward), tuple(reward))


## remember to reshape the actions, action in the trajectory is of dim 3* 5
## trajectory contain action of the block?

    def testTransition(self):
        for traj in self.trajList:
            for timeStep in range(len(traj)):
                reshapeAction = ReshapeAction()
                getCollisionForce = GetCollisionForce()
                applyActionForce = ApplyActionForce(self.wolvesID, self.sheepsID, self.entitiesMovableList, actionDim=2)
                applyEnvironForce = ApplyEnvironForce(self.numEntities, self.entitiesMovableList, self.entitiesSizeList,
                                                      getCollisionForce, getPosFromAgentState)
                integrateState = IntegrateState(self.numEntities, self.entitiesMovableList, self.massList,
                                                     self.entityMaxSpeedList, getVelFromAgentState, getPosFromAgentState)
                transit = TransitMultiAgentChasing(self.numEntities, reshapeAction, applyActionForce, applyEnvironForce, integrateState)

                state = traj[timeStep][0]
                action = traj[timeStep][1]
                action.append(np.zeros(action[0].shape)) # original traj action does not contain block action
                trueNextState = traj[timeStep][3]
                nextState = transit(state, action)

                rounding = 7
                trueNextState = np.round(trueNextState, rounding)
                nextState = np.round(nextState, rounding)

                self.assertTrue([tuple(trueS) == tuple(nextS) for trueS, nextS in zip(trueNextState, nextState)])

@ddt
class TestMultiAgentEnvWithLongTrajIndivid(unittest.TestCase):
    def setUp(self):
        numWolves = 3
        numSheeps = 1
        numBlocks = 2

        self.numAgents = numWolves + numSheeps
        self.numEntities = self.numAgents + numBlocks
        self.wolvesID = list(range(numWolves))
        self.sheepsID = list(range(numWolves, self.numAgents))
        self.blocksID = list(range(self.numAgents, self.numEntities))

        self.entitiesSizeList = [wolfSize] * numWolves + [sheepSize] * numSheeps + [blockSize] * numBlocks
        self.entityMaxSpeedList = [wolfMaxSpeed] * numWolves + [sheepMaxSpeed] * numSheeps + [blockMaxSpeed] * numBlocks
        self.entitiesMovableList = [True] * self.numAgents + [False] * numBlocks
        self.massList = [1.0] * self.numEntities

        self.isCollision = IsCollision(getPosFromAgentState)
        self.collisionReward = 30
        self.collisionPunishment = self.collisionReward
        self.rewardWolf = RewardWolf(self.wolvesID, self.sheepsID, self.entitiesSizeList, self.isCollision, self.collisionReward, individual = 1)
        self.punishForOutOfBound = PunishForOutOfBound()
        self.rewardSheep = RewardSheep(self.wolvesID, self.sheepsID, self.entitiesSizeList, getPosFromAgentState, self.isCollision, self.punishForOutOfBound,
                                       self.collisionPunishment)

        self.rewardFunc = lambda state, action, nextState: \
            list(self.rewardWolf(state, action, nextState)) + list(self.rewardSheep(state, action, nextState))

        self.numTotalAgents = len(self.sheepsID) + len(self.wolvesID)
        self.numBlocks = len(self.blocksID)
        self.reset = ResetMultiAgentChasing(self.numTotalAgents, self.numBlocks)

        observeOneAgent = lambda agentID: Observe(agentID, self.wolvesID, self.sheepsID, self.blocksID, getPosFromAgentState, getVelFromAgentState)
        self.observe = lambda state: [observeOneAgent(agentID)(state) for agentID in range(self.numTotalAgents)]

        trajectoryPath = os.path.join(dirName, 'maddpg3wolves1sheep2blocks60000episodes75stepSheepSpeed1.0WolfActCost0.0individ_mixTraj')
        self.trajList = loadFromPickle(trajectoryPath)


    def testWolfReward(self):
        for traj in self.trajList:
            for timeStep in range(len(traj)):
                state = traj[timeStep][0]
                action = traj[timeStep][1]
                trueReward = traj[timeStep][2]
                nextState = traj[timeStep][3]

                trueWolfReward = np.array(trueReward)[self.wolvesID]
                wolfReward = self.rewardWolf(state, action, nextState)

                self.assertEqual(tuple(trueWolfReward), tuple(wolfReward))

    def testSheepReward(self):
        for traj in self.trajList:

            for timeStep in range(len(traj)):
                state = traj[timeStep][0]
                action = traj[timeStep][1]
                trueReward = traj[timeStep][2]
                nextState = traj[timeStep][3]

                trueSheepReward = np.array(trueReward)[self.sheepsID]

                sheepReward = self.rewardSheep(state, action, nextState)
                self.assertEqual(tuple(trueSheepReward), tuple(sheepReward))

    def testReward(self):
        for traj in self.trajList:
            for timeStep in range(len(traj)):
                state = traj[timeStep][0]
                action = traj[timeStep][1]
                trueReward = traj[timeStep][2]
                nextState = traj[timeStep][3]
                reward = self.rewardFunc(state, action, nextState)
                self.assertEqual(tuple(trueReward), tuple(reward))


## remember to reshape the actions, action in the trajectory is of dim 3* 5
## trajectory contain action of the block?

    def testTransition(self):
        for traj in self.trajList:
            for timeStep in range(len(traj)):
                reshapeAction = ReshapeAction()
                getCollisionForce = GetCollisionForce()
                applyActionForce = ApplyActionForce(self.wolvesID, self.sheepsID, self.entitiesMovableList, actionDim=2)
                applyEnvironForce = ApplyEnvironForce(self.numEntities, self.entitiesMovableList, self.entitiesSizeList,
                                                      getCollisionForce, getPosFromAgentState)
                integrateState = IntegrateState(self.numEntities, self.entitiesMovableList, self.massList,
                                                     self.entityMaxSpeedList, getVelFromAgentState, getPosFromAgentState)
                transit = TransitMultiAgentChasing(self.numEntities, reshapeAction, applyActionForce, applyEnvironForce, integrateState)

                state = traj[timeStep][0]
                action = traj[timeStep][1]
                action.append(np.zeros(action[0].shape)) # original traj action does not contain block action
                trueNextState = traj[timeStep][3]
                nextState = transit(state, action)

                rounding = 7
                trueNextState = np.round(trueNextState, rounding)
                nextState = np.round(nextState, rounding)

                self.assertTrue([tuple(trueS) == tuple(nextS) for trueS, nextS in zip(trueNextState, nextState)])


if __name__ == '__main__':
    unittest.main()
