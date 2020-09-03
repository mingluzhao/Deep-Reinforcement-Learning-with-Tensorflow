import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..', '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
sys.path.append(os.path.join(dirName, '..'))

import unittest
from ddt import ddt, data, unpack
from environment.chasingEnv.multiAgentEnv import *
from environment.chasingEnv.rewardWithFightBackProb import *
import random
from functionTools.loadSaveModel import loadFromPickle

@ddt
class TestMultiAgentEnv(unittest.TestCase):
    def setUp(self):
        numWolves = 3
        numSheeps = 1
        numBlocks = 2

        numAgents = numWolves + numSheeps
        numEntities = numAgents + numBlocks
        self.wolvesID = list(range(numWolves))
        self.sheepsID = list(range(numWolves, numAgents))
        self.blocksID = list(range(numAgents, numEntities))
        self.isCollision = IsCollision(getPosFromAgentState)

        wolfSize = 0.075
        sheepSize = 0.05
        blockSize = 0.2
        self.entitiesSizeList = [wolfSize] * numWolves + [sheepSize] * numSheeps + [blockSize] * numBlocks

        self.punishForOutOfBound = PunishForOutOfBound()
        self.rewardSheep = RewardSheep(self.wolvesID, self.sheepsID, self.entitiesSizeList, getPosFromAgentState, self.isCollision, self.punishForOutOfBound, collisionPunishment=10)

        self.numTotalAgents = len(self.sheepsID) + len(self.wolvesID)
        self.numBlocks = len(self.blocksID)
        self.reset = ResetMultiAgentChasing(self.numTotalAgents, self.numBlocks)

        a = 10
        b = 5
        self.getAgentEffectiveIndex = GetAgentEffectiveIndex(a, b)

        initHurtProb = 0.8
        self.getFightBackFromEffIndex = GetFightBackFromEffIndex(initHurtProb)
        self.getFightBackProb = GetFightBackProb(self.getAgentEffectiveIndex, getPosFromAgentState, self.getFightBackFromEffIndex, computeVectorNorm)

        killProportion = 0.2
        fightedBackReward = -20
        biteReward = 20
        killReward = 100

        self.getRewardFromFightBackProb = GetRewardFromFightBackProb(biteReward, killReward, fightedBackReward, killProportion, sampleFromDistribution)
        self.reward = RewardWithHurtProb(self.wolvesID, self.sheepsID, self.entitiesSizeList, self.isCollision, self.punishForOutOfBound, getPosFromAgentState, self.getFightBackProb, self.getRewardFromFightBackProb)
        self.reshapeAction = ReshapeAction()

    def testEffectiveIndex(self):
        for i in range(10):
            dist = i+ random.random()
            self.assertTrue(self.getAgentEffectiveIndex(dist) > self.getAgentEffectiveIndex(dist+1))

    @data((0.8, {(-20, 0): 0.8, (20, -20): 0.2*0.8, (100, -100): 0.2* 0.2}),
          (0.4, {(-20, 0): 0.4, (20, -20): 0.6*0.8, (100, -100): 0.6* 0.2})
          )
    @unpack
    def testRewardDist(self, fightBackProb, trueProb):
        iterationTime = 100000
        trueDict = {rew: trueProb[rew] * iterationTime for rew in trueProb.keys()}
        rewardList = [self.getRewardFromFightBackProb(fightBackProb) for _ in range(iterationTime)]
        for reward in trueProb.keys():
            self.assertAlmostEqual(rewardList.count(reward), trueDict[reward], delta=500)


    def testRewardWithNoProbOfHuntIndividComparedWithTraj(self):
        initHurtProb = 0
        getFightBackFromEffIndex = GetFightBackFromEffIndex(initHurtProb)

        getFightBackProb = GetFightBackProb(self.getAgentEffectiveIndex, getPosFromAgentState, getFightBackFromEffIndex, computeVectorNorm)

        killProportion = 0
        fightedBackReward = -20
        biteReward = 30
        killReward = 100

        getRewardFromFightBackProb = GetRewardFromFightBackProb(biteReward, killReward, fightedBackReward, killProportion, sampleFromDistribution)
        reward = RewardWithHurtProb(self.wolvesID, self.sheepsID, self.entitiesSizeList, self.isCollision, self.punishForOutOfBound, getPosFromAgentState, getFightBackProb, getRewardFromFightBackProb)


        trajPath = os.path.join(dirName, 'maddpg3wolves1sheep2blocks60000episodes75stepSheepSpeed1.0WolfActCost0.0individ_mixTraj')

        trajList = loadFromPickle(trajPath)
        for traj in trajList:
            for timeStep in range(len(traj)):
                state = traj[timeStep][0]
                action = traj[timeStep][1]
                trueReward = traj[timeStep][2]
                nextState = traj[timeStep][3]

                trueReward = np.array(trueReward)
                agentsReward = reward(state, action, nextState)
                self.assertEqual(tuple(trueReward), tuple(agentsReward))

    def testCalc(self):
        trajPath = os.path.join(dirName, 'maddpg3wolves1sheep2blocks60000episodes75stepSheepSpeed1.0WolfActCost0.0individ_mixTraj')

        traj = loadFromPickle(trajPath)[0]
        for timeStep in range(len(traj)):
            state = traj[timeStep][0]
            action = traj[timeStep][1]
            trueReward = traj[timeStep][2]
            nextState = traj[timeStep][3]

            # print('true ', trueReward)
            # print(self.reward(state, action, nextState))
            #
    def testWithActCost(self):
        costActionRatio = 0.1
        reshapeAction = ReshapeAction()
        getGroupActionCost = GetActionCost(costActionRatio, reshapeAction, individualCost=True)

        getActionCost = lambda action: list(getGroupActionCost([action[wolfID] for wolfID in self.wolvesID])) + [0] * len(self.sheepsID)
        rewardFunc = lambda state, action, nextState: np.array(self.reward(state, action, nextState)) - np.array(getActionCost(action))

        trajPath = os.path.join(dirName, 'maddpg3wolves1sheep2blocks60000episodes75stepSheepSpeed1.0WolfActCost0.0individ_mixTraj')

        traj = loadFromPickle(trajPath)[0]
        for timeStep in range(len(traj)):
            state = traj[timeStep][0]
            action = traj[timeStep][1]
            trueReward = traj[timeStep][2]
            nextState = traj[timeStep][3]

            print('nocost', self.reward(state, action, nextState))
            print('withcost', rewardFunc(state, action, nextState))

if __name__ == '__main__':
    unittest.main()
