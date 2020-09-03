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
from environment.chasingEnv.multiAgentEnvWithIndividReward import RewardWolfIndividual
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
        self.rewardSheep = RewardSheep(self.wolvesID, self.sheepsID, self.entitiesSizeList, getPosFromAgentState, self.isCollision, self.punishForOutOfBound,
                                       collisionPunishment=10)

        self.numTotalAgents = len(self.sheepsID) + len(self.wolvesID)
        self.numBlocks = len(self.blocksID)
        self.reset = ResetMultiAgentChasing(self.numTotalAgents, self.numBlocks)


    @data((np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]), [0, 0, 0, 0], 0.8*(0.5**2)),
          (np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]]), [0, 0, 0, 0], 0.8*(0.5)),
           (np.array([[0,0.2, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]]), [0, 0, 0, 0], 0.8)
           )
    @unpack
    def testHurtProbCalc(self, wolvesState, currentSheepState, trueHurtProb):
        sensitiveZoneRadius = 0.25
        oneWolfSelfHurtProb = 0.8
        getHurtProbOfCatching = GetHurtProbOfCatchingByDeterministicZone(getPosFromAgentState, computeVectorNorm, sensitiveZoneRadius, oneWolfSelfHurtProb)
        hurtProb = getHurtProbOfCatching(wolvesState, currentSheepState)
        self.assertEqual(hurtProb, trueHurtProb)

    @data((np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]), {(20, 20, 20): 0.64, (5, 5, 5): 0.32, (-10, -10, -10): 0.04}), # 2closeby, 2 catch
          (np.array([[0, 0.2, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]]), {(0, 0, 0): 1}), # 1closeby, 0catch
          (np.array([[0, 0.2, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]]), {(10, 10, 10): 0.8, (-5, -5, -5): 0.2}), # 2closeby, 1catch - 0.2
          (np.array([[0, 0.2, 0, 0], [0, 0, 0, 0], [0, 0.2, 0, 0], [0, 0, 0, 0]]), {(10, 10, 10): 0.9, (-5, -5, -5): 0.1}) # 3closeby, 1catch
          )
    @unpack
    def testRewardWithHurtProbWithSharedWolf(self, nextState, trueHurtProb):
        state = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        action = [1, 1, 1, 1]
        sensitiveZoneRadius = 0.25
        oneWolfSelfHurtProb = 0.4
        getHurtProbOfCatching = GetHurtProbOfCatchingByDeterministicZone(getPosFromAgentState, computeVectorNorm, sensitiveZoneRadius, oneWolfSelfHurtProb)
        individualWolf = False
        rewardWolfWithProbOfHurt = RewardWolfWithHurtProb(self.wolvesID, self.sheepsID, self.entitiesSizeList, self.isCollision,
                                                          getHurtProbOfCatching, sampleFromDistribution, individualWolf, hurtReward = -5, collisionReward=10)
        iterationTime = 100000
        trueDict = {rew: trueHurtProb[rew]* iterationTime for rew in trueHurtProb.keys()}

        rewardList = [rewardWolfWithProbOfHurt(state, action, nextState) for _ in range(iterationTime)]
        for reward in trueHurtProb.keys():
            self.assertAlmostEqual(rewardList.count(list(reward)), trueDict[reward] , delta=500)

    @data((np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]), {(10, 0, 10): 0.64, (10, 0, -5): 0.16, (-5, 0, 10): 0.16, (-5, 0, -5): 0.04}), # 2closeby, 2 catch
          (np.array([[0, 0.2, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]]), {(0, 0, 0): 1}), # 1closeby, 0catch
          (np.array([[0, 0.2, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]]), {(0, 10, 0): 0.8, (0, -5, 0): 0.2}), # 2closeby, 1catch - 0.2
          (np.array([[0, 0.2, 0, 0], [0, 0, 0, 0], [0, 0.2, 0, 0], [0, 0, 0, 0]]), {(0, 10, 0): 0.9, (0, -5, 0): 0.1}) # 3closeby, 1catch
          )
    @unpack
    def testRewardWithHurtProbWithIndividWolf(self, nextState, trueHurtProb):
        state = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        action = [1, 1, 1, 1]
        sensitiveZoneRadius = 0.25
        oneWolfSelfHurtProb = 0.4
        getHurtProbOfCatching = GetHurtProbOfCatchingByDeterministicZone(getPosFromAgentState, computeVectorNorm, sensitiveZoneRadius, oneWolfSelfHurtProb)
        individualWolf = True
        rewardWolfWithProbOfHurt = RewardWolfWithHurtProb(self.wolvesID, self.sheepsID, self.entitiesSizeList, self.isCollision,
                                                          getHurtProbOfCatching, sampleFromDistribution, individualWolf, hurtReward = -5, collisionReward=10)
        iterationTime = 100000
        trueDict = {rew: trueHurtProb[rew]* iterationTime for rew in trueHurtProb.keys()}

        rewardList = [rewardWolfWithProbOfHurt(state, action, nextState) for _ in range(iterationTime)]
        for reward in trueHurtProb.keys():
            self.assertAlmostEqual(rewardList.count(list(reward)), trueDict[reward] , delta=500)


    def testRewardWithNoProbOfHuntSharedComparedWithTraj(self):
        sensitiveZoneRadius = 0.25
        oneWolfSelfHurtProb = 0 # not hurt
        getHurtProbOfCatching = GetHurtProbOfCatchingByDeterministicZone(getPosFromAgentState, computeVectorNorm, sensitiveZoneRadius,
                                                                         oneWolfSelfHurtProb)
        individualWolf = False
        rewardWolfWithProbOfHurt = RewardWolfWithHurtProb(self.wolvesID, self.sheepsID, self.entitiesSizeList, self.isCollision,
                                                          getHurtProbOfCatching, sampleFromDistribution, individualWolf, hurtReward = -5, collisionReward=10)

        trajPath =  os.path.join(dirName, '..', '..', '..', 'maddpg', 'trajectories', 'evalOneSheep_2575steps_1to6wolves_11.25speed', 'maddpg3wolves1sheep2blocks60000eps75stepSheepSpeed1.0sharedTraj')
        trajList = loadFromPickle(trajPath)
        for traj in trajList:
            for timeStep in range(len(traj)):
                state = traj[timeStep][0]
                action = traj[timeStep][1]
                trueReward = traj[timeStep][2]
                nextState = traj[timeStep][3]

                trueWolfReward = np.array(trueReward)[self.wolvesID]
                wolfReward = rewardWolfWithProbOfHurt(state, action, nextState)

                self.assertEqual(tuple(trueWolfReward), tuple(wolfReward))


    def testRewardWithNoProbOfHuntIndividComparedWithTraj(self):
        sensitiveZoneRadius = 0.25
        oneWolfSelfHurtProb = 0  # not hurt
        getHurtProbOfCatching = GetHurtProbOfCatchingByDeterministicZone(getPosFromAgentState, computeVectorNorm, sensitiveZoneRadius,
                                                                         oneWolfSelfHurtProb)
        individualWolf = True
        rewardWolfWithProbOfHurt = RewardWolfWithHurtProb(self.wolvesID, self.sheepsID, self.entitiesSizeList,
                                                          self.isCollision,
                                                          getHurtProbOfCatching, sampleFromDistribution,
                                                          individualWolf, hurtReward=-5, collisionReward=10)

        trajPath = os.path.join(dirName, '..', '..', '..', 'maddpg', 'trajectories',
                                '3wolvesMaddpgWithActionCost_sharedWolvesHasIndividCost',
                                'maddpg3wolves1sheep2blocks60000episodes75stepSheepSpeed1.0WolfActCost0.0individ')

        trajList = loadFromPickle(trajPath)
        for traj in trajList:
            for timeStep in range(len(traj)):
                state = traj[timeStep][0]
                action = traj[timeStep][1]
                trueReward = traj[timeStep][2]
                nextState = traj[timeStep][3]

                trueWolfReward = np.array(trueReward)[self.wolvesID]
                wolfReward = rewardWolfWithProbOfHurt(state, action, nextState)

                self.assertEqual(tuple(trueWolfReward), tuple(wolfReward))


if __name__ == '__main__':
    unittest.main()
