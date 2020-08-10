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

@ddt
class TestMultiAgentEnv2Wolves(unittest.TestCase):
    def setUp(self):
        self.sheepsID = [0]
        self.wolvesID = [1, 2]
        self.blocksID = [3, 4]
        self.isCollision = IsCollision(getPosFromAgentState)
        self.entitiesSizeList = [0.1] * len(self.sheepsID) + [0.2] * len(self.wolvesID) + [0.5] * len(self.blocksID)

        self.collisionReward = len(self.wolvesID) * 10 # only for this test
        self.collisionPunishment = self.collisionReward
        self.punishForOutOfBound = PunishForOutOfBound()
        self.rewardSheep = RewardSheep(self.wolvesID, self.sheepsID, self.entitiesSizeList, getPosFromAgentState, self.isCollision, self.punishForOutOfBound,
                                       self.collisionPunishment)

        self.numTotalAgents = len(self.sheepsID) + len(self.wolvesID)
        self.numBlocks = len(self.blocksID)
        self.reset = ResetMultiAgentChasing(self.numTotalAgents, self.numBlocks)

    @data((np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
           [1, 1, 1, 1],
           np.array([[0, 1, 1., 0.], [1,  0, 1.,0.], [1,  0, 1.,0.], [1,  0, 1.,0.], [1,  0, 1.,0.]]),
           1,
           [0, 0] # wolfwolf collide but no reward
           ),
          (np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
           [1, 1, 1, 1],
           np.array([[0, 1, 1., 0.], [1, 0, 1., 0.], [1, 0, 1., 0.], [1, 0, 1., 0.], [1, 0, 1., 0.]]),
            .2,
           [0, 0]  # wolfwolf collide but no reward
           ),
          (np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
           [1, 1, 1, 1],
           np.array([[0, 1, 1., 0.], [1, 0, 1., 0.], [1, 0, 1., 0.], [1, 0, 1., 0.], [1, 0, 1., 0.]]),
            0,
           [0, 0]  # wolfwolf collide but no reward
           )# two wolves catch
          )
    @unpack
    def testRewardWolfNoCatch(self, state, action, nextState, individual, trueWolfReward):
        rewardWolf = RewardWolf(self.wolvesID, self.sheepsID, self.entitiesSizeList, self.isCollision, self.collisionReward, individual)
        wolfReward = rewardWolf(state, action, nextState)
        self.assertEqual(tuple(wolfReward), tuple(trueWolfReward))

    @data((np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
           [1, 1, 1, 1],
           np.array([[0, 0, 1., 0.], [0, 0.2, 1., 0.], [1, 0, 1., 0.], [1, 0, 1., 0.], [1, 0, 1., 0.]]),
           0,
           [10, 10]  # one wolf one sheep collide
           ),
          (np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
           [1, 1, 1, 1],
           np.array([[0, 0, 1., 0.], [0, 0.2, 1., 0.], [1, 0, 1., 0.], [1, 0, 1., 0.], [1, 0, 1., 0.]]),
           .2,
           [12, 8]  # one wolf one sheep collide
           ),
          (np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
           [1, 1, 1, 1],
           np.array([[0, 0, 1., 0.], [0, 0.2, 1., 0.], [1, 0, 1., 0.], [1, 0, 1., 0.], [1, 0, 1., 0.]]),
           1,
           [20, 0]  # one wolf one sheep collide
           )
          )
    @unpack
    def testRewardWolfOneCatch(self, state, action, nextState, individual, trueWolfReward):
        rewardWolf = RewardWolf(self.wolvesID, self.sheepsID, self.entitiesSizeList, self.isCollision, self.collisionReward, individual)
        wolfReward = rewardWolf(state, action, nextState)
        self.assertEqual(tuple(wolfReward), tuple(trueWolfReward))

    @data((np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
           [1, 1, 1, 1],
           np.array([[0, 1, 1., 0.], [0.1, 1, 1., 0.], [-0.25, 1, 1., 0.], [1, 0, 1., 0.], [1, 0, 1., 0.]]),
           1,
           [20, 20]
           ),  # two wolves catch
          (np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
           [1, 1, 1, 1],
           np.array([[0, 1, 1., 0.], [0.1, 1, 1., 0.], [-0.25, 1, 1., 0.], [1, 0, 1., 0.], [1, 0, 1., 0.]]),
           0.3,
           [20, 20]
           ),  # two wolves catch
          (np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
           [1, 1, 1, 1],
           np.array([[0, 1, 1., 0.], [0.1, 1, 1., 0.], [-0.25, 1, 1., 0.], [1, 0, 1., 0.], [1, 0, 1., 0.]]),
           0,
           [20, 20]
           )  # two wolves catch
          )
    @unpack
    def testRewardWolfTwoCatch(self, state, action, nextState, individual, trueWolfReward):
        rewardWolf = RewardWolf(self.wolvesID, self.sheepsID, self.entitiesSizeList, self.isCollision, self.collisionReward, individual)
        wolfReward = rewardWolf(state, action, nextState)
        self.assertEqual(tuple(wolfReward), tuple(trueWolfReward))

@ddt
class TestMultiAgentEnv3Wolves(unittest.TestCase):
    def setUp(self):
        self.wolvesID = [0, 1, 2]
        self.sheepsID = [3]
        self.blocksID = [4, 5]
        self.isCollision = IsCollision(getPosFromAgentState)
        self.entitiesSizeList = [0.1] * len(self.sheepsID) + [0.2] * len(self.wolvesID) + [0.5] * len(self.blocksID)

        self.collisionReward = len(self.wolvesID) * 10 # only for this test
        self.collisionPunishment = self.collisionReward
        self.punishForOutOfBound = PunishForOutOfBound()
        self.rewardSheep = RewardSheep(self.wolvesID, self.sheepsID, self.entitiesSizeList, getPosFromAgentState, self.isCollision, self.punishForOutOfBound,
                                       self.collisionPunishment)

        self.numTotalAgents = len(self.sheepsID) + len(self.wolvesID)
        self.numBlocks = len(self.blocksID)
        self.reset = ResetMultiAgentChasing(self.numTotalAgents, self.numBlocks)

    @data((np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
           [1, 1, 1, 1],
           np.array([[1,  0, 1.,0.], [1,  0, 1.,0.], [1,  0, 1.,0.], [0, 1, 1., 0.], [1,  0, 1.,0.], [0, 0, 0, 0]]),
           1,
           [0, 0, 0] # wolfwolf collide but no reward
           ),
          (np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
           [1, 1, 1, 1],
           np.array([[1,  0, 1.,0.], [1,  0, 1.,0.], [1,  0, 1.,0.], [0, 1, 1., 0.], [1,  0, 1.,0.], [0, 0, 0, 0]]),
            .2,
           [0, 0, 0]  # wolfwolf collide but no reward
           ),
          (np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
           [1, 1, 1, 1],
           np.array([[1,  0, 1.,0.], [1,  0, 1.,0.], [1,  0, 1.,0.], [0, 1, 1., 0.], [1,  0, 1.,0.], [0, 0, 0, 0]]),
            0,
           [0, 0, 0]  # wolfwolf collide but no reward
           )
          )
    @unpack
    def testRewardWolfNoCatch(self, state, action, nextState, individual, trueWolfReward):
        rewardWolf = RewardWolf(self.wolvesID, self.sheepsID, self.entitiesSizeList, self.isCollision, self.collisionReward, individual)
        wolfReward = rewardWolf(state, action, nextState)
        self.assertEqual(tuple(wolfReward), tuple(trueWolfReward))

    @data((np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
           [1, 1, 1, 1],
           np.array([[0, 0.2, 1., 0.], [1, 0, 1., 0.], [1, 0, 1., 0.], [0, 0, 1., 0.], [1, 0, 1., 0.], [0, 0, 0, 0]]),
           0,
           [10, 10, 10]  # one wolf one sheep collide
           ),
          (np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
           [1, 1, 1, 1],
           np.array([[0, 0.2, 1., 0.], [1, 0, 1., 0.], [1, 0, 1., 0.], [0, 0, 1., 0.], [1, 0, 1., 0.], [0, 0, 0, 0]]),
           .2,
           [30*0.2 + 30*0.8/3, 30*0.8/3, 30*0.8/3]  # one wolf one sheep collide
           ),
          (np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
           [1, 1, 1, 1],
           np.array([[0, 0.2, 1., 0.], [1, 0, 1., 0.], [1, 0, 1., 0.], [0, 0, 1., 0.], [1, 0, 1., 0.], [0, 0, 0, 0]]),
           1,
           [30, 0, 0]  # one wolf one sheep collide
           )
          )
    @unpack
    def testRewardWolfOneCatch(self, state, action, nextState, individual, trueWolfReward):
        rewardWolf = RewardWolf(self.wolvesID, self.sheepsID, self.entitiesSizeList, self.isCollision, self.collisionReward, individual)
        wolfReward = rewardWolf(state, action, nextState)
        self.assertEqual(tuple(wolfReward), tuple(trueWolfReward))

    @data((np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
           [1, 1, 1, 1],
           np.array([[0.1, 1, 1., 0.], [-0.25, 1, 1., 0.], [1, 0, 1., 0.], [0, 1, 1., 0.], [1, 0, 1., 0.], [0, 0, 0, 0]]),
           1,
           [30, 30, 0]
           ),  # two wolves catch
          (np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
           [1, 1, 1, 1],
           np.array([[0.1, 1, 1., 0.], [-0.25, 1, 1., 0.], [1, 0, 1., 0.], [0, 1, 1., 0.], [1, 0, 1., 0.], [0, 0, 0, 0]]),
           0.3,
           [30*0.3 + 30* 0.7/3 + 30*0.7/3, 30*0.3 + 30* 0.7/3 + 30*0.7/3, 30* 0.7/3 + 30*0.7/3]
           ),  # two wolves catch
          (np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
           [1, 1, 1, 1],
           np.array([[0.1, 1, 1., 0.], [-0.25, 1, 1., 0.], [1, 0, 1., 0.], [0, 1, 1., 0.], [1, 0, 1., 0.], [0, 0, 0, 0]]),
           0,
           [20, 20, 20]
           )  # two wolves catch
          )
    @unpack
    def testRewardWolfTwoCatch(self, state, action, nextState, individual, trueWolfReward):
        rewardWolf = RewardWolf(self.wolvesID, self.sheepsID, self.entitiesSizeList, self.isCollision, self.collisionReward, individual)
        wolfReward = rewardWolf(state, action, nextState)
        self.assertEqual(tuple(wolfReward), tuple(trueWolfReward))


if __name__ == '__main__':
    unittest.main()
