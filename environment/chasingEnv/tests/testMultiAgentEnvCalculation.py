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
class TestMultiAgentEnv(unittest.TestCase):
    def setUp(self):
        self.sheepsID = [0]
        self.wolvesID = [1, 2]
        self.blocksID = [3, 4]
        self.isCollision = IsCollision(getPosFromAgentState)
        self.entitiesSizeList = [0.1] * len(self.sheepsID) + [0.2] * len(self.wolvesID) + [0.5] * len(self.blocksID)

        self.collisionReward = len(self.wolvesID) * 10 # only for this test
        self.collisionPunishment = self.collisionReward
        self.rewardWolf = RewardWolf(self.wolvesID, self.sheepsID, self.entitiesSizeList, self.isCollision, self.collisionReward, individual = False)
        self.punishForOutOfBound = PunishForOutOfBound()
        self.rewardSheep = RewardSheep(self.wolvesID, self.sheepsID, self.entitiesSizeList, getPosFromAgentState, self.isCollision, self.punishForOutOfBound,
                                       self.collisionPunishment)

        self.numTotalAgents = len(self.sheepsID) + len(self.wolvesID)
        self.numBlocks = len(self.blocksID)
        self.reset = ResetMultiAgentChasing(self.numTotalAgents, self.numBlocks)

        self.rewardWolfIndividual =RewardWolf(self.wolvesID, self.sheepsID, self.entitiesSizeList, self.isCollision, self.collisionReward, individual = True)


    @data(([0, 1, 1., 0.], [1,  0,  1.,  0.], 0.1, 0.2, False),
          ([0, 0, 1., 0.], [0, 0.2, 1., 0.], 0.1, 0.2, True),
          ([-0.5, 0,  1.,  0.], [-0.4 ,  0.1,  1.,  0.], 0.1, 0.05, True))
    @unpack
    def testCollisionCheck(self, state1, state2, size1, size2, trueCollision):
        collide = self.isCollision(state1, state2, size1, size2)
        self.assertEqual(collide, trueCollision)


    @data((np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
           [1, 1, 1, 1],
           np.array([[0, 1, 1., 0.], [1,  0, 1.,0.], [1,  0, 1.,0.], [1,  0, 1.,0.], [1,  0, 1.,0.]]),
           [0, 0] # wolfwolf collide but no reward
           ),
          (np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
           [1, 1, 1, 1],
           np.array([[0, 0, 1., 0.], [0, 0.2, 1., 0.], [1, 0, 1., 0.], [1, 0, 1., 0.], [1, 0, 1., 0.]]),
           [10, 10] # one wolf one sheep collide
           ),
          (np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
           [1, 1, 1, 1],
           np.array([[0, 1, 1., 0.], [0.1, 1, 1., 0.], [-0.25, 1, 1., 0.], [1, 0, 1., 0.], [1, 0, 1., 0.]]),
           [20, 20]
           ) # two wolves catch
          )
    @unpack
    def testRewardWolf(self, state, action, nextState, trueWolfReward):
        wolfReward = self.rewardWolf(state, action, nextState)
        self.assertEqual(tuple(wolfReward), tuple(trueWolfReward))

    @data(([0, 0.8], 0),
          ([0.8, 0.8], 0),
          ([0.95, 0.8], 0.5),
          ([0.95, 0.95], 1.0),
          ([0, 10], 10),
          ([10, 10], 20),
          ([0, 2], np.exp(2))
          )
    @unpack
    def testPunishOutofBound(self, agentPos, truePunishment):
        punishment = self.punishForOutOfBound(agentPos)
        self.assertAlmostEqual(punishment, truePunishment)

    @data((np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
           [1, 1, 1, 1],
           np.array([[0, 1, 1., 0.], [1,  0, 1.,0.], [1,  0, 1.,0.], [1,  0, 1.,0.], [1,  0, 1.,0.]]),
           [-np.exp(2 * 1 - 2)] # not caught, out of boundary
           ),
          (np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
           [1, 1, 1, 1],
           np.array([[0, 0, 1., 0.], [0, 0.2, 1., 0.], [1, 0, 1., 0.], [1, 0, 1., 0.], [1, 0, 1., 0.]]),
           [-20] # caught by one, not out of bound
           ),
          (np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
           [1, 1, 1, 1],
           np.array([[0, 1, 1., 0.], [0.1, 1, 1., 0.], [-0.25, 1, 1., 0.], [1, 0, 1., 0.], [1, 0, 1., 0.]]),
           [-40 - np.exp(2 * 1 - 2)]
           ) # two wolves catch, out of bound
          )
    @unpack
    def testRewardSheep(self, state, action, nextState, trueSheepReward):
        sheepReward = self.rewardSheep(state, action, nextState)
        self.assertEqual(tuple(sheepReward), tuple(trueSheepReward))

    def testReset(self):
        self.assertEqual(self.reset().shape, (5, 4))

    @data((0, (12, )),
          (1, (14, )),
          (2, (14,))
          )
    @unpack
    def testObserve(self, agentID, trueObsShape):
        state = np.array([[0, 1, 1., 0.], [1, 0, 1.,0.], [1,  0, 1.,0.], [1,  0, 1.,0.], [1,  0, 1.,0.]])
        observe = Observe(agentID, self.wolvesID, self.sheepsID, self.blocksID, getPosFromAgentState, getVelFromAgentState)
        obsShape = observe(state).shape
        self.assertEqual(obsShape, trueObsShape)


    @data((np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
           [1, 1, 1, 1],
           np.array([[0, 1, 1., 0.], [1, 0, 1., 0.], [1, 0, 1., 0.], [1, 0, 1., 0.], [1, 0, 1., 0.]]),
           [0, 0]  # wolfwolf collide but no reward
           ),
          (np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
           [1, 1, 1, 1],
           np.array([[0, 0, 1., 0.], [0, 0.2, 1., 0.], [1, 0, 1., 0.], [1, 0, 1., 0.], [1, 0, 1., 0.]]),
           [20, 0]  # one wolf one sheep collide
           ),
          (np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
           [1, 1, 1, 1],
           np.array([[0, 1, 1., 0.], [0.1, 1, 1., 0.], [-0.25, 1, 1., 0.], [1, 0, 1., 0.], [1, 0, 1., 0.]]),
           [20, 20]
           )  # two wolves catch
          )
    @unpack
    def testRewardWolfIndividual(self, state, action, nextState, trueWolfReward):
        wolfReward = self.rewardWolfIndividual(state, action, nextState)
        self.assertEqual(tuple(wolfReward), tuple(trueWolfReward))


if __name__ == '__main__':
    unittest.main()
