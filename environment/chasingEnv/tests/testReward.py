import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..', '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
sys.path.append(os.path.join(dirName, '..'))

import numpy as np
import unittest
from ddt import ddt, data, unpack
from environment.chasingEnv.envNoPhysics import IsTerminal, GetAgentPosFromState
from environment.chasingEnv.reward import RewardFunctionCompete, GetActionCost

@ddt
class TestEnv(unittest.TestCase):
    def setUp(self):
        sheepId = 0
        wolfId = 1
        getSheepXPos = GetAgentPosFromState(sheepId)
        getWolfXPos = GetAgentPosFromState(wolfId)
        killzoneRadius = 1
        isTerminal = IsTerminal(getWolfXPos, getSheepXPos, killzoneRadius)

        maxRunningSteps = 20
        sheepAliveBonus = 1 / maxRunningSteps

        sheepTerminalPenalty = 1
        self.rewardSheep = RewardFunctionCompete(sheepAliveBonus, sheepTerminalPenalty, isTerminal)

        self.actionCostRate = 2

    @data(
        ([10, 10, 10, 10], [1], [10, 10, 10, 10], 1/20 - 1),
        ([0, 0, 10, 10], [1], [0, 0, 10, 10],  1/20)
    )
    @unpack
    def testReward(self, state, action, nextState, trueReward):
        reward = self.rewardSheep(state, action, nextState)
        self.assertEqual(reward, trueReward)

    @data(
        ((3, 4), 10),
        ((1, -1), np.sqrt(2)*2),
        ((-1, -1), np.sqrt(2) * 2)
    )
    @unpack
    def testActionCost(self, action, trueCost):
        getActionCost = GetActionCost(self.actionCostRate)
        cost = getActionCost(action)
        self.assertEqual(cost, trueCost)


if __name__ == '__main__':
    unittest.main()
