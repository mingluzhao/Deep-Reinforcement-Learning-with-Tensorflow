import sys
sys.path.append("..")
import unittest
from ddt import ddt, data, unpack
from environment.chasingEnv.envNoPhysics import IsTerminal, GetAgentPosFromState
from environment.chasingEnv.reward import RewardFunctionCompete

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

        sheepTerminalPenalty = -1
        self.rewardSheep = RewardFunctionCompete(sheepAliveBonus, sheepTerminalPenalty, isTerminal)

    @data(
        ([10, 10, 10, 10], 1/20 - 1),
        ([0, 0, 10, 10], 1/20)
    )
    @unpack
    def testReward(self, state, trueReward):
        reward = self.rewardSheep(state)
        self.assertEqual(reward, trueReward)

if __name__ == '__main__':
    unittest.main()
