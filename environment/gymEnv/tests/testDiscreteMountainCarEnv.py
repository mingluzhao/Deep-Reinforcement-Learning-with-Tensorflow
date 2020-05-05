import sys
sys.path.append("..")
import unittest
from ddt import ddt, data, unpack
from environment.gymEnv.gymDiscreteMountainCarEnv import *
from environment.gymEnv.discreteMountainCarEnv import *

@ddt
class TestPendulumEnv(unittest.TestCase):
    def setUp(self):
        self.seed = 1

    def testReset(self):
        env = MountainCarEnv()
        env.seed(self.seed)
        initState_gym = env.reset()

        reset = ResetMountCarDiscrete(self.seed)
        initState = reset()

        self.assertEqual(tuple(initState_gym), tuple(initState))


    @data(([1]),
          ([0]))
    @unpack
    def testInitialTransition(self, action):
        env = MountainCarEnv()
        env.seed(self.seed)
        env.reset()
        nextState_gym, reward_gym, terminal, info = env.step(action)

        reset = ResetMountCarDiscrete(self.seed)
        state = reset()

        transit = TransitMountCarDiscrete()
        action = [action]
        nextState = transit(state, action)

        self.assertEqual(tuple(nextState), tuple(nextState_gym))


    @data(([[1, 2], 1]),
          ([[10,10], 0]))
    @unpack
    def testTransition(self, state, action):
        env = MountainCarEnv()
        env.seed(self.seed)
        env.state = state
        nextState_gym, reward_gym, terminal, info = env.step(action)

        transit = TransitMountCarDiscrete()
        action = [action]
        nextState = transit(state, action)

        self.assertEqual(tuple(nextState_gym), tuple(nextState))


    @data(([[1, 1], 1]),
          ([[1, 1], 0]),
          ([[10,10], 1]),
          ([[0.01, 0.02], 1])
          )
    @unpack
    def checkTerminal(self, state, action):
        env = MountainCarEnv()
        env.seed(self.seed)
        env.state = state
        nextState_gym, reward_gym, terminal_gym, info = env.step(action)

        transit = TransitMountCarDiscrete()
        action = [action]
        nextState = transit(state, action)

        isTerminal = IsTerminalMountCarDiscrete()
        terminal = isTerminal(nextState)

        self.assertEqual(terminal_gym, terminal)

    @data(([[1, 1], 1]),
          ([[1, 1], 0]),
          ([[10,10], 1]),
          ([[0.01, 0.02], 1])
          )
    @unpack
    def testReward(self, state, action):
        env = MountainCarEnv()
        env.seed(self.seed)
        env.state = state
        nextState_gym, reward_gym, terminal, info = env.step(action)

        getReward = rewardMountCarDiscrete
        action = [action]
        reward = getReward(state, action)

        self.assertEqual(reward_gym, reward)


if __name__ == '__main__':
    unittest.main()
