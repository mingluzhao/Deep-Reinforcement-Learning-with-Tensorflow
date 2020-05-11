import sys
sys.path.append("..")
import unittest
from ddt import ddt, data, unpack
from environment.gymEnv.gymContinousMountainCarEnv import *
from environment.gymEnv.continousMountainCarEnv import *

@ddt
class TestPendulumEnv(unittest.TestCase):
    def setUp(self):
        self.seed = 1

    def testReset(self):
        env = Continuous_MountainCarEnv()
        env.seed(self.seed)
        initState_gym = env.reset()

        reset = ResetMountCarContin(self.seed)
        initState = reset()
        self.assertEqual(tuple(initState_gym), tuple(initState))

    @data(([[1]]),
          ([[10]]))
    @unpack
    def testInitialTransition(self, action):
        env = Continuous_MountainCarEnv()
        env.seed(self.seed)
        env.reset()
        nextState_gym, reward_gym, terminal, info = env.step(action)

        reset = ResetMountCarContin(self.seed)
        state = reset()

        transit = TransitGymMountCarContinuous()
        nextState = transit(state, action)

        self.assertEqual(tuple(nextState), tuple(nextState_gym))


    @data(([[1, 2], [1]]),
          ([[10,10], [10]]))
    @unpack
    def testTransition(self, state, action):
        env = Continuous_MountainCarEnv()
        env.seed(self.seed)
        env.state = state
        nextState_gym, reward_gym, terminal, info = env.step(action)

        transit = TransitGymMountCarContinuous()
        nextState = transit(state, action)

        self.assertEqual(tuple(nextState_gym), tuple(nextState))


    @data(([[0, 0], [2]]),
          ([[0, 0], [0.02]]),
          ([[10,10], [1]]),
          ([[0.4, 0.4], [1]])
          )
    @unpack
    def testTerminal(self, state, action):
        env = Continuous_MountainCarEnv()
        env.seed(self.seed)
        env.state = state
        nextState_gym, reward_gym, terminal_gym, info = env.step(action)

        transit = TransitGymMountCarContinuous()
        nextState = transit(state, action)

        isTerminal = IsTerminalMountCarContin()
        terminal = isTerminal(nextState)

        self.assertEqual(terminal_gym, terminal)


    @data(([[0, 0], [2]]),
          ([[0, 0], [0.02]]),
          ([[10,10], [1]]),
          ([[0.4, 0.4], [1]])
          )
    @unpack
    def testReward(self, state, action):
        env = Continuous_MountainCarEnv()
        env.seed(self.seed)
        env.state = state
        nextState_gym, reward_gym, terminal, info = env.step(action)

        isTerminal = IsTerminalMountCarContin()
        getReward = RewardMountCarContin(isTerminal)

        transit = TransitGymMountCarContinuous()
        nextState = transit(state, action)
        reward = getReward(state, action, nextState)

        self.assertEqual(reward_gym, reward)


    def testTrajectory(self):
        env = Continuous_MountainCarEnv()
        env.seed(self.seed)
        env.reset()

        reset = ResetMountCarContin(self.seed)
        state = reset()

        self.assertEqual(tuple(state), tuple(env.state))

        transit = TransitGymMountCarContinuous()
        isTerminal = IsTerminalMountCarContin()
        getReward = RewardMountCarContin(isTerminal)

        for timeStep in range(10000):
            action = [random.random() * random.randrange(2)]
            nextState = transit(state, action)
            reward = getReward(state, action, nextState)
            terminal = isTerminal(nextState)

            nextState_gym, reward_gym, terminal_gym, info = env.step(action)

            self.assertEqual(tuple(nextState), tuple(nextState_gym))
            self.assertEqual(reward, reward_gym)
            self.assertEqual(terminal, terminal_gym)

            state = nextState



if __name__ == '__main__':
    unittest.main()
