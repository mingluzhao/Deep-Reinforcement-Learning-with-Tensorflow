import sys
sys.path.append("..")
import unittest
from ddt import ddt, data, unpack
from environment.gymEnv.gymPendulumEnv import *
from environment.gymEnv.pendulumEnv import *
import random

@ddt
class TestPendulumEnv(unittest.TestCase):
    def setUp(self):
        self.seed = 1

    def testObserve(self):
        env = PendulumEnv()
        env.seed(self.seed)
        initialObs_gym = env.reset()
        initialObs = observe(env.state)
        self.assertEqual(tuple(initialObs_gym), tuple(initialObs))

    def testReset(self):
        env = PendulumEnv()
        env.seed(self.seed)
        initialObs_gym = env.reset()

        reset = ResetGymPendulum(self.seed)
        initState = reset()
        initObservation = observe(initState)

        self.assertEqual(tuple(initialObs_gym), tuple(initObservation))


    @data(([[1]]),
          ([[10]]))
    @unpack
    def testInitialTransition(self, action):
        env = PendulumEnv()
        env.seed(self.seed)
        env.reset()
        nextObs_gym, reward_gym, terminal, info = env.step(action)

        reset = ResetGymPendulum(self.seed)
        state = reset()

        transit = TransitGymPendulum()
        nextState = transit(state, action)
        nextObservation = observe(nextState)

        self.assertEqual(tuple(nextObs_gym), tuple(nextObservation))


    @data(([[1, 0], [1]]),
          ([[10,10], [10]]))
    @unpack
    def testTransition(self, state, action):
        env = PendulumEnv()
        env.seed(self.seed)
        env.state = state
        nextObs_gym, reward_gym, terminal, info = env.step(action)

        transit = TransitGymPendulum()
        nextState = transit(state, action)
        nextObservation = observe(nextState)

        self.assertEqual(tuple(nextObs_gym), tuple(nextObservation))


    @data(([[1, 2], [1]]),
          ([[10,10], [10]]))
    @unpack
    def testReward(self, state, action):
        env = PendulumEnv()
        env.seed(self.seed)
        env.state = state
        nextObs_gym, reward_gym, terminal, info = env.step(action)

        transit = TransitGymPendulum()
        nextState = transit(state, action)
        getReward = RewardGymPendulum(angle_normalize)
        reward = getReward(state, action, nextState)

        self.assertEqual(reward_gym, reward)

    def testTrajectory(self):
        env = PendulumEnv()
        env.seed(self.seed)
        env.reset()

        reset = ResetGymPendulum(self.seed)
        state = reset()

        self.assertEqual(tuple(state), tuple(env.state))

        transit = TransitGymPendulum()
        getReward = RewardGymPendulum(angle_normalize)

        for timeStep in range(10000):
            action = [random.random() * random.randrange(2)]
            nextState = transit(state, action)
            nextObservation = observe(nextState)
            reward = getReward(state, action, nextState)
            nextObs_gym, reward_gym, terminal_gym, info = env.step(action)

            self.assertEqual(tuple(nextObservation), tuple(nextObs_gym))
            self.assertEqual(reward, reward_gym)

            state = nextState



if __name__ == '__main__':
    unittest.main()
