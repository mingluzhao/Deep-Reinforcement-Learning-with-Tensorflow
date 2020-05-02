import sys
sys.path.append("..")
import unittest
from ddt import ddt, data, unpack
from environment.gymEnv.gymPendulumEnv import *
from environment.gymEnv.pendulumEnv import *

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

        reset = ResetGymPendulum(self.seed, observe)
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

        reset = ResetGymPendulum(self.seed, observe)
        state = reset()

        transit = TransitGymPendulum()
        nextState = transit(state, action)
        nextObservation = observe(nextState)

        self.assertEqual(tuple(nextObs_gym), tuple(nextObservation))


    @data(([[1, 2], [1]]),
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

        getReward = RewardGymPendulum(angle_normalize)
        reward = getReward(state, action)

        self.assertEqual(reward_gym, reward)





if __name__ == '__main__':
    unittest.main()
