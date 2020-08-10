import sys
sys.path.append("..")
import unittest
from ddt import ddt, data, unpack
from environment.gymEnv.gymFuncWrapper_unfinished import *
import gym

@ddt
class TestPendulumEnv(unittest.TestCase):
    def setUp(self):
        self.seed = 1

    @data(
          # (['MountainCarContinuous-v0']),
          # (['Acrobot-v1'])
          (['CartPole-v1'])
          # (['MountainCar-v0']),
          # (['Pendulum-v0'])
    )
    @unpack
    def testTransit(self, envName):

        env = gym.make(envName)
        env.seed(self.seed)
        env = env.unwrapped

        getEnvFunctions = GetEnvFunctions(env)
        transit, rewardFunc, reset, isTerminal, observe = getEnvFunctions()

        state = reset()
        gymObservation= env.reset()
        env.state = state

        for i in range(100):

            if isinstance(env.action_space, gym.spaces.Box):
                print('continuous')
                action = np.random.rand(*env.action_space.shape)
            else:
                print('discrete')
                action = int(1)

            gymNextObservation, gymReward, gymTerminal, info = env.step(action)
            gymNextState = env.state

            nextState = transit(state, action)
            reward = rewardFunc(state, action, nextState)
            terminal = isTerminal(nextState)

            nextObservation = observe(nextState) if observe else nextState

            self.assertEqual(tuple(nextState), tuple(gymNextState))
            self.assertEqual(tuple(nextObservation), tuple(gymNextObservation))
            self.assertEqual(reward, gymReward)

            if terminal != gymTerminal:
                print(state, nextState, reward, terminal)
            # self.assertEqual(terminal, gymTerminal)

            state = nextState
            env.state = state




if __name__ == '__main__':
    unittest.main()
