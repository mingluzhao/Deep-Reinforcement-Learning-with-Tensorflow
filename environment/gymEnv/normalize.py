import numpy as np
import gym

def env_norm(env):
    '''Normalize states (observations) and actions to [-1, 1]'''
    action_space = env.action_space
    state_space = env.observation_space

    env_type = type(env)

    class EnvNormalization(env_type):
        def __init__(self):
            self.__dict__.update(env.__dict__)
            # state (observation - o to match Gym environment class)
            if np.any(state_space.high < 1e10):
                h = state_space.high
                l = state_space.low
                self.o_c = (h+l)/2.
                self.o_sc = (h-l)/2.
            else:
                self.o_c = np.zeros_like(state_space.high)
                self.o_sc = np.ones_like(state_space.high)

            # action
            h = action_space.high
            l = action_space.low
            self.a_c = (h+l)/2.
            self.a_sc = (h-l)/2.

            # reward
            self.r_sc = 0.1
            self.r_c = 0.

            self.observation_space = gym.spaces.Box(self.filter_observation(state_space.low), self.filter_observation(state_space.high))

        def filter_observation(self, o):
            return (o - self.o_c)/self.o_sc

        def filter_action(self, a):
            return self.a_sc*a + self.a_c

        def filter_reward(self, r):
            return self.r_sc*r + self.r_c

        def step(self, a):
            ac_f = np.clip(self.filter_action(a), self.action_space.low, self.action_space.high)
            o, r, done, info = env_type.step(self, ac_f)
            o_f = self.filter_observation(o)

            return o_f, r, done, info
    fenv = EnvNormalization()
    return fenv