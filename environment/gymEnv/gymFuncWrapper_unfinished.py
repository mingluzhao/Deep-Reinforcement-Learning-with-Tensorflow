import numpy as np
import gym

class TransitAndTerminal:
    def __init__(self, env):
        self.env = env

    def __call__(self, state, action):
        self.env.state = state
        nextObservation, reward, terminal, info = self.env.step(action)
        nextState = self.env.state

        return nextState, terminal


class Reward:
    def __init__(self, env):
        self.env = env

    def __call__(self, state, action, nextState):
        self.env.state = state
        nextState, reward, terminal, info = self.env.step(action)
        return reward


class IsTerminal:
    def __init__(self, env):
        self.env = env

    def __call__(self, nextState):
        self.env.state = nextState

        if isinstance(self.env.action_space, gym.spaces.Box):
            action = np.zeros(self.env.action_space.shape)
        else:
            action = int(0)

        nextState, reward, terminal, info = self.env.step(action)
        return terminal

class Reset:
    def __init__(self, env):
        self.env = env

    def __call__(self):
        self.env.reset()
        resetState = self.env.state

        return resetState

class Observe:
    def __init__(self, env):
        self.env = env

    def __call__(self, state):
        self.env.state = state
        observation = self.env._get_obs()
        return observation



class Env:
    def __init__(self, env):
        self.env = env

    def __call__(self):
        self.env.state = state
        nextObservation, reward, terminal, info = self.env.step(action)
        nextState = self.env.state

        return nextState, terminal



class GetEnvFunctions:
    def __init__(self, env):
        self.env = env
        self.hasObserve = getattr(self.env, "_get_obs", None)

    def __call__(self):
        observe = None
        if callable(self.hasObserve):
            observe = Observe(self.env)

        transitAndTerminal = TransitAndTerminal(self.env)
        transit = lambda state, action: transitAndTerminal(state, action)[0]
        terminal = lambda:

        reward = Reward(self.env)
        reset = Reset(self.env)
        isTerminal = IsTerminal(self.env)

        return transit, reward, reset, isTerminal, observe