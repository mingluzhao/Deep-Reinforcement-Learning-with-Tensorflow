import numpy as np

class SampleTrajectory:
    def __init__(self, maxRunningSteps, transit, isTerminal, rewardFunc, reset):
        self.maxRunningSteps = maxRunningSteps
        self.transit = transit
        self.isTerminal = isTerminal
        self.rewardFunc = rewardFunc
        self.reset = reset

    def __call__(self, policy):
        state = self.reset()
        trajectory = []

        for runningStep in range(self.maxRunningSteps):
            action = policy(state)
            nextState = self.transit(state, action)
            reward = self.rewardFunc(state, action, nextState)
            trajectory.append((state, action, reward, nextState))
            state = nextState
            if self.isTerminal(state):
                break

        return trajectory


class SampleTrajectoryResetAtTerminal:
    def __init__(self, maxRunningSteps, transit, isTerminal, rewardFunc, reset):
        self.maxRunningSteps = maxRunningSteps
        self.transit = transit
        self.isTerminal = isTerminal
        self.rewardFunc = rewardFunc
        self.reset = reset

    def __call__(self, policy):
        state = self.reset()
        trajectory = []

        for runningStep in range(self.maxRunningSteps):
            action = policy(state)
            nextState = self.transit(state, action)
            reward = self.rewardFunc(state, action, nextState)
            trajectory.append((state, action, reward, nextState))
            state = nextState
            if self.isTerminal(state):
                state = self.reset()

        return trajectory
