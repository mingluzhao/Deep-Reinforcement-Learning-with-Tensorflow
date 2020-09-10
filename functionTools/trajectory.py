import numpy as np

class SampleTrajectory:
    def __init__(self, maxRunningSteps, transit, isTerminal, rewardFunc, reset):
        self.maxRunningSteps = maxRunningSteps
        self.transit = transit
        self.isTerminal = isTerminal
        self.rewardFunc = rewardFunc
        self.reset = reset

    def __call__(self, policy):
        epsReward = np.array([0, 0, 0, 0, 0])
        state = self.reset()

        trajectory = []
        for runningStep in range(self.maxRunningSteps):

            action = policy(state)
            nextState = self.transit(state, action)
            reward = self.rewardFunc(state, action, nextState)
            trajectory.append((state, action, reward, nextState))
            state = nextState
            epsReward = epsReward + np.array(reward)
            if self.isTerminal(state):
                # print('terminal------------')
                break
        print('eps reward ', np.round(epsReward[:-1], 2))

        return trajectory


class SampleTrajectoryResetAtTerminal:
    def __init__(self, maxRunningSteps, transit, isTerminal, rewardFunc, reset):
        self.maxRunningSteps = maxRunningSteps
        self.transit = transit
        self.isTerminal = isTerminal
        self.rewardFunc = rewardFunc
        self.reset = reset

    def __call__(self, policy):
        epsReward = np.array([0, 0, 0, 0, 0])
        state = self.reset()

        trajectory = []
        for runningStep in range(self.maxRunningSteps):
            action = policy(state)
            nextState = self.transit(state, action)
            reward = self.rewardFunc(state, action, nextState)
            trajectory.append((state, action, reward, nextState))
            state = nextState
            epsReward = epsReward + np.array(reward)
            if self.isTerminal(state):
                # print('terminal------------')
                state = self.reset()

        # print('eps reward ', np.round(epsReward[:-1], 2))

        return trajectory
