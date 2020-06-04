import numpy as np

class SampleTrajectory:
    def __init__(self, maxRunningSteps, transit, isTerminal, rewardFunc, reset):
        self.maxRunningSteps = maxRunningSteps
        self.transit = transit
        self.isTerminal = isTerminal
        self.rewardFunc = rewardFunc
        self.reset = reset

    def __call__(self, policy):
        # epsReward = np.array([0, 0, 0])
        state = self.reset()
        while self.isTerminal(state):
            state = self.reset()

        trajectory = []
        for runningStep in range(self.maxRunningSteps):
            if self.isTerminal(state):
                # print('terminal------------')
                break
            action = policy(state)
            nextState = self.transit(state, action)

            reward = self.rewardFunc(state, action, nextState)
            # print('state: ', state, 'action: ', action, 'nextState: ', nextState, 'reward: ', reward)
            # epsReward += reward

            trajectory.append((state, action, reward, nextState))

            state = nextState
        # print('epsReward: ', epsReward)
        return trajectory

