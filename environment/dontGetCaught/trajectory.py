
import numpy as np
import random

class ForwardOneStep:
    def __init__(self, transitionFunction, rewardFunction):
        self.transitionFunction = transitionFunction
        self.rewardFunction = rewardFunction


    def __call__(self, state, sampleAction):
        action = sampleAction(state)
        nextState = self.transitionFunction(state, action)
        reward = self.rewardFunction(state, action, nextState)
        return (state, action, nextState, reward)


class SampleTrajectory:
    def __init__(self, maxRunningSteps, isTerminal, resetState, forwardOneStep):
        self.maxRunningSteps = maxRunningSteps
        self.isTerminal = isTerminal
        self.resetState = resetState
        self.forwardOneStep = forwardOneStep

    def __call__(self, sampleAction):            
        state = self.resetState()
        while self.isTerminal(state):
            state = self.resetState()

        trajectory = []
        for runningStep in range(self.maxRunningSteps):
            if self.isTerminal(state):
                trajectory.append((state, None, None, 0))
                break
            state, action, nextState, reward = self.forwardOneStep(state, sampleAction)
            trajectory.append((state, action, nextState, reward))
            state = nextState

        return trajectory


