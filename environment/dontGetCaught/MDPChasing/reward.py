import numpy as np

class RewardFunctionTerminalPenalty():
    def __init__(self, aliveBouns, deathPenalty, isTerminal):
        self.aliveBouns = aliveBouns
        self.deathPenalty = deathPenalty
        self.isTerminal = isTerminal
    def __call__(self, state, action, nextState):
        if self.isTerminal(nextState):
            reward = self.deathPenalty
        else:
            reward = self.aliveBouns
        return reward
