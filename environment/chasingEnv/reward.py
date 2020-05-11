class RewardFunctionCompete():
    def __init__(self, aliveBonus, deathPenalty, isTerminal):
        self.aliveBonus = aliveBonus
        self.deathPenalty = deathPenalty
        self.isTerminal = isTerminal

    def __call__(self, state, action, nextState):
        reward = self.aliveBonus
        if self.isTerminal(nextState):
            reward += self.deathPenalty

        return reward