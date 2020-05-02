class RewardFunctionCompete():
    def __init__(self, aliveBonus, deathPenalty, isTerminal):
        self.aliveBonus = aliveBonus
        self.deathPenalty = deathPenalty
        self.isTerminal = isTerminal

    def __call__(self, state):
        reward = self.aliveBonus
        if self.isTerminal(state):
            reward += self.deathPenalty

        return reward