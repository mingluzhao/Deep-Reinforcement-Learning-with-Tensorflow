import  numpy as np
class RewardFunctionCompete():
    def __init__(self, aliveBonus, deathPenalty, isTerminal):
        self.aliveBonus = aliveBonus
        self.deathPenalty = deathPenalty
        self.isTerminal = isTerminal

    def __call__(self, state, action, nextState):
        reward = self.aliveBonus
        if self.isTerminal(nextState):
            reward -= self.deathPenalty

        return reward

# class GetBoundaryPunishment:
#     def __init__(self, xBoundary, yBoundary, x2ndBoundary, y2ndBoundary, sheepIndex = 0, punishmentVal = 10):
#         self.xMin, self.xMax = xBoundary
#         self.yMin, self.yMax = yBoundary
#         self.x2ndMin, self.x2ndMax = x2ndBoundary
#         self.y2ndMin, self.y2ndMax = y2ndBoundary
#         self.sheepIndex = sheepIndex
#         self.punishmentVal = punishmentVal
#
#     def __call__(self, intendedState):
#         sheepX, sheepY = intendedState[self.sheepIndex]
#         cost = 0
#         if self.xMin < sheepX < self.xMax and self.yMin < sheepY < self.yMax: return 0
#
#         cost += (self.xMin - sheepX) * 10 if sheepX > self.x2ndMin
#
#         cost = 0
#         cost += max(0, sheepX - self.xMax) * self.punishmentVal
#         cost += max(0, self.xMin - sheepX) * self.punishmentVal
#         cost += max(0, sheepY - self.yMax) * self.punishmentVal
#         cost += max(0, self.yMin - sheepY) * self.punishmentVal
#
#         def bound(x):
#             if x < 0.9:
#                 return 0
#             if x < 1.0:
#                 return (x - 0.9) * 10
#             return min(np.exp(2 * x - 2), 10)
#
#         return cost


class GetBoundaryPunishment:
    def __init__(self, xBoundary, yBoundary, sheepIndex=0, punishmentVal=10):
        self.xMin, self.xMax = xBoundary
        self.yMin, self.yMax = yBoundary
        self.sheepIndex = sheepIndex
        self.punishmentVal = punishmentVal

    def __call__(self, intendedState):
        sheepX, sheepY = intendedState[self.sheepIndex]

        cost = 0
        cost += max(0, sheepX - self.xMax) * self.punishmentVal
        cost += max(0, self.xMin - sheepX) * self.punishmentVal
        cost += max(0, sheepY - self.yMax) * self.punishmentVal
        cost += max(0, self.yMin - sheepY) * self.punishmentVal

        return min(cost, 20)



class RewardSheepWithBoundaryHeuristics:
    def __init__(self, rewardSheep, getIntendedNextState, getBoundaryPunishment, getSheepPos):
        self.rewardSheep = rewardSheep
        self.getIntendedNextState = getIntendedNextState
        self.getBoundaryPunishment = getBoundaryPunishment
        self.getSheepPos = getSheepPos

    def __call__(self, state, action, nextState):
        reward = self.rewardSheep(state, action, nextState)
        sheepState = self.getSheepPos(state)
        sheepIntendedNextState = self.getIntendedNextState(sheepState, action)
        boundaryPunishment = self.getBoundaryPunishment(sheepIntendedNextState)
        reward -= boundaryPunishment
        return reward

class GetActionCost:
    def __init__(self, actionCostRate):
        self.actionCostRate = actionCostRate

    def __call__(self, action):
        actionMagnitude = np.linalg.norm(np.array(action), ord=2)
        cost = self.actionCostRate* actionMagnitude

        return cost


class RewardWithActionCost:
    def __init__(self, rewardSheep, getActionCost):
        self.rewardSheep = rewardSheep
        self.getActionCost = getActionCost

    def __call__(self, state, action, nextState):
        reward = self.rewardSheep(state, action, nextState)
        actionCost = self.getActionCost(action)
        reward -= actionCost

        return reward
