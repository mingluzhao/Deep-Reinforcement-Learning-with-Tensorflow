import numpy as np
import random

def sampleFromDistribution(distribution):
    hypotheses = list(distribution.keys())
    probs = list(distribution.values())
    normlizedProbs = [prob / sum(probs) for prob in probs]
    selectedIndex = list(np.random.multinomial(1, normlizedProbs)).index(1)
    selectedHypothesis = hypotheses[selectedIndex]
    return selectedHypothesis

def computeVectorNorm(vector):
    return np.power(np.power(vector, 2).sum(), 0.5)


class GetAgentsPercentageOfRewards:
    def __init__(self, sensitivity, collisionDist):
        self.sensitivity = sensitivity
        self.collisionDist = collisionDist
        self.getPercent = lambda dist: (dist + 1 - self.collisionDist) ** (-self.sensitivity)
        self.individualReward = (self.sensitivity > 100)

    def __call__(self, agentsDistanceList, wolfID):
        if self.individualReward:
            percentage = np.zeros(len(agentsDistanceList))
            percentage[wolfID] = 1
            return percentage

        percentageRaw = [self.getPercent(dist) for dist in agentsDistanceList]
        percentage = np.array(percentageRaw)/ np.sum(percentageRaw)

        return percentage


class GetCollisionWolfReward:
    def __init__(self, biteReward, killReward, killProportion, sampleFromDistribution, terminalCheck):
        self.biteReward = biteReward
        self.killReward = killReward
        self.killProportion = killProportion
        self.sampleFromDistribution = sampleFromDistribution
        self.terminalCheck = terminalCheck

    def __call__(self, numWolves, killRewardPercent, collisionID):
        if self.terminalCheck.terminal: # sheep already killed
            return [0]* numWolves

        isKill = self.sampleFromDistribution({1: self.killProportion, 0: 1-self.killProportion})
        if isKill:
            reward = self.killReward * np.array(killRewardPercent)
            self.terminalCheck.isTerminal()
            # print('killed by ', collisionID, '--------------------------------------------------------')
        else:
            reward = [0]* numWolves
            reward[collisionID] = self.biteReward
            # print('      Bite by ', collisionID)
        return reward


class GetWolfSheepDistance:
    def __init__(self, computeVectorNorm, getPosFromState):
        self.computeVectorNorm = computeVectorNorm
        self.getPosFromState = getPosFromState

    def __call__(self, wolvesStates, sheepState):
        wolvesPosList = [self.getPosFromState(wolfState) for wolfState in wolvesStates]
        sheepPos = self.getPosFromState(sheepState)
        dists = [self.computeVectorNorm(np.array(sheepPos) - np.array(wolfPos)) for wolfPos in wolvesPosList]

        return dists


class TerminalCheck(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.terminal = False

    def isTerminal(self):
        self.terminal = True


class RewardWolvesWithKillProb:
    def __init__(self, wolvesID, sheepsID, entitiesSizeList, isCollision, terminalCheck,
                 getWolfSheepDistance, getAgentsPercentageOfRewards, getCollisionWolfReward):
        self.wolvesID = wolvesID
        self.sheepsID = sheepsID
        self.entitiesSizeList = entitiesSizeList
        self.isCollision = isCollision

        self.terminalCheck = terminalCheck
        self.getWolfSheepDistance = getWolfSheepDistance
        self.getAgentsPercentageOfRewards = getAgentsPercentageOfRewards
        self.getCollisionWolfReward = getCollisionWolfReward

    def __call__(self, state, action, nextState):
        self.terminalCheck.reset()

        wolvesNextState = [nextState[wolfID] for wolfID in self.wolvesID]
        numWolves = len(self.wolvesID)
        rewardList = np.zeros(numWolves)

        for sheepID in self.sheepsID:
            sheepNextState = nextState[sheepID]
            sheepSize = self.entitiesSizeList[sheepID]

            wolvesSheepDistance = self.getWolfSheepDistance(wolvesNextState, sheepNextState)

            # randomly order wolves so that when more than one wolf catches the sheep, random one samples first
            wolvesID = self.wolvesID.copy()
            random.shuffle(wolvesID)

            for wolfID in wolvesID:
                wolfSize = self.entitiesSizeList[wolfID]
                wolfNextState = nextState[wolfID]

                if self.isCollision(wolfNextState, sheepNextState, wolfSize, sheepSize):
                    killRewardPercent = self.getAgentsPercentageOfRewards(wolvesSheepDistance, wolfID)
                    wolfReward = self.getCollisionWolfReward(numWolves, killRewardPercent, wolfID)
                    rewardList = rewardList + np.array(wolfReward)

        return rewardList

