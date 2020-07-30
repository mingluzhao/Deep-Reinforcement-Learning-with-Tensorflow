import numpy as np


def sampleFromDistribution(distribution):
    hypotheses = list(distribution.keys())
    probs = list(distribution.values())
    normlizedProbs = [prob / sum(probs) for prob in probs]
    selectedIndex = list(np.random.multinomial(1, normlizedProbs)).index(1)
    selectedHypothesis = hypotheses[selectedIndex]
    return selectedHypothesis

def computeVectorNorm(vector):
    return np.power(np.power(vector, 2).sum(), 0.5)

class RewardWolfWithHurtProb:
    def __init__(self, wolvesID, sheepsID, entitiesSizeList, isCollision, getHurtProbOfCatching, sampleFromDistribution, individualWolf, hurtReward = -5, collisionReward=10):
        self.wolvesID = wolvesID
        self.sheepsID = sheepsID
        self.entitiesSizeList = entitiesSizeList
        self.isCollision = isCollision
        self.sampleFromDistribution = sampleFromDistribution
        self.collisionReward = collisionReward
        self.getHurtProbOfCatching = getHurtProbOfCatching
        self.individualWolf = individualWolf
        self.hurtReward = hurtReward

    def __call__(self, state, action, nextState):
        wolvesNetState = [nextState[wolfID] for wolfID in self.wolvesID]
        rewardList = []

        for wolfID in self.wolvesID:
            wolfReward = 0
            wolfSize = self.entitiesSizeList[wolfID]
            wolfNextState = nextState[wolfID]
            for sheepID in self.sheepsID:
                sheepSize = self.entitiesSizeList[sheepID]
                sheepNextState = nextState[sheepID]

                if self.isCollision(wolfNextState, sheepNextState, wolfSize, sheepSize):
                    getHurtProb = self.getHurtProbOfCatching(wolvesNetState, sheepNextState)
                    rewardDist = {self.hurtReward: getHurtProb, self.collisionReward: 1-getHurtProb}
                    reward = self.sampleFromDistribution(rewardDist)
                    # print(getHurtProb, rewardDist, reward)
                    wolfReward += reward # now: share the potential risk for shared wolves
                    '''
                    0.2 {-5: 0.2, 10: 0.8} 10
                    0.2 {-5: 0.2, 10: 0.8} -5
                    0.2 {-5: 0.2, 10: 0.8} 10
                    [15, 15, 15] (shared)
                    '''
            rewardList.append(wolfReward)

        if not self.individualWolf:
            rewardList = [np.sum(rewardList)] * len(self.wolvesID)

        return rewardList


class GetHurtProbOfCatchingByDeterministicZone:
    def __init__(self, getPosFromState, computeVectorNorm, sensitiveZoneRadius, oneWolfSelfHurtProb):
        self.getPosFromState = getPosFromState
        self.sensitiveZoneRadius = sensitiveZoneRadius
        self.oneWolfSelfHurtProb = oneWolfSelfHurtProb
        self.computeVectorNorm = computeVectorNorm

    def __call__(self, wolvesState, sheepState):
        wolvesPos = [self.getPosFromState(wolfState) for wolfState in wolvesState]
        sheepPos = self.getPosFromState(sheepState)
        distsToSheep = [self.computeVectorNorm(np.array(sheepPos) - np.array(wolfPos)) for wolfPos in wolvesPos]
        numWolvesWithinSentiveZone = np.sum([dist <= self.sensitiveZoneRadius for dist in distsToSheep])
        getHurtProb = self.oneWolfSelfHurtProb* (0.5 ** (numWolvesWithinSentiveZone-1))
        return getHurtProb
