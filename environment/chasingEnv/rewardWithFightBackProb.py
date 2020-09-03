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


# --- continuous zone ----

class GetAgentEffectiveIndex:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, distanceToSheep):
        effectIndex = 1 / (1 + np.exp(self.a * distanceToSheep - self.b))
        return effectIndex


class GetFightBackFromEffIndex:
    def __init__(self, initHurtProb):
        self.initHurtProb = initHurtProb

    def __call__(self, effectIndexList):
        totalEffect = np.sum(effectIndexList)
        hurtProb = self.initHurtProb / totalEffect

        return hurtProb


class GetFightBackProb:
    def __init__(self, getAgentEffectiveIndex, getPosFromState, getFightBackFromEffIndex, computeVectorNorm):
        self.getAgentIndex = getAgentEffectiveIndex
        self.getPosFromState = getPosFromState
        self.computeVectorNorm = computeVectorNorm
        self.getFightBackFromEffIndex = getFightBackFromEffIndex

    def __call__(self, wolvesState, sheepState):
        wolvesPos = [self.getPosFromState(wolfState) for wolfState in wolvesState]
        sheepPos = self.getPosFromState(sheepState)
        distsToSheep = [self.computeVectorNorm(np.array(sheepPos) - np.array(wolfPos)) for wolfPos in wolvesPos]
        effectIndexList = [self.getAgentIndex(dist) for dist in distsToSheep]
        fightBackProb = self.getFightBackFromEffIndex(effectIndexList)
        # print('wolvesState', np.round(wolvesPos, 2), 'sheepState', np.round(sheepPos, 2), 'effectIndexList', effectIndexList)

        return fightBackProb



class GetRewardFromFightBackProb:
    def __init__(self, biteReward, killReward, fightedBackReward, killProportion, sampleFromDistribution):
        assert fightedBackReward < 0
        assert biteReward > 0
        assert killReward > biteReward

        self.biteReward = biteReward
        self.killReward = killReward
        self.fightedBackReward = fightedBackReward

        self.killProportion = killProportion
        self.sampleFromDistribution = sampleFromDistribution

    def __call__(self, fightBackProb):
        killProb = (1 - fightBackProb) * self.killProportion
        biteProb = 1 - fightBackProb - killProb
        wolfRewardDist = {self.biteReward: biteProb, self.killReward: killProb, self.fightedBackReward: fightBackProb}
        # print(wolfRewardDist)
        wolfReward = self.sampleFromDistribution(wolfRewardDist)
        sheepReward = 0 if wolfReward == self.fightedBackReward else -wolfReward # kill or bite, then sheep+wolf = 0, if fightback, sheep reward = 0

        return wolfReward, sheepReward


class RewardWithHurtProb:
    def __init__(self, wolvesID, sheepsID, entitiesSizeList, isCollision, punishForOutOfBound, getPosFromState, getFightBackProb, getRewardFromFightBackProb):
        self.wolvesID = wolvesID
        self.sheepsID = sheepsID
        self.entitiesSizeList = entitiesSizeList
        self.isCollision = isCollision
        self.punishForOutOfBound = punishForOutOfBound

        self.getPosFromState = getPosFromState
        self.getFightBackProb = getFightBackProb
        self.getRewardFromFightBackProb = getRewardFromFightBackProb

    def __call__(self, state, action, nextState):
        wolvesNextState = [nextState[wolfID] for wolfID in self.wolvesID]
        wolvesSize = [self.entitiesSizeList[wolfID] for wolfID in self.wolvesID]
        rewardList = [0] * (len(self.wolvesID) + len(self.sheepsID))

        for sheepID in self.sheepsID:
            sheepNextState = nextState[sheepID]
            sheepNextPos = self.getPosFromState(sheepNextState)
            rewardList[sheepID] -= self.punishForOutOfBound(sheepNextPos)

            sheepSize = self.entitiesSizeList[sheepID]
            collisionCount = np.sum([self.isCollision(wolfNextState, sheepNextState, wolfSize, sheepSize) for wolfNextState, wolfSize in zip(wolvesNextState, wolvesSize)]) #num wolves collided with the sheep

            for wolfID in self.wolvesID:
                wolfSize = self.entitiesSizeList[wolfID]
                wolfNextState = nextState[wolfID]

                if self.isCollision(wolfNextState, sheepNextState, wolfSize, sheepSize):
                    fightBackProbTot = self.getFightBackProb(wolvesNextState, sheepNextState)
                    fightBackProb = fightBackProbTot/ collisionCount # sheeps have equal prob of attacking any wolf that collided with it
                    wolfReward, sheepReward = self.getRewardFromFightBackProb(fightBackProb)
                    rewardList[wolfID] += wolfReward
                    rewardList[sheepID] += sheepReward

        return rewardList

