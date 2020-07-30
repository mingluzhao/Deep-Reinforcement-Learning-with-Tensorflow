
class RewardWolfIndividual:
    def __init__(self, wolvesID, sheepsID, entitiesSizeList, isCollision, collisionReward):
        self.wolvesID = wolvesID
        self.sheepsID = sheepsID
        self.entitiesSizeList = entitiesSizeList
        self.isCollision = isCollision
        self.collisionReward = collisionReward

    def __call__(self, state, action, nextState):
        reward = []

        for wolfID in self.wolvesID:
            currentWolfReward = 0
            wolfSize = self.entitiesSizeList[wolfID]
            wolfNextState = nextState[wolfID]
            for sheepID in self.sheepsID:
                sheepSize = self.entitiesSizeList[sheepID]
                sheepNextState = nextState[sheepID]

                if self.isCollision(wolfNextState, sheepNextState, wolfSize, sheepSize):
                    currentWolfReward += self.collisionReward

            reward.append(currentWolfReward)

        return reward