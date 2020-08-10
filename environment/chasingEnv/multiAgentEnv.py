import numpy as np

getPosFromAgentState = lambda state: np.array([state[0], state[1]])
getVelFromAgentState = lambda agentState: np.array([agentState[2], agentState[3]])

class GetActionCost:
    def __init__(self, costActionRatio, reshapeAction, individualCost):
        self.costActionRatio = costActionRatio
        self.individualCost = individualCost
        self.reshapeAction =reshapeAction

    def __call__(self, agentsActions):
        agentsActions = [self.reshapeAction(action) for action in agentsActions]
        actionMagnitude = [np.linalg.norm(np.array(action), ord=2) for action in agentsActions]
        cost = self.costActionRatio* np.array(actionMagnitude)
        numAgents = len(agentsActions)
        groupCost = cost if self.individualCost else [np.sum(cost)] * numAgents

        return groupCost


class IsCollision:
    def __init__(self, getPosFromState):
        self.getPosFromState = getPosFromState

    def __call__(self, agent1State, agent2State, agent1Size, agent2Size):
        posDiff = self.getPosFromState(agent1State) - self.getPosFromState(agent2State)
        dist = np.sqrt(np.sum(np.square(posDiff)))
        minDist = agent1Size + agent2Size
        return True if dist < minDist else False


class RewardWolf:
    def __init__(self, wolvesID, sheepsID, entitiesSizeList, isCollision, collisionReward, individual):
        self.wolvesID = wolvesID
        self.sheepsID = sheepsID
        self.entitiesSizeList = entitiesSizeList
        self.isCollision = isCollision
        self.collisionReward = collisionReward
        self.individual = float(individual) # self.individual = 0.8: 0.8* reward give myself, 0.2* reward split to other agents

    def __call__(self, state, action, nextState):
        numWolves = len(self.wolvesID)
        reward = [0]* numWolves

        individualReward = self.individual* self.collisionReward
        sharedRewardForEachAgent = (1-self.individual)* self.collisionReward/ numWolves

        for rewardID, wolfID in enumerate(self.wolvesID):
            wolfSize = self.entitiesSizeList[wolfID]
            wolfNextState = nextState[wolfID]

            for sheepID in self.sheepsID:
                sheepSize = self.entitiesSizeList[sheepID]
                sheepNextState = nextState[sheepID]

                if self.isCollision(wolfNextState, sheepNextState, wolfSize, sheepSize):
                    reward = [oldReward + sharedRewardForEachAgent for oldReward in reward]
                    reward[rewardID] += individualReward

        return reward

class PunishForOutOfBound:
    def __init__(self):
        self.physicsDim = 2

    def __call__(self, agentPos):
        punishment = 0
        for i in range(self.physicsDim):
            x = abs(agentPos[i])
            punishment += self.bound(x)
        return punishment

    def bound(self, x):
        if x < 0.9:
            return 0
        if x < 1.0:
            return (x - 0.9) * 10
        return min(np.exp(2 * x - 2), 10)


class RewardSheep:
    def __init__(self, wolvesID, sheepsID, entitiesSizeList, getPosFromState, isCollision, punishForOutOfBound,
                 collisionPunishment):
        self.wolvesID = wolvesID
        self.getPosFromState = getPosFromState
        self.entitiesSizeList = entitiesSizeList
        self.sheepsID = sheepsID
        self.isCollision = isCollision
        self.collisionPunishment = collisionPunishment
        self.punishForOutOfBound = punishForOutOfBound

    def __call__(self, state, action, nextState): #state, action not used
        reward = []
        for sheepID in self.sheepsID:
            sheepReward = 0
            sheepNextState = nextState[sheepID]
            sheepNextPos = self.getPosFromState(sheepNextState)
            sheepSize = self.entitiesSizeList[sheepID]

            sheepReward -= self.punishForOutOfBound(sheepNextPos)
            for wolfID in self.wolvesID:
                wolfSize = self.entitiesSizeList[wolfID]
                wolfNextState = nextState[wolfID]
                if self.isCollision(wolfNextState, sheepNextState, wolfSize, sheepSize):
                    sheepReward -= self.collisionPunishment
            reward.append(sheepReward)
        return reward


class ResetMultiAgentChasing:
    def __init__(self, numTotalAgents, numBlocks):
        self.positionDimension = 2
        self.numTotalAgents = numTotalAgents
        self.numBlocks = numBlocks

    def __call__(self):
        getAgentRandomPos = lambda: np.random.uniform(-1, +1, self.positionDimension)
        getAgentRandomVel = lambda: np.zeros(self.positionDimension)
        agentsState = [list(getAgentRandomPos()) + list(getAgentRandomVel()) for ID in range(self.numTotalAgents)]

        getBlockRandomPos = lambda: np.random.uniform(-0.9, +0.9, self.positionDimension)
        getBlockSpeed = lambda: np.zeros(self.positionDimension)

        blocksState = [list(getBlockRandomPos()) + list(getBlockSpeed()) for blockID in range(self.numBlocks)]
        state = np.array(agentsState + blocksState)
        return state


class Observe:
    def __init__(self, agentID, wolvesID, sheepsID, blocksID, getPosFromState, getVelFromAgentState):
        self.agentID = agentID
        self.wolvesID = wolvesID
        self.sheepsID = sheepsID
        self.blocksID = blocksID
        self.getEntityPos = lambda state, entityID: getPosFromState(state[entityID])
        self.getEntityVel = lambda state, entityID: getVelFromAgentState(state[entityID])

    def __call__(self, state):
        blocksPos = [self.getEntityPos(state, blockID) for blockID in self.blocksID]
        agentPos = self.getEntityPos(state, self.agentID)
        blocksInfo = [blockPos - agentPos for blockPos in blocksPos]

        posInfo = []
        for wolfID in self.wolvesID:
            if wolfID == self.agentID: continue
            wolfPos = self.getEntityPos(state, wolfID)
            posInfo.append(wolfPos - agentPos)

        velInfo = []
        for sheepID in self.sheepsID:
            if sheepID == self.agentID: continue
            sheepPos = self.getEntityPos(state, sheepID)
            posInfo.append(sheepPos - agentPos)
            sheepVel = self.getEntityVel(state, sheepID)
            velInfo.append(sheepVel)

        agentVel = self.getEntityVel(state, self.agentID)
        return np.concatenate([agentVel] + [agentPos] + blocksInfo + posInfo + velInfo)


class GetCollisionForce:
    def __init__(self, contactMargin = 0.001, contactForce = 100):
        self.contactMargin = contactMargin
        self.contactForce = contactForce

    def __call__(self, obj1Pos, obj2Pos, obj1Size, obj2Size, obj1Movable, obj2Movable):
        posDiff = obj1Pos - obj2Pos
        dist = np.sqrt(np.sum(np.square(posDiff)))

        minDist = obj1Size + obj2Size
        penetration = np.logaddexp(0, -(dist - minDist) / self.contactMargin) * self.contactMargin

        force = self.contactForce* posDiff / dist * penetration
        force1 = +force if obj1Movable else None
        force2 = -force if obj2Movable else None

        return [force1, force2]


class ApplyActionForce:
    def __init__(self, wolvesID, sheepsID, entitiesMovableList, actionDim=2):
        self.agentsID = sheepsID + wolvesID
        self.numAgents = len(self.agentsID)
        self.entitiesMovableList = entitiesMovableList
        self.actionDim = actionDim

    def __call__(self, pForce, actions):
        noise = [None] * self.numAgents
        for agentID in self.agentsID:
            movable = self.entitiesMovableList[agentID]
            agentNoise = noise[agentID]
            if movable:
                agentNoise = np.random.randn(self.actionDim) * agentNoise if agentNoise else 0.0
                pForce[agentID] = np.array(actions[agentID]) + agentNoise
        return pForce


class ApplyEnvironForce:
    def __init__(self, numEntities, entitiesMovableList, entitiesSizeList, getCollisionForce, getPosFromState):
        self.numEntities = numEntities
        self.entitiesMovableList = entitiesMovableList
        self.entitiesSizeList = entitiesSizeList
        self.getCollisionForce = getCollisionForce
        self.getEntityPos = lambda state, entityID: getPosFromState(state[entityID])

    def __call__(self, pForce, state):
        for entity1ID in range(self.numEntities):
            for entity2ID in range(self.numEntities):
                if entity2ID <= entity1ID: continue
                obj1Movable = self.entitiesMovableList[entity1ID]
                obj2Movable = self.entitiesMovableList[entity2ID]
                obj1Size = self.entitiesSizeList[entity1ID]
                obj2Size = self.entitiesSizeList[entity2ID]
                obj1Pos = self.getEntityPos(state, entity1ID)
                obj2Pos = self.getEntityPos(state, entity2ID)

                force1, force2 = self.getCollisionForce(obj1Pos, obj2Pos, obj1Size, obj2Size, obj1Movable, obj2Movable)

                if force1 is not None:
                    if pForce[entity1ID] is None: pForce[entity1ID] = 0.0
                    pForce[entity1ID] = force1 + pForce[entity1ID]

                if force2 is not None:
                    if pForce[entity2ID] is None: pForce[entity2ID] = 0.0
                    pForce[entity2ID] = force2 + pForce[entity2ID]

        return pForce


class IntegrateState:
    def __init__(self, numEntities, entitiesMovableList, massList, entityMaxSpeedList, getVelFromAgentState, getPosFromAgentState, damping=0.25, dt=0.1):
        self.numEntities = numEntities
        self.entitiesMovableList = entitiesMovableList
        self.damping = damping
        self.dt = dt
        self.massList = massList
        self.entityMaxSpeedList = entityMaxSpeedList
        self.getEntityVel = lambda state, entityID: getVelFromAgentState(state[entityID])
        self.getEntityPos = lambda state, entityID: getPosFromAgentState(state[entityID])

    def __call__(self, pForce, state):
        getNextState = lambda entityPos, entityVel: list(entityPos) + list(entityVel)
        nextState = []
        for entityID in range(self.numEntities):
            entityMovable = self.entitiesMovableList[entityID]
            entityVel = self.getEntityVel(state, entityID)
            entityPos = self.getEntityPos(state, entityID)

            if not entityMovable:
                nextState.append(getNextState(entityPos, entityVel))
                continue

            entityNextVel = entityVel * (1 - self.damping)
            entityForce = pForce[entityID]
            entityMass = self.massList[entityID]
            if entityForce is not None:
                entityNextVel += (entityForce / entityMass) * self.dt

            entityMaxSpeed = self.entityMaxSpeedList[entityID]
            if entityMaxSpeed is not None:
                speed = np.sqrt(np.square(entityNextVel[0]) + np.square(entityNextVel[1])) #
                if speed > entityMaxSpeed:
                    entityNextVel = entityNextVel / speed * entityMaxSpeed

            entityNextPos = entityPos + entityNextVel * self.dt
            nextState.append(getNextState(entityNextPos, entityNextVel))

        return nextState


class TransitMultiAgentChasing:
    def __init__(self, numEntities, reshapeAction, applyActionForce, applyEnvironForce, integrateState):
        self.numEntities = numEntities
        self.reshapeAction = reshapeAction
        self.applyActionForce = applyActionForce
        self.applyEnvironForce = applyEnvironForce
        self.integrateState = integrateState

    def __call__(self, state, actions):
        actions = [self.reshapeAction(action) for action in actions]
        p_force = [None] * self.numEntities
        p_force = self.applyActionForce(p_force, actions)
        p_force = self.applyEnvironForce(p_force, state)
        nextState = self.integrateState(p_force, state)

        return nextState


class ReshapeAction:
    def __init__(self):
        self.actionDim = 2
        self.sensitivity = 5

    def __call__(self, action): # action: tuple of dim (5,1)
        actionX = action[1] - action[2]
        actionY = action[3] - action[4]
        actionReshaped = np.array([actionX, actionY]) * self.sensitivity
        return actionReshaped

