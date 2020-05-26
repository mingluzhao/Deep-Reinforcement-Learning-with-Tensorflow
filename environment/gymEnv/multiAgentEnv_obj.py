import numpy as np
# predator-prey environment

class IsCollision:
    def __init__(self, getPosFromState, agent1Size, agent2Size):
        self.getPosFromState = getPosFromState
        self.agent1Size = agent1Size
        self.agent2Size = agent2Size

    def __call__(self, agent1State, agent2State):
        posDiff = self.getPosFromState(agent1State) - self.getPosFromState(agent2State)
        dist = np.sqrt(np.sum(np.square(posDiff)))
        minDist = self.agent1Size + self.agent2Size
        return True if dist < minDist else False


def getAgentState(state, agentID):
    return state[agentID]

class RewardWolf:
    def __init__(self, wolvesID, sheepsID, getAgentState, isCollision, collisionReward = 10):
        self.wolvesID = wolvesID
        self.sheepsID = sheepsID
        self.getAgentState = getAgentState
        self.isCollision = isCollision
        self.collisionReward = collisionReward

    def __call__(self, state, action, nextState):
        wolfReward = 0
        getState = lambda ID: self.getAgentState(state, ID)

        for wolfID in self.wolvesID:
            wolfState = getState(wolfID)
            for sheepID in self.sheepsID:
                sheepState = getState(sheepID)

                if self.isCollision(wolfState, sheepState):
                    wolfReward += self.collisionReward

        return wolfReward


class RewardSheep:
    def __init__(self, wolvesID, sheepID, getAgentState, isCollision, punishForOutOfBound, collisionPunishment = 10):
        self.wolvesID = wolvesID
        self.getAgentState = getAgentState
        self.sheepID = sheepID
        self.isCollision = isCollision
        self.collisionPunishment = collisionPunishment
        self.punishForOutOfBound = punishForOutOfBound

    def __call__(self, state, action, nextState):
        sheepReward = 0
        getState = lambda ID: self.getAgentState(state, ID)
        sheepState = getState(self.sheepID)
        sheepReward -= self.punishForOutOfBound(sheepState)
        for wolfID in self.wolvesID:
            wolfState = getState(wolfID)
            if self.isCollision(sheepState, wolfState):
                sheepReward -= self.collisionPunishment

        return sheepReward


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

# state = [position, velocity]

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
        blockSpeed = [0, 0]

        blocksState = [list(getBlockRandomPos()) + blockSpeed for blockID in range(self.numBlocks)]
        state = np.array(agentsState + blocksState)
        return state


class Observe:
    def __init__(self, agentID, wolvesID, sheepsID, getAgentState, getBlocksState, getPosFromState, getAgentVel):
        self.agentID = agentID
        self.wolvesID = wolvesID
        self.sheepsID = sheepsID
        self.getAgentPos = lambda state, agentID: getPosFromState(getAgentState(state, agentID))
        self.getBlocksPos = lambda state: getPosFromState(getBlocksState(state))
        self.getAgentVel = getAgentVel

    def __call__(self, state):
        blocksPos = self.getBlocksPos(state)
        agentPos = self.getAgentPos(state, self.agentID)
        agentVel = self.getAgentVel(state, self.agentID)
        blocksInfo = [blockPos - agentPos for blockPos in blocksPos]

        posInfo = []
        for wolfID in self.wolvesID:
            if wolfID == self.agentID: continue
            wolfPos = self.getAgentPos(state, wolfID)
            posInfo.append(wolfPos - agentPos)

        velInfo = []
        for sheepID in self.sheepsID:
            if sheepID == self.agentID: continue
            sheepPos = self.getAgentPos(state,sheepID)
            posInfo.append(sheepPos - agentPos)
            sheepVel = self.getAgentVel(state, sheepID)
            velInfo.append(sheepVel)

        return np.concatenate([agentVel] + [agentPos] + blocksInfo + posInfo + velInfo)
## notice the order of velocity and pos!!



class Agent(object):
    def __init__(self, isSheep):
        self.movable = True
        self.size = 0.05 if isSheep else 0.075
        self.acceleration = 4.0 if isSheep else 3.0
        self.maxSpeed = 1.3 if isSheep else 1.0
        self.color = np.array([0.35, 0.85, 0.35]) if isSheep else np.array([0.85, 0.35, 0.35])
        self.mass = 1.0
        self.positionDimension = 2

        self.reset()

    def reset(self):
        self.pos = np.random.uniform(-1, +1, self.positionDimension)
        self.vel = np.zeros(self.positionDimension)
        self.state = list(self.pos) + list(self.vel)

    def getVel(self):
        return self.pos

    def getPos(self):
        return self.vel



class Block(object):
    def __init__(self):
        self.movable = False
        self.size = 0.2
        self.acceleration = None
        self.maxSpeed = None
        self.color = np.array([0.25, 0.25, 0.25])
        self.mass = 1.0
        self.positionDimension = 2

        self.resetBlock()

    def resetBlock(self):
        self.pos = np.random.uniform(-0.9, +0.9, self.positionDimension)
        self.vel = np.zeros(self.positionDimension)
        self.state = list(self.pos) + self.vel

    def getVel(self):
        return self.pos

    def getPos(self):
        return self.vel




class TransitMultiAgentChasing:
    def __init__(self, actionDim, numAgents, numBlocks, entitiesSizeList, massList, getCollisionForce, getAgentState, getPosFromState, getBlockState):
        self.actionDim = actionDim
        self.numAgents = numAgents
        self.numBlocks = numBlocks
        self.numEntities = self.numAgents + self.numBlocks
        self.entitiesSizeList = entitiesSizeList
        self.getCollisionForce = getCollisionForce
        self.massList = massList
        self.getEntityPos = lambda state, ID: getPosFromState(getAgentState(state, ID))

        self. entitiesMovable = [True]* self.numAgents + [False] * self.numBlocks

# agents first, then block

    def __call__(self, state, actions):
        p_force = [None] * self.numEntities
        p_force = self.applyActionForce(p_force, actions)
        p_force = self.applyEnvironForce(p_force, state)

        nextState = self.integrateState(p_force, state)

        return nextState

    def applyActionForce(self, p_force, actions):
        noise = None * self.numAgents #####
        for i in range(self.numAgents):
            if self.entitiesMovable[i]:
                noise = np.random.randn(self.actionDim) * noise[i] if noise[i] else 0.0
                p_force[i] = actions[i] + noise
        return p_force

    def applyEnvironForce(self, p_force, state):
        for entity1ID in range(self.numEntities):
            for entity2ID in range(self.numEntities):
                if entity2ID <= entity1ID: continue
                obj1Movable = self.entitiesMovable[entity1ID]
                obj2Movable = self.entitiesMovable[entity2ID]

                obj1Size = self.entitiesSizeList[entity1ID]
                obj2Size = self.entitiesSizeList[entity2ID]

                obj1Pos = self.getEntityPos(state, entity1ID)
                obj2Pos = self.getEntityPos(state, entity2ID)

                force1, force2 = self.getCollisionForce(obj1Pos, obj2Pos, obj1Size, obj2Size, obj1Movable, obj2Movable)

                if force1 is not None:
                    if p_force[entity1ID] is None: p_force[entity1ID] = 0.0
                    p_force[entity1ID] = force1 + p_force[entity1ID]

                if force2 is not None:
                    if p_force[entity2ID] is None: p_force[entity2ID] = 0.0
                    p_force[entity2ID] = force2 + p_force[entity2ID]

        return p_force


    def integrateState(self, p_force, state, damping = 0.25, dt = 0.1):
        nextState = []
        for entityID in range(self.numEntities):
            entityMovable = self.entitiesMovable[entityID]
            if not entityMovable: continue
            entityVel = self.getAgentVel(state, entityID)
            entityNextVel = entityVel * (1- damping)

            entityForce = p_force[entityID]
            entityMass = self.massList[entityID]
            if entityForce is not None:
                entityNextVel += (entityForce / entityMass) * dt

            if entityMaxSpeed is not None:
                speed = np.sqrt(np.square(entityVel[0]) + np.square(entityVel[1]))
                if speed > entityMaxSpeed:
                    entityNextVel = entityNextVel / speed * entityMaxSpeed

            entityNextPos = entityPos + entityNextVel * dt
            nextState.append([entityNextPos, entityNextVel])



def getCollisionForce(obj1Pos, obj2Pos, obj1Size, obj2Size, obj1Movable, obj2Movable, contactMargin = 0.001, contactForce = 100):
    posDiff = obj1Pos - obj2Pos
    dist = np.sqrt(np.sum(np.square(posDiff)))

    minDist = obj1Size + obj2Size
    penetration = np.logaddexp(0, -(dist - minDist) / contactMargin) * contactMargin

    force = contactForce* posDiff / dist * penetration
    force1 = +force if obj1Movable else None
    force2 = -force if obj2Movable else None

    return [force1, force2]


