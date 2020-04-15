import numpy as np

class Reset():
    def __init__(self, xBoundary, yBoundary, numOfAgent):
        self.xBoundary = xBoundary
        self.yBoundary = yBoundary
        self.numOfAgent = numOfAgent

    def __call__(self):
        xMin, xMax = self.xBoundary
        yMin, yMax = self.yBoundary
        initState = []
        for agentID in range(self.numOfAgent):
            initState.append(np.random.uniform(xMin, xMax))
            initState.append(np.random.uniform(yMin, yMax))

        return np.array(initState)

#
# def samplePosition(xBoundary, yBoundary):
#     positionX = np.random.uniform(xBoundary[0], xBoundary[1])
#     positionY = np.random.uniform(yBoundary[0], yBoundary[1])
#     position = [positionX, positionY]
#     return position


class TransitForNoPhysics():
    def __init__(self, stayWithinBoundary):
        self.stayWithinBoundary = stayWithinBoundary

    def __call__(self, state, action):
        numAgents = int(len(state)/2)
        newState = np.array(state) + np.array(action)
        agentsPosition = [[newState[2 * id], newState[2 * id + 1]] for id in range(numAgents)]
        checkedNextState = [self.stayWithinBoundary(position) for position in agentsPosition]
        nextState = np.concatenate(checkedNextState)
        return nextState

class StayWithinBoundary:
    def __init__(self, xBoundary, yBoundary):
        self.xMin, self.xMax = xBoundary
        self.yMin, self.yMax = yBoundary

    def __call__(self, intendedCoord):
        nextX, nextY = intendedCoord
        if nextX < self.xMin:
            nextX = self.xMin
        if nextX > self.xMax:
            nextX = self.xMax
        if nextY < self.yMin:
            nextY = self.yMin
        if nextY > self.yMax:
            nextY = self.yMax
        return nextX, nextY


class TransitWithSingleWolf:
    def __init__(self, transit, wolfPolicy):
        self.transit = transit
        self.wolfPolicy = wolfPolicy

    def __call__(self, state, sheepAction):
        wolfAction = list(self.wolfPolicy(state))
        allAgentsActions = wolfAction + list(sheepAction)
        nextState = self.transit(state, allAgentsActions)
        return nextState


class GetAgentPosFromState:
    def __init__(self, agentId):
        self.agentId = agentId

    def __call__(self, state):
        state = np.asarray(state)
        agentPos = [state[self.agentId*2], state[self.agentId*2+1]]
        return agentPos


class IsTerminal():
    def __init__(self, getPredatorPos, getPreyPos, minDistance):
        self.getPredatorPos = getPredatorPos
        self.getPreyPos = getPreyPos
        self.minDistance = minDistance

    def __call__(self, state):
        terminal = False
        preyPosition = self.getPreyPos(state)
        predatorPosition = self.getPredatorPos(state)
        L2Normdistance = np.linalg.norm((np.array(preyPosition) - np.array(predatorPosition)), ord=2)
        if L2Normdistance <= self.minDistance:
            terminal = True
        return terminal
