import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..', '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
sys.path.append(os.path.join(dirName, '..'))

import numpy as np
import unittest
from ddt import ddt, data, unpack
from environment.chasingEnv.envNoPhysics import Reset, getIntendedNextState, StayWithinBoundary, \
    TransitForNoPhysics, GetAgentPosFromState, IsBoundaryTerminal, IsTerminal

@ddt
class TestEnv(unittest.TestCase):
    def setUp(self):
        self.xBoundary = (0, 20)
        self.yBoundary = (0, 20)
        self.stayWithinBoundary = StayWithinBoundary(self.xBoundary, self.yBoundary)

    @data(((0, 5), (0, 5), 2),
          ((0, 5), (10, 20), 1))
    @unpack
    def testReset(self, xBoundary, yBoundary, numOfAgent):
        reset = Reset(xBoundary, yBoundary, numOfAgent)
        state = reset()
        stateArray = np.array(state)
        self.assertEqual(stateArray.shape, (numOfAgent*2, ))

    @data(((0, 5), (0, 5), (1, 6), (1, 5)),
          ((0, 5), (0, 5), (-2, 2), (0, 2)),
          ((0, 6), (0, 6), (-1, 7), (0, 6)))
    @unpack
    def testBoundaryStaying(self, xBoundary, yBoundary, intendedNextState, trueNextState):
        stayWithinBoundary = StayWithinBoundary(xBoundary, yBoundary)
        nextState = stayWithinBoundary(intendedNextState)
        self.assertEqual(nextState, trueNextState)

    @data(((0, 0, 0, 0), (1, 1, 1, 1),  [1, 1, 1, 1]),
          ((0, 0, 0, 0), (19, 20, 21, 22), [19, 20, 20, 20]),
          ((1, 2, 3, 4), (-5, -2, -1, 0), [0, 0, 2, 4])
          )
    @unpack
    def testTransition(self, state, action, trueNextState):
        transitForNoPhysics = TransitForNoPhysics(getIntendedNextState, self.stayWithinBoundary)
        nextState = tuple(transitForNoPhysics(state, action))
        trueNextState = tuple(trueNextState)
        self.assertEqual(nextState, trueNextState)

    @data((1, [1,2,3,4], [3,4]),
          (0, [1,2], [1,2])
    )
    @unpack
    def testGetAgentPos(self, agentID, state, truePos):
        getAgentPosFromState = GetAgentPosFromState(agentID)
        pos = getAgentPosFromState(state)
        self.assertEqual(pos, truePos)

    @data(((1, 2, 3, 4), False),
          ((0, 2, 3, 4), True),
          ((1, 20, 0, 0), True),
          )
    @unpack
    def testBoundaryTerminal(self, state, trueTerminal):
        sheepID = 0
        getSheepPos = GetAgentPosFromState(sheepID)
        isBoundaryTerminal = IsBoundaryTerminal(self.xBoundary, self.yBoundary, getSheepPos)
        terminal = isBoundaryTerminal(state)
        self.assertEqual(terminal, trueTerminal)

    @data(((1, 2, 3, 4), False),
          ((1, 2, 1, 2), True),
          ((0, 2, 3, 4), True),
          ((1, 20, 0, 0), True),
          )
    @unpack
    def testTerminal(self, state, trueTerminal):
        sheepId = 0
        wolfId = 1
        getSheepPos = GetAgentPosFromState(sheepId)
        getWolfPos = GetAgentPosFromState(wolfId)

        isBoundaryTerminal = IsBoundaryTerminal(self.xBoundary, self.yBoundary, getSheepPos)
        killzoneRadius = 1
        isTerminal = IsTerminal(getWolfPos, getSheepPos, killzoneRadius, isBoundaryTerminal)

        terminal = isTerminal(state)
        self.assertEqual(terminal, trueTerminal)



if __name__ == '__main__':
    unittest.main()
