import sys
sys.path.append("..")
import unittest
from ddt import ddt, data, unpack
import numpy as np
from environment.chasingEnv.envNoPhysics import GetAgentPosFromState
from environment.chasingEnv.chasingPolicy import HeatSeekingContinuousDeterministicPolicy, ActByAngle

@ddt
class TestPolicy(unittest.TestCase):
    def setUp(self):
        self.sheepId = 0
        self.wolfId = 1
        self.getSheepXPos = GetAgentPosFromState(self.sheepId)
        self.getWolfXPos = GetAgentPosFromState(self.wolfId)
        self.actionMagnitude = 1
        self.wolfPolicy = HeatSeekingContinuousDeterministicPolicy(self.getWolfXPos, self.getSheepXPos, self.actionMagnitude)


    @data((np.asarray([-3, 0, -5, 0]), 10, np.asarray((10, 0))),
          (np.asarray([-3, 0, 0, 4]), 5, np.asarray((-3, -4))),
          (np.asarray([0, 0, 1, 0]), 1, np.asarray((-1, 0)))
          )
    @unpack
    def testHeatSeekingContinuesDeterministicPolicy(self, state, actionMagnitude, groundTruthWolfAction):
        wolfPolicy = HeatSeekingContinuousDeterministicPolicy(self.getWolfXPos, self.getSheepXPos, actionMagnitude)
        action = wolfPolicy(state)
        truthValue = np.allclose(action, groundTruthWolfAction)
        self.assertTrue(truthValue)


    @data(
        (np.sqrt(2), np.pi/4, (1, 1)),
        (1, np.pi/3, (1/2, np.sqrt(3)/2)),
        (10, 0, (10, 0)),
        (10, -np.pi, (-10, 0)),
         (5, -np.pi/2, (0, -5))
    )
    @unpack
    def testActByAngle(self, actionMagnitude, angle, groundTruthAction):
        actByAngle = ActByAngle(actionMagnitude)
        action = actByAngle(angle)
        truthValue = np.allclose(action, groundTruthAction)
        self.assertTrue(truthValue)

if __name__ == '__main__':
    unittest.main()
