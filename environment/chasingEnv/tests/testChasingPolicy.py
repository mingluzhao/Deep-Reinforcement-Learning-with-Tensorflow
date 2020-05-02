import sys
sys.path.append("..")
import unittest
from ddt import ddt, data, unpack
from environment.chasingEnv.envNoPhysics import GetAgentPosFromState
from environment.chasingEnv.chasingPolicy import *

@ddt
class TestPolicy(unittest.TestCase):
    def setUp(self):
        self.sheepId = 0
        self.wolfId = 1
        self.getSheepXPos = GetAgentPosFromState(self.sheepId)
        self.getWolfXPos = GetAgentPosFromState(self.wolfId)
        self.actionMagnitude = 1
        self.wolfPolicy = HeatSeekingContinuousDeterministicPolicy(self.getWolfXPos, self.getSheepXPos, self.actionMagnitude)

        actionNoise = 0.1
        noiseDecay = 0.999
        self.actionLow = -np.pi
        self.actionHigh = np.pi
        self.addActionNoise = AddActionNoise(actionNoise, noiseDecay, self.actionLow, self.actionHigh)

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
        ([[1]], 0),
        ([[3]], 2)
    )
    @unpack
    def testAddActionNoise(self, actionPerfect, timeStep):
        actionWithNoise = self.addActionNoise(actionPerfect, timeStep)
        self.assertTrue(self.actionLow <= actionWithNoise <= self.actionHigh)


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
