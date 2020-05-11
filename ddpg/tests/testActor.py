import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))
sys.path.append(os.path.join(dirName, '..'))


import numpy as np
import unittest
from ddt import ddt, data, unpack

from src.ddpg import BuildActorModel, BuildCriticModel,TrainActorFromGradients
from RLframework.RLrun import UpdateParameters

@ddt
class TestActor(unittest.TestCase):
    def setUp(self):
        numAgents = 2
        numStateSpace = numAgents * 2
        actionLow = -np.pi
        actionHigh = np.pi
        actionRange = (actionHigh - actionLow) / 2.0
        actionDim = 1

        self.buildActorModel = BuildActorModel(numStateSpace, actionDim, actionRange)
        self.actorLayerWidths = [20, 20]

        self.tau = 0.01
        self.gamma = 0.95
        self.learningRateActor = 0.0001

        paramUpdateInterval = 1
        self.updateParameters = UpdateParameters(paramUpdateInterval, self.tau)

        self.buildCriticModel = BuildCriticModel(numStateSpace, actionDim)
        self.criticLayerWidths = [10, 10]
        self.learningRateCritic = 0.001


    def testDDPGUpdateActorParams(self):
        stateBatch = [[2, 5, 10, 5]]
        actorWriter, actorModel = self.buildActorModel(self.actorLayerWidths)
        trainActorFromGradients = TrainActorFromGradients(self.learningRateActor, actorWriter)
        actionGradients = [[2]]

        runTime = 20
        for i in range(runTime):
            trainActorFromGradients(actorModel, stateBatch, actionGradients)

        actorGraph = actorModel.graph
        trainParams_ = actorGraph.get_collection_ref("trainParams_")[0]
        targetParams_ = actorGraph.get_collection_ref("targetParams_")[0]
        trainParams, targetParams = actorModel.run([trainParams_, targetParams_])

        updatedActorModel = self.updateParameters(actorModel)

        updatedActorGraph = updatedActorModel.graph
        updatedTrainParams_ = updatedActorGraph.get_collection_ref("trainParams_")[0]
        updatedTargetParams_ = updatedActorGraph.get_collection_ref("targetParams_")[0]
        updatedTrainParams, updatedTargetParams = actorModel.run([updatedTrainParams_, updatedTargetParams_])

        calUpdatedTargetParam = (1 - self.tau) * np.array(targetParams) + self.tau * np.array(updatedTrainParams)
        difference = np.array(updatedTargetParams) - calUpdatedTargetParam

        [self.assertEqual(np.mean(paramDiff), 0) for paramDiff in difference]


if __name__ == '__main__':
    unittest.main()
