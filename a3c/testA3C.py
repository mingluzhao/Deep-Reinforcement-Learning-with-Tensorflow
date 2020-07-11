import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))
sys.path.append(os.path.join(dirName, '..'))

import numpy as np
import unittest
from ddt import ddt, data, unpack

from a3c.a3cWithGlobalSession import *

@ddt
class TestActor(unittest.TestCase):
    def setUp(self):



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


    def testActorTrainImprovement(self):
        stateBatch = [[2, 5, 10, 5, 2, 5, 10, 5], [1,1,1,1,1,1,1,1]]
        actionBatch = [[0.1,0.2, 0.3, 0.4, 0.5], [0.2,0.2,0.2,0.2,0.2]]
        rewardBatch = [[2], [0]]
        targetQValue = [[3], [1]]

        numStateSpace = len(stateBatch[0])
        actionDim = 5
        actionRange = 1

        buildActorModel = BuildActorModel(numStateSpace, actionDim, actionRange)
        actorLayerWidths = [64, 64]
        criticLayerWidths = [64, 64]
        buildCriticModel = BuildCriticModel(numStateSpace, actionDim)

        actorWriter, actorModel = buildActorModel(actorLayerWidths)
        criticWriter, criticModel = buildCriticModel(criticLayerWidths)

        trainCriticBySASRQ = TrainCriticBySASRQ(self.learningRateCritic, self.gamma, criticWriter)

        for i in range(100):
            lossWithTrain, criticModel = trainCriticBySASRQ(criticModel, stateBatch, actionBatch, rewardBatch,
                                                             targetQValue)
            print(lossWithTrain)

        actionUntrained = actByPolicyTrain(actorModel, stateBatch)
        actionUntrainedQVal = evaluateCriticTrain(criticModel, stateBatch, actionUntrained)

        trainActorFromGradients = TrainActorFromGradients(self.learningRateActor, actorWriter)
        trainOneStep = TrainActorOneStep(actByPolicyTrain, trainActorFromGradients, getActionGradients)

        actorModel = trainOneStep(actorModel, criticModel, stateBatch)
        actionTrained = actByPolicyTrain(actorModel, stateBatch)
        actionTrainedValue = evaluateCriticTrain(criticModel, stateBatch, actionTrained)

        [self.assertTrue(trained > untrained) for trained, untrained in zip(actionTrainedValue, actionUntrainedQVal)]


if __name__ == '__main__':
    unittest.main()
