import sys
sys.path.append("..")
import unittest
from ddt import ddt, data, unpack
from src.ddpg_withModifiedCritic import *
from src.policy import *

import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'



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
        self.actorTrainingLayerWidths = [20, 20]
        self.actorTargetLayerWidths = self.actorTrainingLayerWidths

        self.tau = 0.01
        self.gamma = 0.95
        self.learningRateActor = 0.0001

        self.updateParameters = UpdateParameters(self.tau)

        self.buildCriticModel = BuildCriticModel(numStateSpace, actionDim)
        self.criticTrainingLayerWidths = [10, 10]
        self.criticTargetLayerWidths = self.criticTrainingLayerWidths
        self.learningRateCritic = 0.001


    @data(([[10, 5, 2, 5], [10, 5, 2, 5]], [[0], [-np.pi]], [[0], [0]])
          )
    @unpack
    def testActor(self, stateBatch, actionBatch, rewardBatch):
        criticWriter, criticModel = self.buildCriticModel(self.criticTrainingLayerWidths,
                                                          self.criticTargetLayerWidths)
        trainCriticBySASRQ = TrainCriticBySASRQ(self.learningRateCritic, self.gamma, criticWriter)

        getPseudoTargetQ = lambda action: np.array(action)* np.array(action) - 1 # best is +-pi, 0 is worst
        lossList = []
        for i in range(1000):
            targetQValue = getPseudoTargetQ(actionBatch)
            loss, criticModel = trainCriticBySASRQ(criticModel, stateBatch, actionBatch, rewardBatch, targetQValue)
            lossList.append(loss)

        # plt.plot(lossList)
        # plt.show()
        actorWriter, actorModel = self.buildActorModel(self.actorTrainingLayerWidths, self.actorTargetLayerWidths)
        untrainedAct = actByPolicyTrain(actorModel, stateBatch)
        untrainedValue = evaluateCriticTrain(criticModel, stateBatch, untrainedAct)

        trainActorFromGradients = TrainActorFromGradients(self.learningRateActor, actorWriter)
        trainActorOneStep = TrainActorOneStep(actByPolicyTrain, trainActorFromGradients, getActionGradients)

        actorModel = trainActorOneStep(actorModel, criticModel, stateBatch)

        train1StepAct = actByPolicyTrain(actorModel, stateBatch)
        trainedValue = evaluateCriticTrain(criticModel, stateBatch, train1StepAct)

        [self.assertTrue(trained > untrained) for trained, untrained in zip(trainedValue, untrainedValue)]


    def testUpdateActorParams(self):
        stateBatch = [[2, 5, 10, 5]] #-pi is optimal,
        actorWriter, actorModel = self.buildActorModel(self.actorTrainingLayerWidths, self.actorTargetLayerWidths)
        trainActorFromGradients = TrainActorFromGradients(self.learningRateActor, actorWriter)
        actionGradients = [[2]]

        for i in range(20):
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
