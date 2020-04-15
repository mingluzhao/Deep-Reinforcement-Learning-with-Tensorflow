import sys
sys.path.append("..")
import unittest
from ddt import ddt, data, unpack
from src.ddpg_withModifiedCritic import *
from src.policy import *

@ddt
class TestCritic(unittest.TestCase):
    def setUp(self):
        numAgents = 2
        numStateSpace = numAgents * 2
        actionDim = 1
        
        self.buildCriticModel = BuildCriticModel(numStateSpace, actionDim)
        self.criticTrainingLayerWidths = [10, 10]
        self.criticTargetLayerWidths = self.criticTrainingLayerWidths

        self.tau = 0.01
        self.gamma = 0.95
        self.learningRateCritic = 0.001

        self.updateParameters = UpdateParameters(self.tau)


    @data(([[0]], [[1]]),
          ([[1], [2]], [[3], [4]])
          )
    @unpack
    def testValueTargetCalculation(self, rewardBatch, qValue):
        criticWriter, criticModel = self.buildCriticModel(self.criticTrainingLayerWidths, self.criticTargetLayerWidths)
        criticGraph = criticModel.graph

        yi_ = criticGraph.get_collection_ref("yi_")[0]
        gamma_ = criticGraph.get_collection_ref("gamma_")[0]
        reward_ = criticGraph.get_collection_ref("reward_")[0]
        valueTarget_ = criticGraph.get_collection_ref("valueTarget_")[0]

        yiCalculated = criticModel.run(yi_, feed_dict={gamma_: self.gamma, reward_: rewardBatch, valueTarget_: qValue})
        groundTruthYi = np.array(rewardBatch) + self.gamma* np.array(qValue)
        diff = np.concatenate(yiCalculated - groundTruthYi)
        [self.assertAlmostEqual(difference, 0, places = 5) for difference in diff]


    @data(([[1,1,1,1]], [[2]], [[2]], [[2]]),
          ([[1,1,1,1], [2,2,2,2]], [[2], [3]], [[2], [5]], [[2], [8]])
          )
    @unpack
    def testCriticLossCalculation(self, stateBatch, actionBatch, rewardBatch, targetQValue):
        criticWriter, criticModel = self.buildCriticModel(self.criticTrainingLayerWidths, self.criticTargetLayerWidths)
        trainCriticBySASRQ = TrainCriticBySASRQ(self.learningRateCritic, self.gamma, criticWriter)

        criticGraph = criticModel.graph
        states_ = criticGraph.get_collection_ref("states_")[0]
        actionTarget_ = criticGraph.get_collection_ref("action_")[0]
        trainQ_ = criticGraph.get_collection_ref("trainQ_")[0]

        trainQVal = criticModel.run(trainQ_, feed_dict={states_: stateBatch, actionTarget_: actionBatch})
        calculatedLoss, criticModel = trainCriticBySASRQ(criticModel, stateBatch, actionBatch, rewardBatch, targetQValue)

        yi = np.array(rewardBatch) + self.gamma* np.array(targetQValue)
        trueLoss = np.mean(np.square(yi - trainQVal))
        self.assertAlmostEqual(trueLoss, calculatedLoss, places=5)


    @data(([[1,1,1,1]], [[2]], [[2]], [[2]]),
          ([[1,1,1,1], [2,2,2,2]], [[2], [3]], [[2], [5]], [[2], [8]])
          )
    @unpack
    def testCriticImprovement(self, stateBatch, actionBatch, rewardBatch, targetQValue):
        criticWriter, criticModel = self.buildCriticModel(self.criticTrainingLayerWidths, self.criticTargetLayerWidths)
        trainCriticBySASRQ = TrainCriticBySASRQ(self.learningRateCritic, self.gamma, criticWriter)
        lossWithTrain1, criticModel = trainCriticBySASRQ(criticModel, stateBatch, actionBatch, rewardBatch, targetQValue)
        lossWithTrain2, criticModel = trainCriticBySASRQ(criticModel, stateBatch, actionBatch, rewardBatch, targetQValue)

        self.assertTrue(lossWithTrain1 > lossWithTrain2)


    def testUpdateCriticParams(self):
        criticWriter, criticModel = self.buildCriticModel(self.criticTrainingLayerWidths, self.criticTargetLayerWidths)
        trainCriticBySASRQ = TrainCriticBySASRQ(self.learningRateCritic, self.gamma, criticWriter)
        stateBatch = [[1, 1, 1, 1]]
        actionBatch = [[2]]
        rewardBatch = [[2]]
        targetQValue = [[2]]
        for i in range(20):
            calculatedLoss, criticModel = trainCriticBySASRQ(criticModel, stateBatch, actionBatch, rewardBatch, targetQValue)

        criticGraph = criticModel.graph
        trainParams_ = criticGraph.get_collection_ref("trainParams_")[0]
        targetParams_ = criticGraph.get_collection_ref("targetParams_")[0]
        trainParams, targetParams = criticModel.run([trainParams_,targetParams_])

        updatedCriticModel = self.updateParameters(criticModel)

        updatedCriticGraph = updatedCriticModel.graph
        updatedTrainParams_ = updatedCriticGraph.get_collection_ref("trainParams_")[0]
        updatedTargetParams_ = updatedCriticGraph.get_collection_ref("targetParams_")[0]
        updatedTrainParams, updatedTargetParams = criticModel.run([updatedTrainParams_, updatedTargetParams_])

        calUpdatedTargetParam = (1- self.tau)* np.array(targetParams) + self.tau* np.array(updatedTrainParams)
        difference = np.array(updatedTargetParams) - calUpdatedTargetParam

        [self.assertEqual(np.mean(paramDiff), 0) for paramDiff in difference]


    @data(([[1,1,1,1]], [[2]]),
          ([[1,1,1,1], [2,2,2,2]], [[2], [3]])
          )
    @unpack
    def testActionGradients(self, stateBatch, actionBatch):
        criticWriter, criticModel = self.buildCriticModel(self.criticTrainingLayerWidths, self.criticTargetLayerWidths)
        criticGraph = criticModel.graph

        actionGradients_ = criticGraph.get_collection_ref("actionGradients_")[0]
        states_ = criticGraph.get_collection_ref("states_")[0]
        action_ = criticGraph.get_collection_ref("action_")[0]

        trainActionWeights_ = criticGraph.get_collection_ref("weight/trainActionFCToLastFCWeights_")[0]
        finalFCWeights_ = criticGraph.get_collection_ref('weight/trainHidden/dense/kernel:0')[0]

        actionGradients, trainActionWeights, finalFCWeights = \
            criticModel.run([actionGradients_, trainActionWeights_, finalFCWeights_],
                                                        feed_dict= {states_: stateBatch, action_: actionBatch})

        reluGradients = trainActionWeights
        reluGradients[reluGradients < 0 ] = 0  # relu gradient  = 1 for value > 0, 0 for <0

        trueActionGradients = np.sum(np.matmul(reluGradients, finalFCWeights))
        difference = np.array(trueActionGradients) - actionGradients
        [self.assertEqual(diff, 0) for diff in difference]




if __name__ == '__main__':
    unittest.main()
