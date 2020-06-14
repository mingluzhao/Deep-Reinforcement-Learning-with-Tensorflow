import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import numpy as np
import unittest
from ddt import ddt, data, unpack
from src.ddpg_followPaper import BuildCriticModel, TrainCriticBySASRQ
from RLframework.RLrun import UpdateParameters

@ddt
class TestCritic(unittest.TestCase):
    def setUp(self):
        numAgents = 2
        numStateSpace = numAgents * 2
        actionDim = 2
        
        self.buildCriticModel = BuildCriticModel(numStateSpace, actionDim)
        self.criticLayerWidths = [100, 100]

        self.tau = 0.01
        self.gamma = 0.95
        self.learningRateCritic = 0.001

        updateInterval = 1
        self.updateParameters = UpdateParameters(updateInterval, self.tau)


    @data(([[0]], [[1]]),
          ([[1], [2]], [[3], [4]])
          )
    @unpack
    def testValueTargetCalculation(self, rewardBatch, qValue):
        criticWriter, criticModel = self.buildCriticModel(self.criticLayerWidths)
        criticGraph = criticModel.graph

        yi_ = criticGraph.get_collection_ref("yi_")[0]
        gamma_ = criticGraph.get_collection_ref("gamma_")[0]
        reward_ = criticGraph.get_collection_ref("reward_")[0]
        valueTarget_ = criticGraph.get_collection_ref("valueTarget_")[0]

        yiCalculated = criticModel.run(yi_, feed_dict={gamma_: self.gamma, reward_: rewardBatch, valueTarget_: qValue})
        groundTruthYi = np.array(rewardBatch) + self.gamma* np.array(qValue)
        diff = np.concatenate(yiCalculated - groundTruthYi)
        [self.assertAlmostEqual(difference, 0, places = 5) for difference in diff]


    @data(([[1,1,1,1]], [[2, 2]], [[2]], [[2]]),
          ([[1,1,1,1], [2,2,2,2]], [[2, 2], [3, 2]], [[2], [5]], [[2], [8]])
          )
    @unpack
    def testCriticLossCalculation(self, stateBatch, actionBatch, rewardBatch, targetQValue):
        criticWriter, criticModel = self.buildCriticModel(self.criticLayerWidths)
        trainCriticBySASRQ = TrainCriticBySASRQ(self.learningRateCritic, self.gamma, criticWriter)

        criticGraph = criticModel.graph
        states_ = criticGraph.get_collection_ref("states_")[0]
        actionTarget_ = criticGraph.get_collection_ref("action_")[0]
        trainQ_ = criticGraph.get_collection_ref("trainQ_")[0]

        trainQVal = criticModel.run(trainQ_, feed_dict={states_: stateBatch, actionTarget_: actionBatch})
        calculatedLoss, criticModel = trainCriticBySASRQ(criticModel, stateBatch, actionBatch, rewardBatch, targetQValue)

        yi = np.array(rewardBatch) + self.gamma* np.array(targetQValue)
        trueLoss = np.mean(np.square(yi - trainQVal))
        self.assertAlmostEqual(trueLoss, calculatedLoss, places=3)


    @data(([[1,1,1,1]], [[2, 2]], [[2]], [[2]]),
          ([[1,1,1,1], [2,2,2,2]], [[2, 2], [3, 2]], [[2], [5]], [[2], [8]])
          )
    @unpack
    def testCriticImprovement(self, stateBatch, actionBatch, rewardBatch, targetQValue):
        criticWriter, criticModel = self.buildCriticModel(self.criticLayerWidths)
        trainCriticBySASRQ = TrainCriticBySASRQ(self.learningRateCritic, self.gamma, criticWriter)

        for i in range(10):
            lossWithTrain1, criticModel = trainCriticBySASRQ(criticModel, stateBatch, actionBatch, rewardBatch, targetQValue)
            lossWithTrain2, criticModel = trainCriticBySASRQ(criticModel, stateBatch, actionBatch, rewardBatch, targetQValue)
            print(lossWithTrain1)
            print(lossWithTrain2)
            self.assertTrue(lossWithTrain1 > lossWithTrain2)


    def testDDPGUpdateCriticParams(self):
        criticWriter, criticModel = self.buildCriticModel(self.criticLayerWidths)
        trainCriticBySASRQ = TrainCriticBySASRQ(self.learningRateCritic, self.gamma, criticWriter)
        stateBatch = [[1, 1, 1, 1]]
        actionBatch = [[2, 2]]
        rewardBatch = [[2]]
        targetQValue = [[2]]

        runTime = 20
        for i in range(runTime):
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


if __name__ == '__main__':
    unittest.main()
