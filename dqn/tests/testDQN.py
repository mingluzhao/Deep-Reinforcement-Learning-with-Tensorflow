import sys
sys.path.append("..")
import unittest
from ddt import ddt, data, unpack
from dqn.src.dqn import *
from RLframework.RLrun import *

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'



@ddt
class TestDQN(unittest.TestCase):
    def setUp(self):
        stateDim = 3
        actionDim = 25
        self.buildModel = BuildModel(stateDim, actionDim)
        self.layersWidths = [30, 30]
        writer, model = self.buildModel(self.layersWidths)

        learningRate = 0.001
        gamma = 0.9
        self.trainModelBySASRQ = TrainModelBySASRQ(learningRate, gamma, writer)

        paramUpdateInterval = 100
        self.updateParameters = UpdateParameters(paramUpdateInterval)


    @data(([[1,1,1]], [2], [2], [[1,5,1]]),
          ([[1,1,1], [2,2,2]], [2, 3], [2, 5], [[1,2,1], [2,3,3]])
          )
    @unpack
    def testUpdateParamsNoUpdate(self, stateBatch, actionBatch, rewardBatch, nextStateBatch):
        writer, model = self.buildModel(self.layersWidths)
        model = resetTargetParamToTrainParam([model])[0]
        targetNextStateQ = getTargetQValue(model, nextStateBatch)

        graph = model.graph
        trainParams_ = graph.get_collection_ref("trainParams_")[0]
        targetParams_ = graph.get_collection_ref("targetParams_")[0]
        trainParams, targetParams = model.run([trainParams_, targetParams_])

        runTime = 50
        for i in range(runTime):
            loss, model = self.trainModelBySASRQ(model, stateBatch, actionBatch, rewardBatch, targetNextStateQ)
            model = self.updateParameters(model, runTime)

        updatedModel = model
        updatedGraph = updatedModel.graph
        updatedTrainParams_ = updatedGraph.get_collection_ref("trainParams_")[0]
        updatedTargetParams_ = updatedGraph.get_collection_ref("targetParams_")[0]
        updatedTrainParams, updatedTargetParams = updatedModel.run([updatedTrainParams_, updatedTargetParams_])


        difference = np.array(targetParams) - trainParams
        [self.assertEqual(np.mean(paramDiff), 0) for paramDiff in difference]

        difference = np.array(updatedTargetParams) - targetParams
        [self.assertEqual(np.mean(paramDiff), 0) for paramDiff in difference]


    @data(([[1,3,1]], [2], [2], [[1,5,1]]),
          ([[1,10,1], [2,2,2]], [2, 3], [2, 5], [[1,2,1], [2,3,3]])
          )
    @unpack
    def testUpdateParamsUpdate(self, stateBatch, actionBatch, rewardBatch, nextStateBatch):
        writer, model = self.buildModel(self.layersWidths)
        model = resetTargetParamToTrainParam([model])[0]
        targetNextStateQ = getTargetQValue(model, nextStateBatch)

        runTime = 100
        for i in range(runTime):
            loss, model = self.trainModelBySASRQ(model, stateBatch, actionBatch, rewardBatch, targetNextStateQ)
            model = self.updateParameters(model, runTime)

        updatedModel = model
        updatedGraph = updatedModel.graph
        updatedTrainParams_ = updatedGraph.get_collection_ref("trainParams_")[0]
        updatedTargetParams_ = updatedGraph.get_collection_ref("targetParams_")[0]
        updatedTrainParams, updatedTargetParams = updatedModel.run([updatedTrainParams_, updatedTargetParams_])

        difference = np.array(updatedTargetParams) - updatedTrainParams
        [self.assertEqual(np.mean(paramDiff), 0) for paramDiff in difference]


    @data(([[1,1,1]], [2], [2], [[1,5,1]]),
          ([[1,1,1], [2,2,2]], [2, 3], [2, 5], [[1,2,1], [2,3,3]])
          )
    @unpack
    def testDQNTrainImprovement(self, stateBatch, actionBatch, rewardBatch, nextStateBatch):
        writer, model = self.buildModel(self.layersWidths)
        model = resetTargetParamToTrainParam([model])[0]
        targetNextStateQ = getTargetQValue(model, nextStateBatch)

        lossBeforeUpdate, updatedModel = self.trainModelBySASRQ(model, stateBatch, actionBatch, rewardBatch, targetNextStateQ)
        lossAfter1Update, updatedModel = self.trainModelBySASRQ(updatedModel, stateBatch, actionBatch, rewardBatch, targetNextStateQ)

        self.assertTrue(lossAfter1Update < lossBeforeUpdate)

if __name__ == '__main__':
    unittest.main()
