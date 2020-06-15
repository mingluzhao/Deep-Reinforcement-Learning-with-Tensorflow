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
sys.path.append("..")
import unittest
from ddt import ddt, data, unpack
from collections import deque

from dqn.src.dqn import BuildModel, TrainModelBySASRQ, getTargetQValue
from ddpg.src.ddpg import BuildActorModel, BuildCriticModel, TrainActorFromGradients, TrainCriticBySASRQ
from RLframework.RLrun import UpdateParameters, resetTargetParamToTrainParam, SampleFromMemory, getBuffer
import random

@ddt
class TestDQNParameterUpdate(unittest.TestCase):
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

    @data(([[1, 1, 1]], [2], [2], [[1, 5, 1]]),
          ([[1, 1, 1], [2, 2, 2]], [2, 3], [2, 5], [[1, 2, 1], [2, 3, 3]])
          )
    @unpack
    def testDQNUpdateParamsNoUpdate(self, stateBatch, actionBatch, rewardBatch, nextStateBatch):
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
            model = self.updateParameters(model)

        updatedModel = model
        updatedGraph = updatedModel.graph
        updatedTrainParams_ = updatedGraph.get_collection_ref("trainParams_")[0]
        updatedTargetParams_ = updatedGraph.get_collection_ref("targetParams_")[0]
        updatedTrainParams, updatedTargetParams = updatedModel.run([updatedTrainParams_, updatedTargetParams_])

        difference = np.array(targetParams) - trainParams
        [self.assertEqual(np.mean(paramDiff), 0) for paramDiff in difference]

        difference = np.array(updatedTargetParams) - targetParams
        [self.assertEqual(np.mean(paramDiff), 0) for paramDiff in difference]

    @data(([[1, 3, 1]], [2], [2], [[1, 5, 1]]),
          ([[1, 10, 1], [2, 2, 2]], [2, 3], [2, 5], [[1, 2, 1], [2, 3, 3]])
          )
    @unpack
    def testDQNUpdateParamsUpdate(self, stateBatch, actionBatch, rewardBatch, nextStateBatch):
        writer, model = self.buildModel(self.layersWidths)
        model = resetTargetParamToTrainParam([model])[0]
        targetNextStateQ = getTargetQValue(model, nextStateBatch)

        runTime = 101
        for i in range(runTime):
            loss, model = self.trainModelBySASRQ(model, stateBatch, actionBatch, rewardBatch, targetNextStateQ)
            model = self.updateParameters(model)

        updatedModel = model
        updatedGraph = updatedModel.graph
        updatedTrainParams_ = updatedGraph.get_collection_ref("trainParams_")[0]
        updatedTargetParams_ = updatedGraph.get_collection_ref("targetParams_")[0]
        updatedTrainParams, updatedTargetParams = updatedModel.run([updatedTrainParams_, updatedTargetParams_])

        difference = np.array(updatedTargetParams) - updatedTrainParams
        [self.assertEqual(np.mean(paramDiff), 0) for paramDiff in difference]




@ddt
class TestDDPGParameterUpdate(unittest.TestCase):
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

    def testDDPGUpdateCriticParams(self):
        criticWriter, criticModel = self.buildCriticModel(self.criticLayerWidths)
        trainCriticBySASRQ = TrainCriticBySASRQ(self.learningRateCritic, self.gamma, criticWriter)
        stateBatch = [[1, 1, 1, 1]]
        actionBatch = [[2]]
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


@ddt
class TestReplayBuffer(unittest.TestCase):
    def setUp(self):
        minibatchSize  = 32
        self.sampleFromBuffer = SampleFromMemory(minibatchSize)

    def testBufferSize(self):
        bufferSize = 5
        buffer = getBuffer(bufferSize)
        for i in range(5):
            buffer.append((1, 1))
            self.assertEqual(len(buffer), i+1)

        buffer.append((1, 2))
        self.assertEqual(len(buffer), 5)

    def testBufferSampleSize(self):
        bufferSize = 5
        buffer = getBuffer(bufferSize)
        for i in range(5):
            buffer.append((1, i))

        minibatchSize = 2
        sampleBuffer = SampleFromMemory(minibatchSize)

        sample = sampleBuffer(buffer)
        self.assertEqual(len(sample), minibatchSize)

    def testMultiAgentBufferRetrieval(self):
        getAgentBuffer = lambda buffer, id: [[bufferElement[id] for bufferElement in timeStepBuffer] for timeStepBuffer
                                             in buffer]

        buffer = getBuffer(5)
        buffer.append((((1, 1), (2, 2)), (2, 3), (4, 5)))
        buffer.append((((1, 1), (2, 2)), (2, 3), (4, 5)))
        buffer.append((((1, 1), (2, 2)), (2, 3), (4, 5)))

        agentBuffer = getAgentBuffer(buffer, 1)
        trueAgentBuffer = ([(2, 2), (3), (5)], [(2, 2), (3), (5)], [(2, 2), (3), (5)])

        self.assertEqual(tuple(agentBuffer), trueAgentBuffer)

if __name__ == '__main__':
    unittest.main()
