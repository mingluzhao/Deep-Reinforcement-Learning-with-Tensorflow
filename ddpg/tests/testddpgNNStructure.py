import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))
sys.path.append(os.path.join(dirName, '..'))

import unittest
from ddt import ddt, data, unpack
from src.ddpg import *


@ddt
class TestBuildActorModel(unittest.TestCase):
    def setUp(self):
        self.numStateSpace = 4
        self.actionDim = 1
        self.actionRatio = np.pi
        self.buildActorModel = BuildActorModel(self.numStateSpace, self.actionDim, self.actionRatio)
        self.buildCriticModel = BuildCriticModel(self.numStateSpace, self.actionDim)

    def testBuildActorModelWithEmptyInput(self):
        actorWriter, model  = self.buildActorModel([])
        groundTruthShapes = [[4, 1], [4, 1]]

        actorGraph = model.graph
        weights_ = actorGraph.get_collection("weights")
        generatedWeightShapes = [w_.shape.as_list() for w_ in weights_]
        self.assertEqual(generatedWeightShapes, groundTruthShapes)

        biasShapes = [[shape[1]] for shape in groundTruthShapes]
        biases_ = actorGraph.get_collection("biases")
        generatedBiasShapes = [b_.shape.as_list() for b_ in biases_]
        self.assertEqual(generatedBiasShapes, biasShapes)

        activationShapes = [[None, shape[1]] for shape in groundTruthShapes]
        activations_ = actorGraph.get_collection("activations")
        generatedActivationShapes = [b_.shape.as_list() for b_ in activations_]
        self.assertEqual(generatedActivationShapes, activationShapes)

    @data(([10], [[4, 10], [10, 1], [4, 10], [10, 1]]),
          ([10, 15], [[4, 10], [10, 15], [15, 1], [4, 10], [10, 15], [15, 1]]),
          ([10, 15, 16], [[4, 10], [10, 15], [15, 16], [16, 1], [4, 10], [10, 15], [15, 16], [16, 1]])
          )
    @unpack
    def testBuildActorModelWithInput(self, layerWidths, groundTruthShapes):
        actorWriter, model = self.buildActorModel(layerWidths)
        actorGraph = model.graph
        weights_ = actorGraph.get_collection("weights")
        generatedWeightShapes = [w_.shape.as_list() for w_ in weights_]
        self.assertEqual(generatedWeightShapes, groundTruthShapes)

        biasShapes = [[shape[1]] for shape in groundTruthShapes]
        biases_ = actorGraph.get_collection("biases")
        generatedBiasShapes = [b_.shape.as_list() for b_ in biases_]
        self.assertEqual(generatedBiasShapes, biasShapes)

        activationShapes = [[None, shape[1]] for shape in groundTruthShapes]
        activations_ = actorGraph.get_collection("activations")
        generatedActivationShapes = [b_.shape.as_list() for b_ in activations_]
        self.assertEqual(generatedActivationShapes, activationShapes)


    @data(([3, 5], [[4, 3], [3, 5], [1, 5], [5, 1], [4, 3], [3, 5], [1, 5], [5, 1]]),
          ([3, 6, 7], [[4, 3], [3, 6], [6, 7], [1, 7], [7, 1],[4, 3], [3, 6], [6, 7], [1, 7], [7, 1]])
          )
    @unpack
    def testBuildCriticModel(self, layerWidths, groundTruthShapes):
        criticWriter, model = self.buildCriticModel(layerWidths)
        criticGraph = model.graph
        weights_ = criticGraph.get_collection("weights")
        generatedWeightShapes = [w_.shape.as_list() for w_ in weights_]
        self.assertEqual(generatedWeightShapes, groundTruthShapes)

        numLayersForTrain = int(len(groundTruthShapes)/2)
        groundTruthShapesForTrain = groundTruthShapes[:numLayersForTrain]
        groundTruthShapesForTrainWithoutActionLayer = groundTruthShapesForTrain[:len(groundTruthShapesForTrain)-2]+\
                                                      groundTruthShapesForTrain[len(groundTruthShapesForTrain)-1:] # remove second last element

        biasShapesForTrain = [[shape[1]] for shape in groundTruthShapesForTrainWithoutActionLayer]
        biasShapes = biasShapesForTrain + biasShapesForTrain
        biases_ = criticGraph.get_collection("biases")
        generatedBiasShapes = [b_.shape.as_list() for b_ in biases_]
        self.assertEqual(generatedBiasShapes, biasShapes)

        activationShapesForTrain = [[None, shape[1]] for shape in groundTruthShapesForTrainWithoutActionLayer]
        activationShapes = activationShapesForTrain + activationShapesForTrain
        activations_ = criticGraph.get_collection("activations")
        generatedActivationShapes = [b_.shape.as_list() for b_ in activations_]
        self.assertEqual(generatedActivationShapes, activationShapes)


    @data(([100, 200, 300, 400], 1),
          ([1, 2, 3, 4], 1)
          )
    @unpack
    def testActByTrainModelWithEmptyInputLayerWidths(self, states, miniBatchSize):
        layerWidths = []
        actorWriter, actorModel = self.buildActorModel(layerWidths)

        stateBatch = np.asarray(states).reshape(miniBatchSize, -1)
        actionAngleTrainModelOutput = actByPolicyTrain(actorModel, stateBatch)

        actorGraph = actorModel.graph
        states_ = actorGraph.get_collection_ref("states_")[0]
        weights_ = actorGraph.get_collection_ref("weights")[0]
        bias_ = actorGraph.get_collection_ref("biases")[0]
        weights, bias = actorModel.run([weights_, bias_], feed_dict={states_: stateBatch})
        activ = np.matmul(weights.transpose(), (stateBatch.transpose())) + bias
        actionAngleGroundTruth = np.tanh(activ) * np.pi

        self.assertAlmostEqual(float(actionAngleTrainModelOutput), float(actionAngleGroundTruth), places=2)


if __name__ == '__main__':
    unittest.main()
