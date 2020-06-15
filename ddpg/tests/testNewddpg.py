import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import unittest
from ddt import ddt, data, unpack

from src.newddpg import *
from RLframework.RLrun import UpdateParameters

@ddt
class TestActor(unittest.TestCase):
    def setUp(self):
        numAgents = 2
        numStateSpace = numAgents * 2
        actionRange = 1
        actionDim = 2

        self.buildModel = BuildDDPGModels(numStateSpace, actionDim, actionRange)
        self.layerWidths = [64, 64]

        self.tau = 0.01
        self.gamma = 0.95
        self.learningRateActor = 0.01

        paramUpdateInterval = 1
        self.updateParameters = UpdateParameters(paramUpdateInterval, self.tau)

        self.learningRateCritic = 0.01


    def testDDPGUpdateParams(self):
        stateBatch = [[2, 5, 10, 5]]
        writer, model = self.buildModel(self.layerWidths)
        trainActorFromState = TrainActorFromState(self.learningRateActor, writer)

        runTime = 20
        for i in range(runTime):
            model = trainActorFromState(model, stateBatch)

        graph = model.graph
        actorTrainParams_ = graph.get_collection_ref("actorTrainParams_")[0]
        actorTargetParams_ = graph.get_collection_ref("actorTargetParams_")[0]
        actorTrainParams, actorTargetParams = model.run([actorTrainParams_, actorTargetParams_])

        criticTrainParams_ = graph.get_collection_ref("criticTrainParams_")[0]
        criticTargetParams_ = graph.get_collection_ref("criticTargetParams_")[0]
        criticTrainParams, criticTargetParams = model.run([criticTrainParams_, criticTargetParams_])

        updatedModel = self.updateParameters(model)

        updatedGraph = updatedModel.graph
        actorTrainParamsUpdated_ = updatedGraph.get_collection_ref("actorTrainParams_")[0]
        actorTargetParamsUpdated_ = updatedGraph.get_collection_ref("actorTargetParams_")[0]
        actorTrainParamsUpdated, actorTargetParamsUpdated = model.run([actorTrainParamsUpdated_, actorTargetParamsUpdated_])

        criticTrainParamsUpdated_ = updatedGraph.get_collection_ref("criticTrainParams_")[0]
        criticTargetParamsUpdated_ = updatedGraph.get_collection_ref("criticTargetParams_")[0]
        criticTrainParamsUpdated, criticTargetParamsUpdated = model.run([criticTrainParamsUpdated_, criticTargetParamsUpdated_])


        # update target param
        calUpdatedTargetParamActor = (1 - self.tau) * np.array(actorTargetParams) + self.tau * np.array(actorTrainParams)
        difference = np.array(actorTargetParamsUpdated) - calUpdatedTargetParamActor
        [self.assertEqual(np.mean(paramDiff), 0) for paramDiff in difference]

        calUpdatedTargetParamCritic = (1 - self.tau) * np.array(criticTargetParams) + self.tau * np.array(criticTrainParams)
        difference = np.array(criticTargetParamsUpdated) - calUpdatedTargetParamCritic
        [self.assertEqual(np.mean(paramDiff), 0) for paramDiff in difference]

        # keep train param unchanged
        difference = np.array(actorTrainParamsUpdated) - np.array(actorTrainParams)
        [self.assertEqual(np.mean(paramDiff), 0) for paramDiff in difference]

        difference = np.array(criticTrainParamsUpdated) - np.array(criticTrainParams)
        [self.assertEqual(np.mean(paramDiff), 0) for paramDiff in difference]



    @data(([[0]], [[1]]),
          ([[1], [2]], [[3], [4]])
          )
    @unpack
    def testValueTargetCalculation(self, rewardBatch, criticTargetActivation):
        writer, model = self.buildModel(self.layerWidths)
        graph = model.graph

        yi_ = graph.get_collection_ref("yi_")[0]
        gamma_ = graph.get_collection_ref("gamma_")[0]
        reward_ = graph.get_collection_ref("reward_")[0]
        criticTargetActivation_ = graph.get_collection_ref("criticTargetActivation_")[0]

        yiCalculated = model.run(yi_, feed_dict={gamma_: self.gamma, reward_: rewardBatch, criticTargetActivation_: criticTargetActivation})
        groundTruthYi = np.array(rewardBatch) + self.gamma* np.array(criticTargetActivation)
        diff = np.concatenate(yiCalculated - groundTruthYi)
        [self.assertAlmostEqual(difference, 0, places = 5) for difference in diff]


    @data(([[1,1,1,1]], [[2, 2]], [[2]], [[1,1,3,2]]),
          ([[1,1,1,1], [2,2,2,2]], [[2, 2], [3, 2]], [[2], [5]], [[1,0, 0, 1], [2,3,2,1]])
          )
    @unpack
    def testCriticLossCalculation(self, stateBatch, actionBatch, rewardBatch, nextStateBatch):
        writer, model = self.buildModel(self.layerWidths)
        trainCriticBySASR = TrainCriticBySASR(self.learningRateCritic, self.gamma, writer)

        graph = model.graph
        states_ = graph.get_collection_ref("states_")[0]
        action_ = graph.get_collection_ref("action_")[0]
        criticTrainActivationOfGivenAction_ = graph.get_collection_ref("criticTrainActivationOfGivenAction_")[0]
        criticTrainActivationOfGivenAction = model.run(criticTrainActivationOfGivenAction_, feed_dict={states_: stateBatch, action_: actionBatch})

        yi_ = graph.get_collection_ref("yi_")[0]
        nextStates_ = graph.get_collection_ref("nextStates_")[0]
        gamma_ = graph.get_collection_ref("gamma_")[0]
        reward_ = graph.get_collection_ref("reward_")[0]

        yi = model.run(yi_, feed_dict={gamma_: self.gamma, reward_: rewardBatch, nextStates_: nextStateBatch})
        trueLoss = np.mean(np.square(yi - criticTrainActivationOfGivenAction))

        calculatedLoss, model = trainCriticBySASR(model, stateBatch, actionBatch, nextStateBatch, rewardBatch)

        self.assertAlmostEqual(trueLoss, calculatedLoss)


    @data(([[1,1,1,1]], [[2, 2]], [[2]], [[1,1,3,2]]),
          ([[1, 1, 1, 1], [2, 2, 2, 2]], [[2, 2], [3, 2]], [[2], [5]], [[1, 0, 0, 1], [2, 3, 2, 1]]),
          ([[1, 0, 1, 1], [2, 2, 2, 2]], [[2, 2], [0.3, 0]], [[.2], [5]], [[1, 0, 0, 1], [2, 3, 2, 1]]),
          ([[0, 1, 1, 1], [2, 3, 2, 2]], [[2, 0], [0, 0]], [[2], [.5]], [[1, 0, 0, .1], [2, .3, 2, 1]])
          )
    @unpack
    def testCriticImprovement(self, stateBatch, actionBatch, rewardBatch, nextStateBatch):
        writer, model = self.buildModel(self.layerWidths)
        trainCriticBySASR = TrainCriticBySASR(self.learningRateCritic, self.gamma, writer)

        calculatedLoss1, model = trainCriticBySASR(model, stateBatch, actionBatch, nextStateBatch, rewardBatch)
        calculatedLoss2, model = trainCriticBySASR(model, stateBatch, actionBatch, nextStateBatch, rewardBatch)

        self.assertTrue(calculatedLoss1 > calculatedLoss2)

    @data(([[[1, 1, 1, 1]]]),
          ([[[1, 1, 1, 1]]]),
          ([[[1, 0, 1, 1]]]),
          ([[[0, 1, 1, 1]]]),
          ([[[1, 1, 1, 1], [2, 2, 2, 2]]])
          )
    @unpack
    def testActorTrainImprovement(self, stateBatch):
        writer, model = self.buildModel(self.layerWidths)
        actionUntrained = actByPolicyTrain(model, stateBatch)

        def evaluateCriticTrain(model, stateBatch, actionBatch):
            graph = model.graph
            states_ = graph.get_collection_ref("states_")[0]
            action_ = graph.get_collection_ref("action_")[0]
            criticTrainActivationOfGivenAction_ = graph.get_collection_ref("criticTrainActivationOfGivenAction_")[0]
            currentActionQVal = model.run(criticTrainActivationOfGivenAction_, feed_dict={states_: stateBatch, action_: actionBatch})
            return currentActionQVal

        actionUntrainedQVal = evaluateCriticTrain(model, stateBatch, actionUntrained)

        trainActorFromState = TrainActorFromState(self.learningRateActor, writer)
        model = trainActorFromState(model, stateBatch)

        actionTrained = actByPolicyTrain(model, stateBatch)
        actionTrainedValue = evaluateCriticTrain(model, stateBatch, actionTrained)

        [self.assertTrue(trained > untrained) for trained, untrained in zip(actionTrainedValue, actionUntrainedQVal)]


if __name__ == '__main__':
    unittest.main()
