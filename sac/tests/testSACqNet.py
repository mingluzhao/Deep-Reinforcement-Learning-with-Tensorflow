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
from sac.src.algorithm import *

@ddt
class TestSACQNet(unittest.TestCase):
    def setUp(self):
        self.hyperparamDict = dict()
        self.hyperparamDict['valueNetWeightInit'] = tf.random_uniform_initializer(-3e-3, 3e-3)
        self.hyperparamDict['valueNetBiasInit'] = tf.random_uniform_initializer(-3e-3, 3e-3)
        self.hyperparamDict['valueNetActivFunction'] = [tf.nn.relu, tf.nn.relu]
        self.hyperparamDict['valueNetLayersWidths'] = [256, 256]

        self.hyperparamDict['qNetWeightInit'] = tf.random_uniform_initializer(-3e-3, 3e-3)
        self.hyperparamDict['qNetBiasInit'] = tf.random_uniform_initializer(-3e-3, 3e-3)
        self.hyperparamDict['qNetActivFunction'] = [tf.nn.relu, tf.nn.relu]
        self.hyperparamDict['qNetLayersWidths'] = [256, 256]

        self.hyperparamDict['policyWeightInit'] = tf.random_uniform_initializer(-3e-3, 3e-3)
        self.hyperparamDict['policyBiasInit'] = tf.random_uniform_initializer(-3e-3, 3e-3)
        self.hyperparamDict['policyActivFunction'] = [tf.nn.relu, tf.nn.relu]
        self.hyperparamDict['policyLayersWidths'] = [256, 256]
        self.hyperparamDict['policyMuWeightInit'] = tf.random_uniform_initializer(-3e-3, 3e-3)
        self.hyperparamDict['policySDWeightInit'] = tf.random_uniform_initializer(-3e-3, 3e-3)

        self.hyperparamDict['policySDlow'] = -20
        self.hyperparamDict['policySDhigh'] = 2
        self.hyperparamDict['muActivationFunc'] = tf.nn.tanh
        self.hyperparamDict['epsilon'] = 1e-6

        self.hyperparamDict['valueNetLR'] = 3e-3
        self.hyperparamDict['qNetLR'] = 3e-3
        self.hyperparamDict['policyNetLR'] = 3e-3
        self.hyperparamDict['tau'] = 0.005
        self.hyperparamDict['gamma'] = 0.99
        self.hyperparamDict['rewardScale'] = 1  #

        self.stateDim = 4
        self.actionDim = 2
        self.actionRange = [-2, 2]

        self.buildValueNet = BuildValueNet(self.hyperparamDict)
        self.buildQNet = BuildQNet(self.hyperparamDict)
        self.buildPolicyNet = BuildPolicyNet(self.hyperparamDict)
        self.gamma = self.hyperparamDict['gamma']

    @data(([[0]], [[1]]),
          ([[2], [5]], [[2], [8]])
          )
    @unpack
    def testValueTargetCalculation(self, rewardBatch, nextStateValueTarget):
        tf.reset_default_graph()
        rewardBatch = [[0]]
        nextStateValueTarget = [[1]]
        session = tf.Session()
        qNet = DoubleQNet(self.buildQNet, self.stateDim, self.actionDim, session, self.hyperparamDict)
        valueNet = ValueNet(self.buildValueNet, self.stateDim, self.actionDim, session, self.hyperparamDict)
        policyNet = PolicyNet(self.buildPolicyNet, self.stateDim, self.actionDim, session, self.hyperparamDict, self.actionRange)

        session.run(tf.global_variables_initializer())
        
        valueTarget = qNet.session.run(qNet.qTarget_, feed_dict = {qNet.reward_: rewardBatch, qNet.nextValueTarget_: nextStateValueTarget})
        groundTruthValueTarget = np.array(rewardBatch) + self.gamma* np.array(nextStateValueTarget)
        diff = np.concatenate(valueTarget - groundTruthValueTarget)
        [self.assertAlmostEqual(difference, 0) for difference in diff]



    @data(([[1,1,1,1]], [[2, 2]], [[2]], [[2]]),
          ([[1,1,1,1], [2,2,2,2]], [[2, 2], [3, 2]], [[2], [5]], [[2], [8]])
          )
    @unpack
    def testQLossCalculation(self, stateBatch, actionBatch, rewardBatch, nextStateValueTarget):
        tf.reset_default_graph()
        session = tf.Session()
        qNet = DoubleQNet(self.buildQNet, self.stateDim, self.actionDim, session, self.hyperparamDict)
        valueNet = ValueNet(self.buildValueNet, self.stateDim, self.actionDim, session, self.hyperparamDict)
        policyNet = PolicyNet(self.buildPolicyNet, self.stateDim, self.actionDim, session, self.hyperparamDict, self.actionRange)
        session.run(tf.global_variables_initializer())

        loss = qNet.session.run(qNet.q1Loss_, feed_dict = {qNet.states_: stateBatch, qNet.actions_: actionBatch,
                                                              qNet.reward_: rewardBatch, qNet.nextValueTarget_: nextStateValueTarget})

        q1Val = qNet.session.run(qNet.q1TrainOutput_, feed_dict = {qNet.states_: stateBatch, qNet.actions_: actionBatch})
        valueTarget = np.array(rewardBatch) + self.gamma* np.array(nextStateValueTarget)
        trueLoss = np.mean(np.square(valueTarget - q1Val))
        self.assertAlmostEqual(trueLoss, loss, places=3)
        session.close()


    @data(([[1,1,1,1]], [[2, 2]], [[2]], [[2]]),
          ([[1,1,1,1], [2,2,2,2]], [[2, 2], [3, 2]], [[2], [5]], [[2], [8]])
          )
    @unpack
    def testQImprovement(self, stateBatch, actionBatch, rewardBatch, nextStateValueTarget):
        tf.reset_default_graph()
        session = tf.Session()
        qNet = DoubleQNet(self.buildQNet, self.stateDim, self.actionDim, session, self.hyperparamDict)
        valueNet = ValueNet(self.buildValueNet, self.stateDim, self.actionDim, session, self.hyperparamDict)
        policyNet = PolicyNet(self.buildPolicyNet, self.stateDim, self.actionDim, session, self.hyperparamDict, self.actionRange)
        session.run(tf.global_variables_initializer())

        for i in range(10):
            lossBefore = qNet.session.run(qNet.q1Loss_, feed_dict={qNet.states_: stateBatch, qNet.actions_: actionBatch,
                                                             qNet.reward_: rewardBatch, qNet.nextValueTarget_: nextStateValueTarget})
            qNet.train(stateBatch, actionBatch, rewardBatch, nextStateValueTarget)
            lossAfter = qNet.session.run(qNet.q1Loss_, feed_dict={qNet.states_: stateBatch, qNet.actions_: actionBatch,
                                                             qNet.reward_: rewardBatch, qNet.nextValueTarget_: nextStateValueTarget})
            print(lossBefore, lossAfter)
            self.assertTrue(lossBefore > lossAfter)




if __name__ == '__main__':
    unittest.main()
