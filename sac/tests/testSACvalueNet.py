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
class TestSACValueNet(unittest.TestCase):
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
    def testValueTargetCalculation(self, logpi, minQCurrentState):
        tf.reset_default_graph()
        session = tf.Session()
        qNet = DoubleQNet(self.buildQNet, self.stateDim, self.actionDim, session, self.hyperparamDict)
        valueNet = ValueNet(self.buildValueNet, self.stateDim, self.actionDim, session, self.hyperparamDict)
        policyNet = PolicyNet(self.buildPolicyNet, self.stateDim, self.actionDim, session, self.hyperparamDict, self.actionRange)

        session.run(tf.global_variables_initializer())
        
        valueTarget = valueNet.session.run(valueNet.valueTargetOfUpdate_, feed_dict = {valueNet.logPi_: logpi, valueNet.minQ_: minQCurrentState})
        groundTruthValueTarget = np.array(minQCurrentState) - np.array(logpi)
        diff = np.concatenate(valueTarget - groundTruthValueTarget)
        [self.assertAlmostEqual(difference, 0) for difference in diff]


    @data(([[1,1,1,1]],  [[2]], [[2]]),
          ([[1,1,1,1], [2,2,2,2]], [[2], [5]], [[2], [8]])
          )
    @unpack
    def testValueLossCalculation(self, stateBatch, logpi, minQCurrentState):
        tf.reset_default_graph()
        session = tf.Session()
        qNet = DoubleQNet(self.buildQNet, self.stateDim, self.actionDim, session, self.hyperparamDict)
        valueNet = ValueNet(self.buildValueNet, self.stateDim, self.actionDim, session, self.hyperparamDict)
        policyNet = PolicyNet(self.buildPolicyNet, self.stateDim, self.actionDim, session, self.hyperparamDict, self.actionRange)
        session.run(tf.global_variables_initializer())

        loss = valueNet.session.run(valueNet.valueLoss_, feed_dict = {valueNet.states_: stateBatch, valueNet.logPi_: logpi, valueNet.minQ_: minQCurrentState})

        valueTrain = valueNet.session.run(valueNet.trainValue_, feed_dict = {valueNet.states_: stateBatch})
        valueTarget = np.array(minQCurrentState) - np.array(logpi)
        trueLoss = np.mean(np.square(valueTarget - valueTrain))
        self.assertAlmostEqual(trueLoss, loss, places=3)
        session.close()


    @data(([[1,1,1,1]],  [[10]], [[2]]),
          ([[1,1,1,1], [2,2,2,2]], [[2], [5]], [[2], [8]])
          )
    @unpack
    def testValueImprovement(self, stateBatch, logpi, minQCurrentState):
        tf.reset_default_graph()
        session = tf.Session()
        qNet = DoubleQNet(self.buildQNet, self.stateDim, self.actionDim, session, self.hyperparamDict)
        valueNet = ValueNet(self.buildValueNet, self.stateDim, self.actionDim, session, self.hyperparamDict)
        policyNet = PolicyNet(self.buildPolicyNet, self.stateDim, self.actionDim, session, self.hyperparamDict, self.actionRange)
        session.run(tf.global_variables_initializer())

        for i in range(10):
            lossBefore = valueNet.session.run(valueNet.valueLoss_, feed_dict={valueNet.states_: stateBatch, valueNet.logPi_: logpi,  valueNet.minQ_: minQCurrentState})
            valueNet.train(stateBatch, minQCurrentState, logpi)
            lossAfter = valueNet.session.run(valueNet.valueLoss_, feed_dict={valueNet.states_: stateBatch, valueNet.logPi_: logpi,  valueNet.minQ_: minQCurrentState})
            print(lossBefore, lossAfter)
            self.assertTrue(lossBefore > lossAfter)

    @data(([[1,1,1,1]],  [[10]], [[2]]),
          ([[1,1,1,1], [2,2,2,2]], [[2], [5]], [[2], [8]])
          )
    @unpack
    def testValueReplaceParam(self, stateBatch, logpi, minQCurrentState):
        tf.reset_default_graph()
        session = tf.Session()
        qNet = DoubleQNet(self.buildQNet, self.stateDim, self.actionDim, session, self.hyperparamDict)
        valueNet = ValueNet(self.buildValueNet, self.stateDim, self.actionDim, session, self.hyperparamDict)
        policyNet = PolicyNet(self.buildPolicyNet, self.stateDim, self.actionDim, session, self.hyperparamDict, self.actionRange)
        session.run(tf.global_variables_initializer())

        for i in range(10):
            valueNet.train(stateBatch, minQCurrentState, logpi)

        trainParams = valueNet.session.run(valueNet.trainValueParams_)
        targetParams = valueNet.session.run(valueNet.targetValueParams_)

        valueNet.updateParams()
        targetParamsAfter = valueNet.session.run(valueNet.targetValueParams_)
        trueTargetParam = (1- self.hyperparamDict['tau']) * np.array(targetParams) + self.hyperparamDict['tau']* np.array(trainParams)

        difference = np.array(targetParamsAfter) - trueTargetParam
        [self.assertEqual(np.mean(paramDiff), 0) for paramDiff in difference]


if __name__ == '__main__':
    unittest.main()
