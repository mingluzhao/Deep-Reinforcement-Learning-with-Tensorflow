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
from sac.src.algorithm_withV_IP import *

@ddt
class TestSACPolicyNet(unittest.TestCase):
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

    # @data(([[1,1,1,1]], [100]),
    #       ([[1,1,1,1], [2,2,2,2]], [2, 8])
    #       )
    # @unpack
    # def testPolicyLossCalculation(self, stateBatch, minQWithCurrentStateAndPolicyNetAction):
    #     tf.reset_default_graph()
    #     session = tf.Session()
    #     qNet = DoubleQNet(self.buildQNet, self.stateDim, self.actionDim, session, self.hyperparamDict)
    #     valueNet = ValueNet(self.buildValueNet, self.stateDim, self.actionDim, session, self.hyperparamDict)
    #     policyNet = PolicyNet(self.buildPolicyNet, self.stateDim, self.actionDim, session, self.hyperparamDict, self.actionRange)
    #     session.run(tf.global_variables_initializer())
    #
    #     logpi = policyNet.session.run(policyNet.logPi_, feed_dict = {policyNet.states_: stateBatch})
    #     trueLoss = np.mean(np.array(logpi, dtype = 'float32') - np.array(minQWithCurrentStateAndPolicyNetAction, dtype = 'float32'))
    #
    #     loss = policyNet.session.run(policyNet.policyLoss, feed_dict = {policyNet.logPi_: logpi, policyNet.minQ_: minQWithCurrentStateAndPolicyNetAction})
    #     self.assertAlmostEqual(trueLoss, loss, places=5)
    #     session.close()
    #
    #
    # @data(([[1,1,1,1]], [-100]),
    #       ([[1,1,1,1], [2,2,2,2]], [2, 8])
    #       )
    # @unpack
    # # def testPolicyImprovement(self, stateBatch, minQWithCurrentStateAndPolicyNetAction):
    # def testPolicyLoss(self, stateBatch, minQWithCurrentStateAndPolicyNetAction):
    #     tf.reset_default_graph()
    #     session = tf.Session()
    #     qNet = DoubleQNet(self.buildQNet, self.stateDim, self.actionDim, session, self.hyperparamDict)
    #     valueNet = ValueNet(self.buildValueNet, self.stateDim, self.actionDim, session, self.hyperparamDict)
    #     policyNet = PolicyNet(self.buildPolicyNet, self.stateDim, self.actionDim, session, self.hyperparamDict, self.actionRange)
    #     session.run(tf.global_variables_initializer())
    #
    #     logpi, loss = policyNet.session.run([policyNet.logPi_, policyNet.policyLoss], feed_dict = {policyNet.states_: stateBatch, policyNet.minQ_: minQWithCurrentStateAndPolicyNetAction})
    #     trueLoss = np.mean(np.array(logpi, dtype = 'float32') - np.array(minQWithCurrentStateAndPolicyNetAction, dtype = 'float32'))
    #
    #     self.assertAlmostEqual(trueLoss, loss, places=5)
    #     session.close()


    @data(([[1,1,1,1]], [-1]),
          ([[1,1,1,1], [2,2,2,2]], [2, 8])
          )
    @unpack
    def testPolicyImprovement(self, stateBatch, minQWithCurrentStateAndPolicyNetAction):
        tf.reset_default_graph()
        session = tf.Session()
        qNet = DoubleQNet(self.buildQNet, self.stateDim, self.actionDim, session, self.hyperparamDict)
        valueNet = ValueNet(self.buildValueNet, self.stateDim, self.actionDim, session, self.hyperparamDict)
        policyNet = PolicyNet(self.buildPolicyNet, self.stateDim, self.actionDim, session, self.hyperparamDict, self.actionRange)
        session.run(tf.global_variables_initializer())

        for runGroup in range(10):
            lossBefore = policyNet.session.run(policyNet.policyLossFortest, feed_dict={policyNet.states_: stateBatch, policyNet.minQtest_: minQWithCurrentStateAndPolicyNetAction})
            for i in range(1000):
                policyNet.train(stateBatch, minQWithCurrentStateAndPolicyNetAction)
            lossAfter = policyNet.session.run(policyNet.policyLossFortest, feed_dict={policyNet.states_: stateBatch, policyNet.minQtest_: minQWithCurrentStateAndPolicyNetAction})
            print(lossBefore, lossAfter)
            # self.assertTrue(lossBefore > lossAfter)

        # for runGroup in range(10):
        #     lossBefore = policyNet.session.run(policyNet.policyLossFortest, feed_dict={policyNet.states_: stateBatch, policyNet.minQtest_: minQWithCurrentStateAndPolicyNetAction})
        #     actDet = policyNet.session.run(policyNet.actionDeterministic, feed_dict={policyNet.states_: stateBatch})
        #     # print(actDet)
        #
        #     for i in range(1000):
        #         policyNet.session.run(policyNet.policyOptTest_, feed_dict = {policyNet.states_: stateBatch, policyNet.minQtest_: minQWithCurrentStateAndPolicyNetAction})
        #     lossAfter = policyNet.session.run(policyNet.policyLossFortest, feed_dict={policyNet.states_: stateBatch, policyNet.minQtest_: minQWithCurrentStateAndPolicyNetAction})
        #     print(lossBefore, lossAfter)

        actDet = policyNet.session.run(policyNet.actionDeterministic, feed_dict={policyNet.states_: stateBatch})
        # print(actDet)


if __name__ == '__main__':
    unittest.main()
