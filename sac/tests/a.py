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
from tensorflow.python.framework import ops

hyperparamDict = dict()
hyperparamDict['valueNetWeightInit'] = tf.random_uniform_initializer(-3e-3, 3e-3)
hyperparamDict['valueNetBiasInit'] = tf.random_uniform_initializer(-3e-3, 3e-3)
hyperparamDict['valueNetActivFunction'] = [tf.nn.relu, tf.nn.relu]
hyperparamDict['valueNetLayersWidths'] = [256, 256]

hyperparamDict['qNetWeightInit'] = tf.random_uniform_initializer(-3e-3, 3e-3)
hyperparamDict['qNetBiasInit'] = tf.random_uniform_initializer(-3e-3, 3e-3)
hyperparamDict['qNetActivFunction'] = [tf.nn.relu, tf.nn.relu]
hyperparamDict['qNetLayersWidths'] = [256, 256]

hyperparamDict['policyWeightInit'] = tf.random_uniform_initializer(-3e-3, 3e-3)
hyperparamDict['policyBiasInit'] = tf.random_uniform_initializer(-3e-3, 3e-3)
hyperparamDict['policyActivFunction'] = [tf.nn.relu, tf.nn.relu]
hyperparamDict['policyLayersWidths'] = [256, 256]
hyperparamDict['policyMuWeightInit'] = tf.random_uniform_initializer(-3e-3, 3e-3)
hyperparamDict['policySDWeightInit'] = tf.random_uniform_initializer(-3e-3, 3e-3)

hyperparamDict['policySDlow'] = -20
hyperparamDict['policySDhigh'] = 2
hyperparamDict['muActivationFunc'] = tf.nn.tanh
hyperparamDict['epsilon'] = 1e-6

hyperparamDict['valueNetLR'] = 3e-3
hyperparamDict['qNetLR'] = 3e-3
hyperparamDict['policyNetLR'] = 3e-3
hyperparamDict['tau'] = 0.005
hyperparamDict['gamma'] = 0.99
hyperparamDict['rewardScale'] = 1  #

stateDim = 4
actionDim = 2
actionRange = [-2, 2]

buildValueNet = BuildValueNet(hyperparamDict)
buildQNet = BuildQNet(hyperparamDict)
buildPolicyNet = BuildPolicyNet(hyperparamDict)
gamma = hyperparamDict['gamma']


for i in range(2):
    tf.reset_default_graph()
    rewardBatch = [[0]]
    nextStateValueTarget = [[1]]

    session = tf.Session()
    qNet = DoubleQNet(buildQNet, stateDim, actionDim, session, hyperparamDict)
    valueNet = ValueNet(buildValueNet, stateDim, actionDim, session, hyperparamDict)
    policyNet = PolicyNet(buildPolicyNet, stateDim, actionDim, session, hyperparamDict, actionRange)

    session.run(tf.global_variables_initializer())

    valueTarget = qNet.session.run(qNet.qTarget_, feed_dict = {qNet.reward_: rewardBatch, qNet.nextValueTarget_: nextStateValueTarget})
    groundTruthValueTarget = np.array(rewardBatch) + gamma* np.array(nextStateValueTarget)
    diff = np.concatenate(valueTarget - groundTruthValueTarget)
    print(diff)

