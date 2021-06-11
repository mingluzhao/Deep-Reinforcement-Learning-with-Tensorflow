import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import gym
from sac.src.algorithm_withV_IP import *
from functionTools.loadSaveModel import saveVariables
import matplotlib.pyplot as plt

def main():
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
    hyperparamDict['policyMuBiasInit'] = tf.random_uniform_initializer(-3e-3, 3e-3)
    hyperparamDict['policySDBiasInit'] = tf.random_uniform_initializer(-3e-3, 3e-3)

    hyperparamDict['policySDlow'] = -20
    hyperparamDict['policySDhigh'] = 2
    hyperparamDict['muActivationFunc'] = tf.nn.tanh
    hyperparamDict['epsilon'] = 1e-6

    hyperparamDict['valueNetLR'] = 3e-3
    hyperparamDict['qNetLR'] = 3e-3
    hyperparamDict['policyNetLR'] = 3e-3
    hyperparamDict['tau'] = 0.005
    hyperparamDict['gamma'] = 0.99
    hyperparamDict['rewardScale'] = 1 #

    bufferSize= 1000000
    minibatchSize= 64
    learningStartBufferSize = minibatchSize
    maxEpisode = 100
    maxTimeStep = 500

    env = gym.make("Pendulum-v0")
    stateDim = env.observation_space.shape[0]
    actionDim = env.action_space.shape[0]
    actionRange = [env.action_space.low, env.action_space.high]
    # actionRange = (env.action_space.high - env.action_space.low)/2
    #
    buildValueNet = BuildValueNet(hyperparamDict)
    buildQNet = BuildQNet(hyperparamDict)
    buildPolicyNet = BuildPolicyNet(hyperparamDict)

    session = tf.Session()
    qNet = DoubleQNet(buildQNet, stateDim, actionDim, session, hyperparamDict)
    valueNet = ValueNet(buildValueNet, stateDim, actionDim, session, hyperparamDict)
    policyNet = PolicyNet(buildPolicyNet, stateDim, actionDim, session, hyperparamDict, actionRange)

    policyUpdateInterval = 2
    trainModels = TrainSoftACOneStep(policyNet, valueNet, qNet, reshapeBatchToGetSASR, policyUpdateInterval)
    learnFromBuffer = LearnFromBuffer(learningStartBufferSize, trainModels)
    memoryBuffer = MemoryBuffer(bufferSize, minibatchSize)
    actOneStep = ActOneStep( policyNet)

    saver = tf.train.Saver(max_to_keep=None)
    tf.add_to_collection("saver", saver)
    writer = tf.summary.FileWriter('tensorBoard/', graph=session.graph)
    tf.add_to_collection("writer", writer)

    session.run(tf.global_variables_initializer())
    valueNet.hardReplaceTargetParam()

    fileName = 'softACPendulum'
    modelPath = os.path.join(dirName, '..', 'trainedModels', fileName)
    modelSaveRate = 50
    saveModel = SaveModel(modelSaveRate, saveVariables, modelPath, session, saveAllmodels=False)

    trainSoftAC = TrainSoftAC(maxEpisode, maxTimeStep, memoryBuffer, actOneStep, learnFromBuffer, env, saveModel)

    episodeRewardList = trainSoftAC()
    imageSavePath = os.path.join(dirName, '..', 'plots')
    if not os.path.exists(imageSavePath):
        os.makedirs(imageSavePath)
    plt.plot(range(len(episodeRewardList)), episodeRewardList)
    plt.savefig(os.path.join(imageSavePath, fileName + str('.png')))
    plt.show()


if __name__ == '__main__':
    main()
