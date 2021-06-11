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
from sac.src.algorithm import *
from functionTools.loadSaveModel import saveVariables
import matplotlib.pyplot as plt

def main():
    hyperparamDict = dict()
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
    hyperparamDict['muActivationFunc'] = tf.nn.tanh

    hyperparamDict['policySDlow'] = -20
    hyperparamDict['policySDhigh'] = 2
    hyperparamDict['epsilon'] = 1e-6

    hyperparamDict['qNetLR'] = 3e-3
    hyperparamDict['policyNetLR'] = 3e-3
    hyperparamDict['tau'] = 0.005
    hyperparamDict['gamma'] = 0.99
    hyperparamDict['alpha'] = 0.2

    bufferSize= 1000000
    minibatchSize= 64
    learningStartBufferSize = minibatchSize
    maxEpisode = 100
    maxTimeStep = 500

    env = gym.make("Pendulum-v0")
    stateDim = env.observation_space.shape[0]
    actionDim = env.action_space.shape[0]
    actionRange = [env.action_space.low, env.action_space.high]

    buildQNet = BuildQNet(hyperparamDict)
    buildPolicyNet = BuildPolicyNet(hyperparamDict, actionRange)

    session = tf.Session()
    sacAgent =  SACAgent(buildQNet, buildPolicyNet, stateDim, actionDim, session, hyperparamDict)

    policyUpdateInterval = 1
    trainModels = TrainSoftACOneStep(sacAgent, reshapeBatchToGetSASR, policyUpdateInterval)
    learnFromBuffer = LearnFromBuffer(learningStartBufferSize, trainModels)
    memoryBuffer = MemoryBuffer(bufferSize, minibatchSize)
    actOneStep = lambda stateBatch: sacAgent.act(stateBatch)

    saver = tf.train.Saver(max_to_keep=None)
    tf.add_to_collection("saver", saver)
    session.run(tf.global_variables_initializer())

    sacAgent.reset()

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
