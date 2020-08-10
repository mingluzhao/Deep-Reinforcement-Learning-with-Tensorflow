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
from functionTools.loadSaveModel import saveVariables
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from src.objBased.ddpg_objBased import GetActorNetwork, Actor, GetCriticNetwork, Critic, ActOneStep, MemoryBuffer, \
    ExponentialDecayGaussNoise, SaveModel, TrainDDPGWithGym, LearnFromBuffer, reshapeBatchToGetSASR, TrainDDPGModelsOneStep
from environment.gymEnv.normalize import env_norm

def myDDPG(hyperparamDict, env, modelSavePath):
    actionHigh = env.action_space.high
    actionLow = env.action_space.low
    actionBound = (actionHigh - actionLow) / 2

    session = tf.Session()
    stateDim = env.observation_space.shape[0]
    actionDim = env.action_space.shape[0]
    getActorNetwork = GetActorNetwork(hyperparamDict, batchNorm=True)
    actor = Actor(getActorNetwork, stateDim, actionDim, session, hyperparamDict, agentID=None, actionRange=actionBound)

    getCriticNetwork = GetCriticNetwork(hyperparamDict, addActionToLastLayer=True, batchNorm=True)
    critic = Critic(getCriticNetwork, stateDim, actionDim, session, hyperparamDict)

    saver = tf.train.Saver(max_to_keep=None)
    tf.add_to_collection("saver", saver)
    session.run(tf.global_variables_initializer())

    modelSaveRate = 100
    saveModel = SaveModel(modelSaveRate, saveVariables, modelSavePath, session)

    trainDDPGOneStep = TrainDDPGModelsOneStep(reshapeBatchToGetSASR, actor, critic)
    learningStartBufferSize = hyperparamDict['minibatchSize']
    learnFromBuffer = LearnFromBuffer(learningStartBufferSize, trainDDPGOneStep, learnInterval=1)
    buffer = MemoryBuffer(hyperparamDict['bufferSize'], hyperparamDict['minibatchSize'])

    noise = ExponentialDecayGaussNoise(hyperparamDict['noiseInitVariance'], hyperparamDict['varianceDiscount'], hyperparamDict['noiseDecayStartStep'], hyperparamDict['minVar'])
    actOneStep = ActOneStep(actor, actionLow, actionHigh)

    env = env_norm(env) if hyperparamDict['normalizeEnv'] else env
    ddpg = TrainDDPGWithGym(hyperparamDict['maxEpisode'], hyperparamDict['maxTimeStep'], buffer, noise, actOneStep, learnFromBuffer, env, saveModel)
    return ddpg


def main():
    env_name = 'MountainCarContinuous-v0'
    env = gym.make(env_name)

    hyperparamDict = dict()
    hyperparamDict['actorHiddenLayersWidths'] = [256] #[400, 300]
    hyperparamDict['actorActivFunction'] = [tf.nn.relu]* len(hyperparamDict['actorHiddenLayersWidths'])+ [tf.nn.tanh]
    hyperparamDict['actorHiddenLayersWeightInit'] = [tf.random_uniform_initializer(-1/np.sqrt(units), 1/np.sqrt(units)) for units in hyperparamDict['actorHiddenLayersWidths']]
    hyperparamDict['actorHiddenLayersBiasInit'] = [tf.random_uniform_initializer(-1/np.sqrt(units), 1/np.sqrt(units)) for units in hyperparamDict['actorHiddenLayersWidths']]
    hyperparamDict['actorOutputWeightInit'] = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
    hyperparamDict['actorOutputBiasInit'] = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
    hyperparamDict['actorLR'] = 1e-4

    hyperparamDict['criticHiddenLayersWidths'] = [256] #[400, 300]
    hyperparamDict['criticActivFunction'] = [tf.nn.relu]* len(hyperparamDict['criticHiddenLayersWidths'])+ [None]
    hyperparamDict['criticHiddenLayersWeightInit'] = [tf.random_uniform_initializer(-1/np.sqrt(units), 1/np.sqrt(units)) for units in hyperparamDict['criticHiddenLayersWidths']]
    hyperparamDict['criticHiddenLayersBiasInit'] = [tf.random_uniform_initializer(-1/np.sqrt(units), 1/np.sqrt(units)) for units in hyperparamDict['criticHiddenLayersWidths']]
    hyperparamDict['criticOutputWeightInit'] = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
    hyperparamDict['criticOutputBiasInit'] = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
    hyperparamDict['criticLR'] = 1e-3

    hyperparamDict['tau'] = 0.001
    hyperparamDict['gamma'] = 0.99
    hyperparamDict['minibatchSize'] = 64

    hyperparamDict['gradNormClipValue'] = None
    hyperparamDict['maxEpisode'] = 300
    hyperparamDict['maxTimeStep'] = 1000
    hyperparamDict['bufferSize'] = 1e5

    hyperparamDict['noiseInitVariance'] = 1
    hyperparamDict['varianceDiscount'] = .99995
    hyperparamDict['noiseDecayStartStep'] = hyperparamDict['bufferSize']
    hyperparamDict['minVar'] = .1
    hyperparamDict['normalizeEnv'] = False

    fileName = 'ddpg_mujoco_MountCar_normalized_paperParams'
    modelSavePath = os.path.join(dirName, '..', 'trainedModels', fileName)

    ddpg = myDDPG(hyperparamDict, env, modelSavePath)
    episodeRewardList = ddpg()
    plt.plot(range(len(episodeRewardList)), episodeRewardList)
    plt.show()


if __name__ == '__main__':
    main()