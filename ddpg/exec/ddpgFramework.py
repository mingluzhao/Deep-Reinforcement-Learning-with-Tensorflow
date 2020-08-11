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
    SaveModel, TrainDDPGWithGym, LearnFromBuffer, reshapeBatchToGetSASR, TrainDDPGModelsOneStep
from environment.gymEnv.normalize import env_norm
from functionTools.loadSaveModel import saveToPickle, loadFromPickle
from environment.noise.noise import ExponentialDecayGaussNoise, MinusDecayGaussNoise


class LucyDDPG:
    def __init__(self, hyperparamDict):
        self.hyperparamDict = hyperparamDict

    def __call__(self, env):
        actionHigh = env.action_space.high
        actionLow = env.action_space.low
        actionBound = (actionHigh - actionLow) / 2

        session = tf.Session()
        stateDim = env.observation_space.shape[0]
        actionDim = env.action_space.shape[0]
        getActorNetwork = GetActorNetwork(self.hyperparamDict, batchNorm=True)
        actor = Actor(getActorNetwork, stateDim, actionDim, session, self.hyperparamDict, agentID=None,
                      actionRange=actionBound)

        getCriticNetwork = GetCriticNetwork(self.hyperparamDict, addActionToLastLayer=True, batchNorm=True)
        critic = Critic(getCriticNetwork, stateDim, actionDim, session, self.hyperparamDict)

        saver = tf.train.Saver(max_to_keep=None)
        tf.add_to_collection("saver", saver)
        session.run(tf.global_variables_initializer())

        modelSaveRate = 100
        saveModel = SaveModel(modelSaveRate, saveVariables, self.hyperparamDict['modelSavePathLucy'], session)

        trainDDPGOneStep = TrainDDPGModelsOneStep(reshapeBatchToGetSASR, actor, critic)
        learningStartBufferSize = self.hyperparamDict['minibatchSize']
        learnFromBuffer = LearnFromBuffer(learningStartBufferSize, trainDDPGOneStep, learnInterval=1)
        buffer = MemoryBuffer(self.hyperparamDict['bufferSize'], self.hyperparamDict['minibatchSize'])

        noise = MinusDecayGaussNoise(self.hyperparamDict['noiseInitVariance'], self.hyperparamDict['varianceDiscount'],
                                           self.hyperparamDict['noiseDecayStartStep'], self.hyperparamDict['minVar'])
        actOneStep = ActOneStep(actor, actionLow, actionHigh)

        env = env_norm(env) if self.hyperparamDict['normalizeEnv'] else env
        ddpg = TrainDDPGWithGym(self.hyperparamDict['maxEpisode'], self.hyperparamDict['maxTimeStep'], buffer, noise,
                                actOneStep, learnFromBuffer, env, saveModel)
        meanEpsRewardList = ddpg()
        saveToPickle(meanEpsRewardList, self.hyperparamDict['rewardSavePathLucy'])

        return


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
    hyperparamDict['varianceDiscount'] = 1e-5
    hyperparamDict['noiseDecayStartStep'] = hyperparamDict['bufferSize']
    hyperparamDict['minVar'] = .1
    hyperparamDict['normalizeEnv'] = False

    fileName = 'mountainCarContinuous_'
    modelDir = os.path.join(dirName, '..', 'trainedModels', 'models')
    if not os.path.exists(modelDir):
        os.makedirs(modelDir)
    hyperparamDict['modelSavePathLucy'] = os.path.join(modelDir, fileName + 'Lucy')
    hyperparamDict['modelSavePathPhil'] = os.path.join(modelDir, fileName + 'Phil')
    hyperparamDict['modelSavePathMartin'] = os.path.join(modelDir, fileName + 'Martin')

    rewardDir = os.path.join(dirName, '..', 'trainedModels', 'rewards')
    if not os.path.exists(rewardDir):
        os.makedirs(rewardDir)
    hyperparamDict['rewardSavePathLucy'] = os.path.join(rewardDir, fileName + 'Lucy')
    hyperparamDict['rewardSavePathPhil'] = os.path.join(rewardDir, fileName + 'Phil')
    hyperparamDict['rewardSavePathMartin'] = os.path.join(rewardDir, fileName + 'Martin')

    lucyDDPG = LucyDDPG(hyperparamDict)
    lucyDDPG(env)



if __name__ == '__main__':
    main()