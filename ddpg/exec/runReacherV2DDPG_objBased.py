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

from src.objBased.ddpg_objBased import GetActorNetwork, Actor, GetCriticNetwork, Critic, ActOneStep, MemoryBuffer, \
    ExponentialDecayGaussNoise, SaveModel, TrainDDPGWithGym, LearnFromBuffer, reshapeBatchToGetSASR, TrainDDPGModelsOneStep


def main():
    env_name = 'Reacher-v2'
    env = gym.make(env_name)

    hyperparamDict = dict()
    hyperparamDict['actorWeightInit'] = tf.random_uniform_initializer(minval=0, maxval=0.3)
    hyperparamDict['actorBiasInit'] = tf.constant_initializer(0.1)
    hyperparamDict['actorActivFunction'] = [tf.nn.relu, tf.nn.relu, tf.nn.tanh] #
    hyperparamDict['actorLayersWidths'] = [256, 256] #
    hyperparamDict['actorLR'] = 1e-3 #

    hyperparamDict['criticWeightInit'] = tf.random_uniform_initializer(minval=0, maxval=0.1)
    hyperparamDict['criticBiasInit'] = tf.constant_initializer(0.1)
    hyperparamDict['criticActivFunction']= [tf.nn.relu, tf.nn.relu] #
    hyperparamDict['criticLayersWidths'] = [256, 256] #
    hyperparamDict['criticLR'] = 1e-3 #

    hyperparamDict['tau'] = 1e-3 #
    hyperparamDict['gamma'] = 0.99 #
    hyperparamDict['gradNormClipValue'] = .5 #

    maxEpisode = 5000
    maxTimeStep = 200
    bufferSize = 1e5 #
    minibatchSize = 128 #

    # init random action = 1e4
    # weight_decay=1e-6
    # ounoise theta = 0, sigma = 0

    actionHigh = env.action_space.high
    actionLow = env.action_space.low
    actionBound = (actionHigh - actionLow)/2

    session = tf.Session()

    stateDim = env.observation_space.shape[0] #11
    actionDim = env.action_space.shape[0] #1
    getActorNetwork = GetActorNetwork(hyperparamDict, batchNorm= False)
    actor = Actor(getActorNetwork, stateDim, actionDim, session, hyperparamDict, agentID= None, actionRange=actionBound)

    getCriticNetwork = GetCriticNetwork(hyperparamDict, addActionToLastLayer = True, batchNorm = False)
    critic = Critic(getCriticNetwork, stateDim, actionDim, session, hyperparamDict)

    saver = tf.train.Saver(max_to_keep=None)
    tf.add_to_collection("saver", saver)
    session.run(tf.global_variables_initializer())

    fileName = 'ddpg_mujoco_DblInvPendulum'
    modelPath = os.path.join(dirName, '..', 'trainedModels', fileName)
    modelSaveRate = 500
    saveModel = SaveModel(modelSaveRate, saveVariables, modelPath, session)

    trainDDPGOneStep = TrainDDPGModelsOneStep(reshapeBatchToGetSASR, actor, critic)

    learningStartBufferSize = minibatchSize
    learnFromBuffer = LearnFromBuffer(learningStartBufferSize, trainDDPGOneStep, learnInterval = 1)

    buffer = MemoryBuffer(bufferSize, minibatchSize)

    noiseDecayStartStep = 2000
    noiseInitVariance = 1
    varianceDiscount = .9995
    minVar = .1
    noise = ExponentialDecayGaussNoise(noiseInitVariance, varianceDiscount, noiseDecayStartStep, minVar)

    actOneStep = ActOneStep(actor, actionLow, actionHigh)
    ddpg = TrainDDPGWithGym(maxEpisode, maxTimeStep, buffer, noise, actOneStep, learnFromBuffer, env, saveModel)

    episodeRewardList = ddpg()
    plt.plot(range(len(episodeRewardList)), episodeRewardList)
    plt.show()


if __name__ == '__main__':
    main()