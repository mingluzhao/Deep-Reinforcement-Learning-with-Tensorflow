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

from ddpg.src.ddpg_objBased import GetActorNetwork, Actor, GetCriticNetwork, Critic, ActOneStep, MemoryBuffer, \
    ExponentialDecayGaussNoise, SaveModel, TrainDDPGWithGym, LearnFromBuffer, reshapeBatchToGetSASR, TrainDDPGModelsOneStep

def main():
    env_name = 'MountainCarContinuous-v0'
    env = gym.make(env_name)

    hyperparamDict = dict()
    hyperparamDict['actorWeightInit'] = tf.random_uniform_initializer(minval=0, maxval=0.3)
    hyperparamDict['actorBiasInit'] = tf.constant_initializer(0.1)
    hyperparamDict['actorActivFunction'] = [tf.nn.relu, tf.nn.tanh]
    hyperparamDict['actorLayersWidths'] = [30]
    hyperparamDict['actorLR'] = 0.001

    hyperparamDict['criticWeightInit'] = tf.random_uniform_initializer(minval=0, maxval=0.1)
    hyperparamDict['criticBiasInit'] = tf.constant_initializer(0.1)
    hyperparamDict['criticActivFunction']= [tf.nn.relu]
    hyperparamDict['criticLayersWidths'] = [30]
    hyperparamDict['criticLR'] = 0.001

    hyperparamDict['tau'] = 0.01
    hyperparamDict['gamma'] = 0.9
    hyperparamDict['gradNormClipValue'] = 5

    maxEpisode = 300
    maxTimeStep = 2000
    bufferSize = 200000
    minibatchSize = 128

    actionHigh = env.action_space.high
    actionLow = env.action_space.low
    actionBound = (actionHigh - actionLow)/2

    session = tf.Session()

    stateDim = env.observation_space.shape[0]
    actionDim = env.action_space.shape[0]
    getActorNetwork = GetActorNetwork(hyperparamDict, batchNorm= False)
    actor = Actor(getActorNetwork, stateDim, actionDim, session, hyperparamDict, agentID= None, actionRange=actionBound)

    getCriticNetwork = GetCriticNetwork(hyperparamDict, addActionToLastLayer = True, batchNorm = False)
    critic = Critic(getCriticNetwork, stateDim, actionDim, session, hyperparamDict)

    saver = tf.train.Saver(max_to_keep=None)
    tf.add_to_collection("saver", saver)
    session.run(tf.global_variables_initializer())

    fileName = 'ddpg_mujoco_MountCar'
    modelPath = os.path.join(dirName, '..', 'trainedModels', fileName)
    modelSaveRate = 100
    saveModel = SaveModel(modelSaveRate, saveVariables, modelPath, session)

    trainDDPGOneStep = TrainDDPGModelsOneStep(reshapeBatchToGetSASR, actor, critic)

    learningStartBufferSize = minibatchSize
    learnFromBuffer = LearnFromBuffer(learningStartBufferSize, trainDDPGOneStep, learnInterval = 1)

    buffer = MemoryBuffer(bufferSize, minibatchSize)

    noiseInitVariance = 1  # control exploration
    varianceDiscount = .99995
    noiseDecayStartStep = bufferSize
    minVar = .1
    noise = ExponentialDecayGaussNoise(noiseInitVariance, varianceDiscount, noiseDecayStartStep, minVar)

    actOneStep = ActOneStep(actor, actionLow, actionHigh)
    ddpg = TrainDDPGWithGym(maxEpisode, maxTimeStep, buffer, noise, actOneStep, learnFromBuffer, env, saveModel)

    episodeRewardList = ddpg()
    plt.plot(range(len(episodeRewardList)), episodeRewardList)
    plt.show()


if __name__ == '__main__':
    main()