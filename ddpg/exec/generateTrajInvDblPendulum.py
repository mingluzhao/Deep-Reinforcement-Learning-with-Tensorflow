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
from functionTools.loadSaveModel import restoreVariables
# def main():
#     env_name = 'InvertedDoublePendulum-v2'
#     env = gym.make(env_name)
#
#     hyperparamDict = dict()
#     hyperparamDict['actorWeightInit'] = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
#     hyperparamDict['actorActivFunction'] = [tf.nn.relu, tf.nn.relu, tf.nn.tanh]
#     hyperparamDict['actorLayersWidths'] = [400, 300]
#     hyperparamDict['actorLR'] = 1e-4
#
#     hyperparamDict['criticWeightInit'] = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
#     hyperparamDict['criticActivFunction']= [tf.nn.relu, tf.nn.relu]
#     hyperparamDict['criticLayersWidths'] = [400, 300]
#     hyperparamDict['criticLR'] = 1e-3
#
#     hyperparamDict['tau'] = 0.001
#     hyperparamDict['gamma'] = 0.99
#     hyperparamDict['gradNormClipValue'] = 5
#
#     maxEpisode = 5000* 1000
#     maxTimeStep = 100
#     bufferSize = 1e5
#     minibatchSize = 128
#
#     session = tf.Session()
#
#     stateDim = env.observation_space.shape[0] #11
#     actionDim = env.action_space.shape[0] #1
#     getActorNetwork = GetActorNetwork(hyperparamDict, batchNorm= True)
#     actor = Actor(getActorNetwork, stateDim, actionDim, session, hyperparamDict, actionRange= 3)
#
#     getCriticNetwork = GetCriticNetwork(hyperparamDict, addActionToLastLayer = True, batchNorm = True)
#     critic = Critic(getCriticNetwork, stateDim, actionDim, session, hyperparamDict)
#
#     saver = tf.train.Saver(max_to_keep=None)
#     tf.add_to_collection("saver", saver)
#     session.run(tf.global_variables_initializer())
#
#     fileName = 'ddpg_mujoco_invDblPendulum'
#     modelPath = os.path.join(dirName, '..', 'trainedModels', fileName)
#     modelSaveRate = 500
#     saveModel = SaveModel(modelSaveRate, saveVariables, modelPath, session)
#
#     trainDDPGOneStep = TrainDDPGModelsOneStep(reshapeBatchToGetSASR, actor, critic)
#
#     learningStartBufferSize = minibatchSize
#     learnFromBuffer = LearnFromBuffer(learningStartBufferSize, trainDDPGOneStep, learnInterval = 1)
#
#     buffer = MemoryBuffer(bufferSize, minibatchSize)
#
#     noiseMu = np.zeros((actionDim, 1))
#     noiseSigma = 0.05
#     noise = OrnsteinUhlenbeckActionNoise(noiseMu, noiseSigma)
#
#     actOneStep = ActOneStepForInvPendulum(actor, maxEpisode, env.action_space)
#     ddpg = TrainDDPGWithGym(maxEpisode, maxTimeStep, buffer, noise, actOneStep, learnFromBuffer, env, saveModel)
#
#     episodeRewardList = ddpg()
#     plt.plot(range(len(episodeRewardList)), episodeRewardList)


def main():
    env_name = 'InvertedDoublePendulum-v2'
    env = gym.make(env_name)

    hyperparamDict = dict()
    hyperparamDict['actorWeightInit'] = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
    hyperparamDict['actorBiasInit'] = tf.constant_initializer(0.01)
    hyperparamDict['actorActivFunction'] = [tf.nn.relu, tf.nn.relu, tf.nn.tanh]
    hyperparamDict['actorLayersWidths'] = [400, 300]
    hyperparamDict['actorLR'] = 1e-4

    hyperparamDict['criticWeightInit'] = tf.random_uniform_initializer(-0.003, maxval=0.003)
    hyperparamDict['criticBiasInit'] = tf.constant_initializer(0.01)
    hyperparamDict['criticActivFunction']= [tf.nn.relu, tf.nn.relu]
    hyperparamDict['criticLayersWidths'] = [400, 300]
    hyperparamDict['criticLR'] = 1e-2

    hyperparamDict['tau'] = 0.001
    hyperparamDict['gamma'] = 0.99
    hyperparamDict['gradNormClipValue'] = 5

    maxEpisode = 100000
    maxTimeStep = 200
    bufferSize = 1e6
    minibatchSize = 256
    noiseDecayStartStep = 25e3

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
    restoreVariables(session, modelPath)

    for i in range(50):
        state = env.reset()
        epsReward = 0
        for timestep in range(maxTimeStep):
            env.render()
            state = state.reshape(1, -1)
            action = actor.actByTarget(state)
            nextState, reward, done, info = env.step(action)
            epsReward+= reward
            if done:
                print(epsReward)
                break
            state = nextState



if __name__ == '__main__':
    main()