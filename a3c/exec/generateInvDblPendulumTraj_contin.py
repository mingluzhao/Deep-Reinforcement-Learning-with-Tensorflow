import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import multiprocessing
import gym
from algorithm.a3cContinuous import *
from functionTools.loadSaveModel import saveVariables, restoreVariables

def main():
    hyperparamDict = dict()
    weightSd = 0.1
    hyperparamDict['actorLR'] = 1e-4
    hyperparamDict['criticLR'] = 1e-3
    hyperparamDict['entropyBeta'] = 0.01
    maxTimeStepPerEps = 200
    maxGlobalEpisode = 10000
    updateInterval = 20
    gamma = 0.99
    numWorkers = multiprocessing.cpu_count()
    bootStrap = 1

    hyperparamDict['weightInit'] = tf.truncated_normal_initializer(mean=0.0, stddev=weightSd)
    hyperparamDict['actorActivFunction'] = tf.nn.relu6
    hyperparamDict['actorMuOutputActiv'] = tf.nn.tanh
    hyperparamDict['actorSigmaOutputActiv'] = tf.nn.softplus
    hyperparamDict['criticActivFunction'] = tf.nn.relu6
    hyperparamDict['actorLayersWidths'] = [200]
    hyperparamDict['criticLayersWidths'] = [200]

    game = 'InvertedDoublePendulum-v2'
    env = gym.make(game)

    stateDim = env.observation_space.shape[0] #11
    actionDim = env.action_space.shape[0] #1
    actionRange = [env.action_space.low, env.action_space.high] #-1 ~ 1

    getValueTargetList = GetValueTargetList(gamma)
    globalCount = Count()
    globalReward = GlobalReward()

    session = tf.Session()
    coord = tf.train.Coordinator()

    bootStr = 'withBoot' if bootStrap else 'noBoot'
    fileName = 'a3cInvDblPendCont{}{}eps{}steps{}actlr{}crtlr{}beta{}updt{}weightDev{}gamma{}worker'.format(bootStr, maxGlobalEpisode,
            maxTimeStepPerEps, hyperparamDict['actorLR'], hyperparamDict['criticLR'], hyperparamDict['entropyBeta'],
            updateInterval, weightSd, gamma, numWorkers)
    modelPath = os.path.join(dirName, '..', 'trainedModels', fileName)
    modelSaveRate = 500
    saveModel = SaveModel(modelSaveRate, saveVariables, modelPath, session, saveAllmodels=False)

    with tf.device("/cpu:0"):
        workers = []
        globalModel = GlobalNet(stateDim, actionDim, hyperparamDict)

        for workerID in range(numWorkers):
            workerEnv = gym.make(game)
            workerName = 'worker_%i' % workerID
            workerNet = WorkerNet(stateDim, actionDim, hyperparamDict, actionRange, workerName, globalModel, session)
            worker = A3CWorkerUsingGym(maxGlobalEpisode, coord, getValueTargetList, workerEnv, maxTimeStepPerEps, globalCount, globalReward, updateInterval, workerNet, saveModel, bootStrap)

            workers.append(worker)

    saver = tf.train.Saver(max_to_keep=None)
    tf.add_to_collection("saver", saver)

    restoreVariables(session, modelPath)

    for i in range(20):
        epsReward = 0
        state = env.reset()
        for timestep in range(maxTimeStepPerEps):
            env.render()
            action = workerNet.act(state)
            nextState, reward, done, info = env.step(action)
            if done:
                epsReward += reward
                break
            state = nextState
        print(epsReward)



if __name__ == '__main__':
    main()