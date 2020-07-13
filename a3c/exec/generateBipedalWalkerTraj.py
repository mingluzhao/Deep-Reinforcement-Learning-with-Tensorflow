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
import threading
import gym
import time
import matplotlib.pyplot as plt
from algorithm.a3cRNN import *
from functionTools.loadSaveModel import saveVariables, restoreVariables

def main():
    hyperparamDict = dict()
    hyperparamDict['actorLR'] = 0.00002
    hyperparamDict['criticLR'] = 0.0001
    hyperparamDict['weightInit'] = tf.random_normal_initializer(0., .1)
    hyperparamDict['actorActivFunction'] = tf.nn.relu6
    hyperparamDict['actorMuOutputActiv'] = tf.nn.tanh
    hyperparamDict['actorSigmaOutputActiv'] = tf.nn.softplus
    hyperparamDict['criticActivFunction'] = tf.nn.relu6
    hyperparamDict['actorLayersWidths'] = [512]
    hyperparamDict['criticLayersWidths'] = [512]
    hyperparamDict['entropyBeta'] = 0.001
    hyperparamDict['cellSize'] = 128

    game = 'BipedalWalker-v2'
    env = gym.make(game)

    numWorkers = multiprocessing.cpu_count()
    maxTimeStepPerEps = 1000
    maxGlobalEpisode = 8000
    updateInterval = 10
    gamma = 0.9

    stateDim = env.observation_space.shape[0]
    actionDim = env.action_space.shape[0]
    actionRange = [env.action_space.low, env.action_space.high]

    getValueTargetList = GetValueTargetList(gamma)
    globalCount = Count()
    globalReward = GlobalReward()

    session = tf.Session()
    coord = tf.train.Coordinator()
    fileName = 'a3cBipedalWalker_gym'
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
            worker = A3CWorkerUsingGym(maxGlobalEpisode, coord, getValueTargetList, workerEnv, maxTimeStepPerEps, globalCount, globalReward, updateInterval, workerNet, saveModel)

            workers.append(worker)

    saver = tf.train.Saver(max_to_keep=None)
    tf.add_to_collection("saver", saver)


    restoreVariables(session, modelPath)

    for i in range(10):
        state = env.reset()
        rnnInitState = workerNet.initializeRnnState()
        for timestep in range(maxTimeStepPerEps):
            env.render()
            action, rnnFinalState = workerNet.act(state, rnnInitState)
            nextState, reward, done, info = env.step(action)
            if done:
                break
            state = nextState
            rnnInitState = rnnFinalState
            time.sleep(.01)



if __name__ == '__main__':
    main()