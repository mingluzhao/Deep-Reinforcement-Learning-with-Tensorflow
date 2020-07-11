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

from algorithm.a3cAlgorithm import *
from functionTools.loadSaveModel import saveVariables, restoreVariables

def main():
    game = 'InvertedDoublePendulum-v2'
    env = gym.make(game)

    numWorkers = multiprocessing.cpu_count()
    maxTimeStepPerEps = 10000
    maxGlobalEpisode = 2000
    updateInterval = 10
    gamma = 0.99

    actorLayersWidths = [256, 256]
    criticLayersWidths = [256]

    stateDim = env.observation_space.shape[0] #11
    actionDim = env.action_space.shape[0] #1
    actionRange = [env.action_space.low, env.action_space.high] #-1 ~ 1

    getValueTargetList = GetValueTargetList(gamma)
    globalCount = Count()
    globalReward = GlobalReward()

    session = tf.Session()
    coord = tf.train.Coordinator()

    fileName = 'a3cInvDoublePendulum'
    modelPath = os.path.join(dirName, '..', 'trainedModels', fileName)
    modelSaveRate = 200
    saveModel = SaveModel(modelSaveRate, saveVariables, modelPath, session, saveAllmodels=False)

    with tf.device("/cpu:0"):
        workers = []
        globalModel = GlobalNet(stateDim, actionDim, actorLayersWidths, criticLayersWidths)

        for workerID in range(numWorkers):
            workerEnv = gym.make(game)
            workerName = 'worker_%i' % workerID
            workerNet = WorkerNet(stateDim, actionDim, actionRange, workerName, actorLayersWidths, criticLayersWidths, globalModel, session)
            worker = A3CWorkerUsingGym(maxGlobalEpisode, coord, getValueTargetList, workerEnv, maxTimeStepPerEps, globalCount, globalReward, updateInterval, workerNet, saveModel)

            workers.append(worker)

    saver = tf.train.Saver(max_to_keep=None)
    tf.add_to_collection("saver", saver)

    fileName = 'a3cInvDoublePendulum'
    modelPath = os.path.join(dirName, '..', 'trainedModels', fileName)
    restoreVariables(session, modelPath)

    for i in range(10):
        state = env.reset()
        for timestep in range(maxTimeStepPerEps):
            env.render()
            action = workerNet.act(state)
            nextState, reward, done, info = env.step(action)
            if done:
                break
            state = nextState
            time.sleep(.01)



if __name__ == '__main__':
    main()