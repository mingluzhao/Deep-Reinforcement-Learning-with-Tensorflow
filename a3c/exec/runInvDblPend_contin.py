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
import matplotlib.pyplot as plt
from algorithm.a3cContinuous import *
from functionTools.loadSaveModel import saveVariables
import json

def main():
    debug = 1
    hyperparamDict = dict()

    if debug:
        weightSd = 0.1
        hyperparamDict['actorLR'] = 1e-4
        hyperparamDict['criticLR'] = 1e-3
        hyperparamDict['entropyBeta'] = 0.01
        maxTimeStepPerEps = 200
        maxGlobalEpisode = 10000
        updateInterval = 20
        gamma = 0.99
        numWorkers = multiprocessing.cpu_count()
        bootStrap = 0

    else:
        condition = json.loads(sys.argv[1])
        weightSd = float(condition['weightSd'])
        hyperparamDict['actorLR'] = float(condition['actorLR'])
        hyperparamDict['criticLR'] = float(condition['criticLR'])
        hyperparamDict['entropyBeta'] = float(condition['entropyBeta'])
        maxTimeStepPerEps = int(condition['maxTimeStepPerEps'])
        maxGlobalEpisode = int(condition['maxGlobalEpisode'])
        updateInterval = int(condition['updateInterval'])
        gamma = float(condition['gamma'])
        numWorkers = int(condition['numWorkers'])
        bootStrap = int(condition['bootStrap'])

    # fixed parameters
    hyperparamDict['weightInit'] = tf.truncated_normal_initializer(mean=0.0, stddev=weightSd)
    hyperparamDict['actorActivFunction'] = tf.nn.relu6
    hyperparamDict['actorMuOutputActiv'] = tf.nn.tanh
    hyperparamDict['actorSigmaOutputActiv'] = tf.nn.softplus
    hyperparamDict['criticActivFunction'] = tf.nn.relu6
    hyperparamDict['actorLayersWidths'] = [200]
    hyperparamDict['criticLayersWidths'] = [200]

    # environment
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
    session.run(tf.global_variables_initializer())

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)

    coord.join(worker_threads)

    imageSavePath = os.path.join(dirName, '..', 'plots')
    if not os.path.exists(imageSavePath):
        os.makedirs(imageSavePath)
    plt.plot(range(len(globalReward.reward)), globalReward.reward)
    plt.savefig(os.path.join(imageSavePath, fileName + str('.png')))


if __name__ == '__main__':
    main()




