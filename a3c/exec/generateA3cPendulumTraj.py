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


from algorithm.a3cAlgorithm import *
from environment.gymEnv.pendulumEnv import TransitGymPendulum, RewardGymPendulum, isTerminalGymPendulum, \
    observe, angle_normalize, VisualizeGymPendulum, ResetGymPendulum
from functionTools.loadSaveModel import restoreVariables
from functionTools.trajectory import SampleTrajectory

def main():
    game = 'Pendulum-v0'
    numWorkers = multiprocessing.cpu_count()
    maxTimeStepPerEps = 200
    maxGlobalEpisode = 2000
    updateInterval = 10
    gamma = 0.9

    actorLayersWidths = [200]
    criticLayersWidths = [100]
    env = gym.make(game)

    stateDim = env.observation_space.shape[0]
    actionDim = env.action_space.shape[0]
    actionRange = [env.action_space.low, env.action_space.high]

    isTerminal = isTerminalGymPendulum
    transit = TransitGymPendulum()
    getReward = RewardGymPendulum(angle_normalize)
    sampleOneStep = SampleOneStep(transit, getReward)
    seed = None
    reset = ResetGymPendulum(seed)

    getValueTargetList = GetValueTargetList(gamma)

    globalCount = Count()
    globalReward = GlobalReward()

    session = tf.Session()
    coord = tf.train.Coordinator()

    with tf.device("/cpu:0"):
        workers = []
        globalModel = GlobalNet(stateDim, actionDim, actorLayersWidths, criticLayersWidths)

        for workerID in range(numWorkers):
            workerName = 'worker_%i' % workerID
            workerNet = WorkerNet(stateDim, actionDim, actionRange, workerName, actorLayersWidths, criticLayersWidths, globalModel, session)
            worker = A3CWorker(maxGlobalEpisode, coord, reset, getValueTargetList, isTerminal,
                         maxTimeStepPerEps, sampleOneStep, globalCount, globalReward, updateInterval, workerNet,
                         observe)

            workers.append(worker)

    saver = tf.train.Saver(max_to_keep=None)
    tf.add_to_collection("saver", saver)

    fileName = 'a3cPendulumModel'
    modelPath = os.path.join(dirName, '..', 'trainedModels', fileName)
    restoreVariables(session, modelPath)

    policy = lambda state: workerNet.act(observe(state))

    for i in range(10):
        maxRunningSteps = 200
        sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, getReward, reset)
        trajectory = sampleTrajectory(policy)

        # plots& plot
        showDemo = True
        if showDemo:
            visualize = VisualizeGymPendulum()
            visualize(trajectory)




if __name__ == '__main__':
    main()