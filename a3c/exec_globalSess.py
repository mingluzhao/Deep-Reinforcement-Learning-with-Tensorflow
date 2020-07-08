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


from a3c.a3cWithGlobalSession import *
from environment.gymEnv.pendulumEnv import TransitGymPendulum, RewardGymPendulum, isTerminalGymPendulum, \
    observe, angle_normalize, VisualizeGymPendulum, ResetGymPendulum


def main():
    game = 'Pendulum-v0'
    # numWorkers = multiprocessing.cpu_count()
    numWorkers = 2
    maxTimeStepPerEps = 200
    maxGlobalEpisode = 2000
    tmax = 10
    gamma = 0

    actorLayersWidths = [200]
    criticLayersWidths = [100]
    env = gym.make(game)

    stateDim = env.observation_space.shape[0]
    actionDim = env.action_space.shape[0]
    actionRange = [env.action_space.low, env.action_space.high]

    globalModel = GlobalNet(stateDim, actionDim, actionRange, actorLayersWidths, criticLayersWidths)

    transit = TransitGymPendulum()
    getReward = RewardGymPendulum(angle_normalize)
    sampleOneStep = SampleOneStep(transit, getReward)
    seed = 1
    reset = ResetGymPendulum(seed)

    getValueTargetList = GetValueTargetList(gamma)
    isTerminal = lambda state: False

    globalCount = Count()
    # workersCount = [Count() for _ in range(numWorkers)]

    session = tf.Session()

    with tf.device("/cpu:0"):
        workers = []
        for workerID in range(numWorkers):
            workerName = 'worker_%i' % workerID
            a3cNet = A3C(stateDim, actionDim, actionRange, workerName, actorLayersWidths, criticLayersWidths,
                         globalModel, session)
            workers.append(a3cNet)
    COORD = tf.train.Coordinator()
    session.run(tf.global_variables_initializer())

    worker_threads = []
    for worker in workers:
        workerCount = Count()
        actNStepsBuffer = ActNStepsBuffer(isTerminal, maxTimeStepPerEps, sampleOneStep, workerCount, globalCount, observe)
        runEpisode = RunOneThreadEpisode(reset, maxTimeStepPerEps, getValueTargetList, actNStepsBuffer)
        runOneWorker = RunA3C(runEpisode, maxGlobalEpisode, COORD, globalCount)
        job = lambda: runOneWorker(worker)
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)



if __name__ == '__main__':
    main()
