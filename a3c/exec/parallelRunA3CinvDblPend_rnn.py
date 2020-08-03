import time
import sys
import os
DIRNAME = os.path.dirname(__file__)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.append(os.path.join(DIRNAME, '..', '..', '..'))

from subprocess import Popen, PIPE
import json
import math
import numpy as np
import itertools as it

class ExcuteCodeOnConditionsParallel:
    def __init__(self, codeFileName, numSample, numCmdList):
        self.codeFileName = codeFileName
        self.numSample = numSample
        self.numCmdList = numCmdList

    def __call__(self, conditions):
        assert self.numCmdList >= len(conditions), "condition number > cmd number, use more cores or less conditions"
        numCmdListPerCondition = math.floor(self.numCmdList / len(conditions))
        if self.numSample:
            startSampleIndexes = np.arange(0, self.numSample, math.ceil(self.numSample / numCmdListPerCondition))
            endSampleIndexes = np.concatenate([startSampleIndexes[1:], [self.numSample]])
            startEndIndexesPair = zip(startSampleIndexes, endSampleIndexes)
            conditionStartEndIndexesPair = list(it.product(conditions, startEndIndexesPair))
            cmdList = [['python3', self.codeFileName, json.dumps(condition), str(startEndSampleIndex[0]), str(startEndSampleIndex[1])]
                       for condition, startEndSampleIndex in conditionStartEndIndexesPair]
        else: 
            cmdList = [['python3', self.codeFileName, json.dumps(condition)]
                       for condition in conditions]
        processList = [Popen(cmd, stdout=PIPE, stderr=PIPE) for cmd in cmdList]
        for proc in processList:
            proc.communicate()
            # proc.wait()
        return cmdList

def main():
    startTime = time.time()
    fileName = 'runInvDblPend_rnn.py'
    numSample = None
    numCpuToUse = int(0.8 * os.cpu_count())
    excuteCodeParallel = ExcuteCodeOnConditionsParallel(fileName, numSample, numCpuToUse)
    print("start")

    weightSdLevels = [0.001, 0.01]
    actorLrLevels = [1e-4, 1e-5]
    criticLrLevels = [1e-4, 1e-5]
    entropyBetaLevels = [0.01]
    maxTimeStepPerEpsLevels = [100, 200]
    maxGlobalEpisodeLevels = [1e4, 1e5]
    updateIntervalLevels = [20]
    gammaLevels = [0.99]
    numWorkersLevels = [4, 8, 16]
    bootLevels = [0, 1]

    conditionLevels = [(weightSd, actorLR, criticLR, entropyBeta, maxTimeStepPerEps, maxGlobalEpisode, updateInterval, gamma, numWorkers, bootStrap)
                       for weightSd in weightSdLevels
                       for actorLR in actorLrLevels
                       for criticLR in criticLrLevels
                       for entropyBeta in entropyBetaLevels
                       for maxTimeStepPerEps in maxTimeStepPerEpsLevels
                       for maxGlobalEpisode in maxGlobalEpisodeLevels
                       for updateInterval in updateIntervalLevels
                       for gamma in gammaLevels
                       for numWorkers in numWorkersLevels
                       for bootStrap in bootLevels
                       ]

    conditions = []
    for condition in conditionLevels:
        weightSd, actorLR, criticLR, entropyBeta, maxTimeStepPerEps, maxGlobalEpisode, updateInterval, gamma, numWorkers, bootStrap= condition
        parameters = {'weightSd': weightSd, 'actorLR': actorLR, 'criticLR': criticLR,
                      'entropyBeta': entropyBeta, 'maxTimeStepPerEps': maxTimeStepPerEps,
                      'maxGlobalEpisode': maxGlobalEpisode, 'updateInterval': updateInterval,
                      'gamma': gamma, 'numWorkers': numWorkers, 'bootStrap': bootStrap}
        conditions.append(parameters)

    cmdList = excuteCodeParallel(conditions)
    print(cmdList)

    endTime = time.time()
    print("Time taken {} seconds".format((endTime - startTime)))

if __name__ == '__main__':
    main()
