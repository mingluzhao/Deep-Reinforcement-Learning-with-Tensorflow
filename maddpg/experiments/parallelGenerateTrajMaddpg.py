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


class ExcuteMultipleCodeOnConditionsParallel:
    def __init__(self, codeFileNameList, numSample, numCmdList):
        self.codeFileNameList = codeFileNameList
        self.numSample = numSample
        self.numCmdList = numCmdList

    def __call__(self, conditions):
        totalPrograms = len(self.codeFileNameList)* len(conditions)
        assert self.numCmdList >= totalPrograms, "condition number > cmd number, use more cores or less conditions"
        numCmdListPerCondition = math.floor(self.numCmdList / len(conditions))
        if self.numSample:
            startSampleIndexes = np.arange(0, self.numSample, math.ceil(self.numSample / numCmdListPerCondition))
            endSampleIndexes = np.concatenate([startSampleIndexes[1:], [self.numSample]])
            startEndIndexesPair = zip(startSampleIndexes, endSampleIndexes)
            conditionStartEndIndexesPair = list(it.product(conditions, startEndIndexesPair))
            cmdList = [['python3', fileName, json.dumps(condition)]
                       for condition in conditions for fileName in self.codeFileNameList]
        else:
            cmdList = [['python3', fileName, json.dumps(condition)]
                       for condition in conditions for fileName in self.codeFileNameList]
        processList = [Popen(cmd, stdout=PIPE, stderr=PIPE) for cmd in cmdList]
        for proc in processList:
            proc.communicate()
            # proc.wait()
        return cmdList


def main():
    startTime = time.time()
    fileNameList = ['generateTraj_maddpg_cmdControl.py']

    numSample = None
    numCpuToUse = int(0.8 * os.cpu_count())
    excuteCodeParallel = ExcuteMultipleCodeOnConditionsParallel(fileNameList, numSample, numCpuToUse)
    print("start")

    # numEntitiessList = [(3, 1, 2), (3, 2, 2), (3, 3, 2), (3, 4, 2)]
    numEntitiessList = [(2, 1, 3), (1, 1, 0)]

    conditions = []
    for numEntities in numEntitiessList:
        numWolves, numSheeps, numBlocks = numEntities
        parameters = {'numWolves': numWolves, 'numSheeps': numSheeps, 'numBlocks': numBlocks}
        conditions.append(parameters)

    cmdList = excuteCodeParallel(conditions)
    print(cmdList)

    endTime = time.time()
    print("Time taken {} seconds".format((endTime - startTime)))






if __name__ == '__main__':
    main()
