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
from collections import OrderedDict
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
    fileName = 'train_sourceCodeWithRefEnviron.py'
    numSample = None
    numCpuToUse = int(0.8 * os.cpu_count())
    excuteCodeParallel = ExcuteCodeOnConditionsParallel(fileName, numSample, numCpuToUse)
    print("start")

    numWolvesLevels = [3]
    numSheepsLevels = [1]
    numBlocksLevels = [2]
    fileIDList = list(range(2))

    conditionLevels = [(wolfNum, sheepNum, blockNum, fileID)
                       for wolfNum in numWolvesLevels
                       for sheepNum in numSheepsLevels
                       for blockNum in numBlocksLevels
                       for fileID in fileIDList]

    conditions = []
    for condition in conditionLevels:
        numWolves, numSheeps, numBlocks, fileID = condition
        parameters = {'numWolves': numWolves, 'numSheeps': numSheeps, 'numBlocks': numBlocks,'fileID': fileID}
        conditions.append(parameters)


    cmdList = excuteCodeParallel(conditions)
    print(cmdList)

    endTime = time.time()
    print("Time taken {} seconds".format((endTime - startTime)))

if __name__ == '__main__':
    main()
