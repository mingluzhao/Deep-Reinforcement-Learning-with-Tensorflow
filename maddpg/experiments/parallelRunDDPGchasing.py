import time
import sys
import os
DIRNAME = os.path.dirname(__file__)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.append(os.path.join(DIRNAME, '..', '..', '..'))

from subprocess import Popen, PIPE
import json



class GenerateTrajectoriesParallel:
    def __init__(self, codeFileName):
        self.codeFileName = codeFileName

    def __call__(self, parameters):
        parametersString = dict([(key, str(value)) for key, value in parameters.items()])
        parametersStringJS = json.dumps(parametersString)
        cmdList = [['python3', self.codeFileName, parametersStringJS]]
        print(cmdList)
        processList = [Popen(cmd, stdout=PIPE, stderr=PIPE) for cmd in cmdList]
        for proc in processList:
            proc.communicate()
        return cmdList


def main():
    startTime = time.time()
    sampleTrajectoryFileName = 'runMyDDPGChasing.py'
    generateTrajectoriesParallel = GenerateTrajectoriesParallel(sampleTrajectoryFileName)
    print("start")

    numEntitiessList = [(1, 1, 0), (2, 1, 3)]

    for numEntities in numEntitiessList:
        numWolves, numSheeps, numBlocks = numEntities
        pathParameters = {'numWolves': numWolves, 'numSheeps': numSheeps, 'numBlocks': numBlocks}

        cmdList = generateTrajectoriesParallel(pathParameters)
        print(cmdList)

    endTime = time.time()
    print("Time taken {} seconds".format((endTime - startTime)))


if __name__ == '__main__':
    main()
