import pickle
import os
import glob
import pandas as pd
import numpy as np
import itertools as it


def loadFromPickle(path):
    pickleIn = open(path, 'rb')
    object = pickle.load(pickleIn)
    pickleIn.close()
    return object


def saveToPickle(data, path):
    pklFile = open(path, "wb")
    pickle.dump(data, pklFile)
    pklFile.close()


class GetSavePath:
    def __init__(self, dataDirectory, extension, fixedParameters={}):
        self.dataDirectory = dataDirectory
        self.extension = extension
        self.fixedParameters = fixedParameters

    def __call__(self, parameters):
        allParameters = dict(list(parameters.items()) + list(self.fixedParameters.items()))
        sortedParameters = sorted(allParameters.items())
        nameValueStringPairs = [parameter[0] + '=' + str(parameter[1]) for parameter in sortedParameters]

        fileName = '_'.join(nameValueStringPairs) + self.extension
        fileName = fileName.replace(" ", "")
        
        path = os.path.join(self.dataDirectory, fileName)

        return path


def readParametersFromDf(oneConditionDf):
    indexLevelNames = oneConditionDf.index.names
    parameters = {levelName: oneConditionDf.index.get_level_values(levelName)[0] for levelName in indexLevelNames}
    return parameters

def conditionDfFromParametersDict(parametersDict):
    levelNames = list(parametersDict.keys())
    levelValues = list(parametersDict.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    conditionDf = pd.DataFrame(index=modelIndex)
    return conditionDf


class LoadTrajectories:
    def __init__(self, getSavePath, loadFromPickle, fuzzySearchParameterNames=[]):
        self.getSavePath = getSavePath
        self.loadFromPickle = loadFromPickle
        self.fuzzySearchParameterNames = fuzzySearchParameterNames

    def __call__(self, parameters, parametersWithSpecificValues={}):
        parametersWithFuzzy = dict(list(parameters.items()) + [(parameterName, '*') for parameterName in self.fuzzySearchParameterNames])
        productedSpecificValues = it.product(*[[(key, value) for value in values] for key, values in parametersWithSpecificValues.items()])
        parametersFinal = np.array([dict(list(parametersWithFuzzy.items()) + list(specificValueParameter)) for specificValueParameter in productedSpecificValues])
        genericSavePath = [self.getSavePath(parameters) for parameters in parametersFinal]
        if len(genericSavePath) != 0:
            filesNames = np.concatenate([glob.glob(savePath) for savePath in genericSavePath])
        else:
            filesNames = []
        mergedTrajectories = []
        for fileName in filesNames:
            oneFileTrajectories = self.loadFromPickle(fileName)
            mergedTrajectories.extend(oneFileTrajectories)
        return mergedTrajectories

class GenerateAllSampleIndexSavePaths:
    def __init__(self, getSavePath):
        self.getSavePath = getSavePath

    def __call__(self, numSamples, pathParameters):
        parametersWithSampleIndex = lambda sampleIndex: dict(list(pathParameters.items()) + [('sampleIndex', sampleIndex)])
        genericSavePath = self.getSavePath(parametersWithSampleIndex('*'))
        existingFilesNames = glob.glob(genericSavePath)
        numExistingFiles = len(existingFilesNames)
        allIndexParameters = {sampleIndex: parametersWithSampleIndex(sampleIndex+numExistingFiles) for sampleIndex in
                              range(numSamples)}
        allSavePaths = {sampleIndex: self.getSavePath(indexParameters) for sampleIndex, indexParameters in
                        allIndexParameters.items()}

        return allSavePaths


class SaveAllTrajectories:
    def __init__(self, saveData, generateAllSampleIndexSavePaths):
        self.saveData = saveData
        self.generateAllSampleIndexSavePaths = generateAllSampleIndexSavePaths

    def __call__(self, trajectories, pathParameters):
        numSamples = len(trajectories)
        allSavePaths = self.generateAllSampleIndexSavePaths(numSamples, pathParameters)
        saveTrajectory = lambda sampleIndex: self.saveData(trajectories[sampleIndex], allSavePaths[sampleIndex])
        [saveTrajectory(sampleIndex) for sampleIndex in range(numSamples)]
        print("SAVED TRAJECTORIES")

        return None

