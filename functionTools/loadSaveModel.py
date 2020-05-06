import os
import pickle

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


def saveVariables(model, path):
    graph = model.graph
    saver = graph.get_collection_ref("saver")[0]
    saver.save(model, path)
    print("Model saved in {}".format(path))


def saveToPickle(data, path):
    pklFile = open(path, "wb")
    pickle.dump(data, pklFile)
    pklFile.close()

def loadFromPickle(path):
    pickleIn = open(path, 'rb')
    object = pickle.load(pickleIn)
    pickleIn.close()
    return object

def restoreVariables(model, path):
    graph = model.graph
    saver = graph.get_collection_ref("saver")[0]
    saver.restore(model, path)
    print("Model restored from {}".format(path))
    return model