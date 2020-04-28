from src.envNoPhysics import *
from src.policy import *
import os
import pickle

def loadFromPickle(path):
    pickleIn = open(path, 'rb')
    object = pickle.load(pickleIn)
    pickleIn.close()
    return object


def main():
    # xBoundary = (0, 20)
    # yBoundary = (0, 20)
    # numAgents = 2
    #
    # reset = Reset(xBoundary, yBoundary, numAgents)
    #
    # actionNoise = 0.1
    # noiseDecay = 0.999
    # actionLow = -1
    # actionHigh = 1
    # addActionNoise = AddActionNoise(actionNoise, noiseDecay, actionLow, actionHigh)
    #
    # actionPerfect = (1,1)
    # timeStep = 3
    # action = addActionNoise(actionPerfect, timeStep)
    # print(action)

    dirName = os.path.dirname(__file__)
    path = os.path.join(dirName, '..', 'trajectory', 'expModelTraj200Steps.pickle')
    trajOfExpModel = loadFromPickle(path)

    path = os.path.join(dirName, '..', 'trajectory', 'traj200steps.pickle')
    trajOfMyModel = loadFromPickle(path)

    print(trajOfExpModel)
    print(trajOfMyModel)

if __name__ == '__main__':
    main()
