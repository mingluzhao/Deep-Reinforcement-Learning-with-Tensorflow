import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
sys.path.append(os.path.join(dirName, '..', '..', '..'))

from environment.gymEnv.visualizeMultiAgent import *
from environment.gymEnv.multiAgentEnv_func import getPosFromAgentState
from functionTools.loadSaveModel import loadFromPickle

wolfSize = 0.075
sheepSize = 0.05
blockSize = 0.2

wolfColor = np.array([0.85, 0.35, 0.35])
sheepColor = np.array([0.35, 0.85, 0.35])
blockColor = np.array([0.25, 0.25, 0.25])

def main():

    # check rendering
    trajectoryPath = os.path.join(dirName, 'trajectoryFull.pickle')
    traj = loadFromPickle(trajectoryPath)

    entitiesSizeList = [wolfSize, wolfSize, sheepSize, blockSize]
    entitiesColorList = [wolfColor, wolfColor, sheepColor, blockColor]

    numAgents = 3
    render = Render(entitiesSizeList, entitiesColorList, numAgents, getPosFromAgentState)

    render(traj)


if __name__ == '__main__':
    main()