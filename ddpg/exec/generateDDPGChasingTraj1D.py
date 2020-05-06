import os
import pickle

DIRNAME = os.path.dirname(__file__)
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from environment.chasingEnv.envNoPhysics import *
from environment.chasingEnv.chasingPolicy import *

from src.ddpg import *
from src.policy import *
from functionTools.trajectory import *
from functionTools.loadSaveModel import *
from environment.chasingEnv.continuousChasingVisualization import *
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def main():
    dirName = os.path.dirname(__file__)
    numAgents = 2
    numStateSpace = numAgents * 2
    actionLow = -np.pi
    actionHigh = np.pi
    actionRange = (actionHigh - actionLow) / 2.0
    actionDim = 1
    
    buildActorModel = BuildActorModel(numStateSpace, actionDim, actionRange)
    actorLayerWidths = [20, 20]
    actorWriter, actorModel = buildActorModel(actorLayerWidths)

    # sheep NN Policy
    sheepActModelPath = os.path.join(dirName, '..', 'trainedDDPGModels', 'actorModel=0_dimension=2_gamma=0.9_learningRateActor=0.001_learningRateCritic=0.001_maxEpisode=200_maxTimeStep=200_minibatchSize=32.ckpt')
    restoreVariables(actorModel, sheepActModelPath)
    velocity = 1
    actByAngle = ActByAngle(velocity)
    sheepPolicy = ActByDDPG1D(actByAngle, actByPolicyTrain, actorModel)

###physics
    xBoundary = (0, 20)
    yBoundary = (0, 20)
    stayWithinBoundary = StayWithinBoundary(xBoundary, yBoundary)
    transit = TransitForNoPhysics(stayWithinBoundary)

    sheepId = 0
    wolfId = 1
    getSheepXPos = GetAgentPosFromState(sheepId)
    getWolfXPos = GetAgentPosFromState(wolfId)

    actionMagnitude = 1
    wolfPolicy = HeatSeekingContinuousDeterministicPolicy(getWolfXPos, getSheepXPos, actionMagnitude)

    killzoneRadius = 1
    isTerminal = IsTerminal(getWolfXPos, getSheepXPos, killzoneRadius)
    numAgents = 2
    reset = Reset(xBoundary, yBoundary, numAgents)

    policy = lambda state: list(sheepPolicy(state)) + list(wolfPolicy(state))
    maxRunningSteps = 20        # max possible length of the trajectory/episode
    sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, reset)
    trajectory = sampleTrajectory(policy)

    dataPath = os.path.join(dirName, '..', 'trajectory', 'traj200steps' + '.pickle')
    saveToPickle(trajectory, dataPath)

if __name__ == '__main__':
    main()


# always output action [-1, 0] -> training output angle = pi -> activation output = +-1