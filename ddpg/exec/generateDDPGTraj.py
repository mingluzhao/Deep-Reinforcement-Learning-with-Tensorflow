import os
import pickle

DIRNAME = os.path.dirname(__file__)
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from src.envNoPhysics import *
from src.ddpg import *
from src.policy import *
from src.continuousVisualization import *
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def restoreVariables(model, path):
    graph = model.graph
    saver = graph.get_collection_ref("saver")[0]
    saver.restore(model, path)
    print("Model restored from {}".format(path))
    return model


class SampleTrajectory:
    def __init__(self, maxRunningSteps, transit, isTerminal, reset):
        self.maxRunningSteps = maxRunningSteps
        self.transit = transit
        self.isTerminal = isTerminal
        self.reset = reset

    def __call__(self, policy):
        state = self.reset()
        while self.isTerminal(state):
            state = self.reset()

        trajectory = [state]
        for runningStep in range(self.maxRunningSteps):
            if self.isTerminal(state):
                break
            action = policy(state)
            nextState = self.transit(state, action)
            trajectory.append(nextState)
            state = nextState
        return trajectory

def saveToPickle(data, path):
    pklFile = open(path, "wb")
    pickle.dump(data, pklFile)
    pklFile.close()

def main():
    # Neural Network
    dirName = os.path.dirname(__file__)
    
    numAgents = 2
    numStateSpace = numAgents * 2
    actionLow = -np.pi
    actionHigh = np.pi
    actionRange = (actionHigh - actionLow) / 2.0
    actionDim = 1
    
    buildActorModel = BuildActorModel(numStateSpace, actionDim, actionRange)
    actorTrainingLayerWidths = [20, 20]
    actorTargetLayerWidths = actorTrainingLayerWidths
    actorWriter, actorModel = buildActorModel(actorTrainingLayerWidths, actorTargetLayerWidths)

    # sheep NN Policy
    sheepActModelPath = os.path.join(dirName, '..', 'trainedDDPGModels', 'actorModel=0_gamma=0.95_learningRateActor=0.0001_learningRateCritic=0.001_maxEpisode=50_maxTimeStep=200_minibatchSize=32.ckpt')

    restoreVariables(actorModel, sheepActModelPath)
    velocity = 1
    actByAngle = ActByAngle(velocity)
    sheepPolicy = ActByDDPG(actByAngle, actByPolicyTrain, actorModel)

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


# always output action [-1, 0] -> training output angle = pi -> activation output = 1