import matplotlib.pyplot as plt
import gym
from collections import deque
import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))
sys.path.append(os.path.join(dirName, '..'))

from dqn.src.policy import *
from dqn.src.dqn import *
from dqn.src.policy import ActByTrainNetEpsilonGreedy
from RLframework.RLrun import resetTargetParamToTrainParam, UpdateParameters, SampleOneStep, SampleFromMemory,\
    LearnFromBuffer, RunTimeStep, RunEpisode, RunAlgorithm
from environment.gymEnv.discreteMountainCarEnv import TransitMountCarDiscrete, IsTerminalMountCarDiscrete, \
    rewardMountCarDiscrete, ResetMountCarDiscrete, VisualizeMountCarDiscrete
from functionTools.loadSaveModel import saveToPickle, GetSavePath, saveVariables

env = gym.make('MountainCar-v0')
env = env.unwrapped

minibatchSize = 128
bufferSize = 200000
maxTimeStep = 2000
maxEpisode = 1000

def main():
    stateDim = env.observation_space.shape[0]
    actionDim = env.action_space.n
    buildModel = BuildModel(stateDim, actionDim)
    layersWidths = [30]
    writer, model = buildModel(layersWidths)

    learningRate = 0.001
    gamma = 0.9
    trainModelBySASRQ = TrainModelBySASRQ(learningRate, gamma, writer)

    paramUpdateInterval = 300
    updateParameters = UpdateParameters(paramUpdateInterval)
    model = resetTargetParamToTrainParam([model])[0]
    trainModels = TrainDQNModel(getTargetQValue, trainModelBySASRQ, updateParameters, model)

    epsilonMax = 0.9
    epsilonIncrease = 0.00002
    epsilonMin = 0
    decayStartStep = bufferSize
    getEpsilon = GetEpsilon(epsilonMax, epsilonMin, epsilonIncrease, decayStartStep)

    actGreedyByModel = ActGreedyByModel(getTrainQValue, model)
    actRandom = ActRandom(actionDim)
    actByTrainNetEpsilonGreedy = ActByTrainNetEpsilonGreedy(getEpsilon, actGreedyByModel, actRandom)

    learningStartBufferSize = minibatchSize
    sampleFromMemory = SampleFromMemory(minibatchSize)
    learnFromBuffer = LearnFromBuffer(learningStartBufferSize, sampleFromMemory, trainModels)

    transit = TransitMountCarDiscrete()
    getReward = rewardMountCarDiscrete
    sampleOneStep = SampleOneStep(transit, getReward)

    runDQNTimeStep = RunTimeStep(actByTrainNetEpsilonGreedy, sampleOneStep, learnFromBuffer)

    reset = ResetMountCarDiscrete(seed = None, low= -1, high= 0.4)
    isTerminal = IsTerminalMountCarDiscrete()
    runEpisode = RunEpisode(reset, runDQNTimeStep, maxTimeStep, isTerminal)

    dqn = RunAlgorithm(runEpisode, maxEpisode)
    replayBuffer = deque(maxlen=int(bufferSize))
    meanRewardList, trajectory = dqn(replayBuffer)

    trainedModel = trainModels.getTrainedModels()

# save Model
    parameters = {'maxEpisode': maxEpisode, 'maxTimeStep': maxTimeStep, 'minibatchSize': minibatchSize, 'gamma': gamma,
                  'learningRate': learningRate, 'epsilonIncrease': epsilonIncrease , 'epsilonMin': epsilonMin}

    modelSaveDirectory = "../trainedDQNModels"
    modelSaveExtension = '.ckpt'
    getSavePath = GetSavePath(modelSaveDirectory, modelSaveExtension)
    savePath = getSavePath(parameters)

    with trainedModel.as_default():
        saveVariables(trainedModel, savePath)

    dirName = os.path.dirname(__file__)
    trajectoryPath = os.path.join(dirName, '..', 'trajectory', 'mountaincarDQNTrajectory.pickle')
    saveToPickle(trajectory, trajectoryPath)

    plotResult = True
    if plotResult:
        plt.plot(list(range(maxEpisode)), meanRewardList)
        plt.show()

    showDemo = False
    if showDemo:
        visualize = VisualizeMountCarDiscrete()
        visualize(trajectory[-50000: ])


if __name__ == "__main__":
    main()
