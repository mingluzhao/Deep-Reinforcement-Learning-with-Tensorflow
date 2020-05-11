from dqn.src.dqn import *
from RLframework.RLrun import *
from environment.gymEnv.discreteMountainCarEnv import *
import os
import matplotlib.pyplot as plt
from functionTools.loadSaveModel import saveToPickle, GetSavePath, saveVariables
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from collections import deque

env = gym.make('MountainCar-v0')
env = env.unwrapped

def main():
    stateDim = env.observation_space.shape[0]
    actionDim = env.action_space.n
    buildModel = BuildModel(stateDim, actionDim)
    layersWidths = [30]
    writer, model = buildModel(layersWidths)
    modelList = resetTargetParamToTrainParam([model])

    learningRate = 0.001
    gamma = 0.9
    trainModelBySASRQ = TrainModelBySASRQ(learningRate, gamma, writer)

    paramUpdateInterval = 300
    updateParameters = UpdateParameters(paramUpdateInterval)
    trainModels = TrainDQNModel(getTargetQValue, trainModelBySASRQ, updateParameters)

    epsilonMax = 0.9
    epsilonDecay = 0.0002
    epsilonMin = 0
    actByTrainNetEpsilonGreedy = ActByTrainNetEpsilonGreedy(epsilonMax, epsilonMin, epsilonDecay, getTrainQValue)

    minibatchSize = 128
    learningStartBufferSize = minibatchSize
    sampleFromMemory = SampleFromMemory(minibatchSize)
    learnFromBuffer = LearnFromBuffer(learningStartBufferSize, sampleFromMemory, trainModels)

    transit = TransitMountCarDiscrete()
    getReward = rewardMountCarDiscrete
    isTerminal = IsTerminalMountCarDiscrete()
    sampleOneStep = SampleOneStep(transit, getReward, isTerminal)

    runDQNTimeStep = RunTimeStep(actByTrainNetEpsilonGreedy, sampleOneStep, learnFromBuffer)

    reset = ResetMountCarDiscrete(seed = None)
    maxTimeStep = 2000
    runEpisode = RunEpisode(reset, runDQNTimeStep, maxTimeStep)

    bufferSize = 3000
    maxEpisode = 300

    replayBuffer = deque(maxlen=int(bufferSize))
    dqn = RunAlgorithm(runEpisode, maxEpisode)
    meanRewardList, trajectory, trainedModelList = dqn(replayBuffer, modelList)
    trainedModel = trainedModelList[0]

# save Model
    parameters = {'maxEpisode': maxEpisode, 'maxTimeStep': maxTimeStep, 'minibatchSize': minibatchSize, 'gamma': gamma,
                  'learningRate': learningRate, 'epsilonDecay': epsilonDecay , 'epsilonMin': epsilonMin}

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
