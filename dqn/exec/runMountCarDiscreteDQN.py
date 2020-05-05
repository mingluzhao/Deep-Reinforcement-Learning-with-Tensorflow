from dqn.src.dqn import *
from RLframework.RLrun import *
from environment.gymEnv.discreteMountainCarEnv import *
import os
import matplotlib.pyplot as plt
from functionTools.loadSaveModel import saveToPickle, GetSavePath, saveVariables
os.environ['KMP_DUPLICATE_LIB_OK']='True'

seed = 1
env = gym.make('MountainCar-v0')
env = env.unwrapped

def main():
    stateDim = env.observation_space.shape[0]
    actionDim = env.action_space.n
    buildModel = BuildModel(stateDim, actionDim)
    layersWidths = [64, 64]
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


    minibatchSize = 32
    learningStartBufferSize = 1000
    transit = TransitMountCarDiscrete()
    # getReward = modifiedRewardMountCarDiscrete
    getReward = rewardMountCarDiscrete
    isTerminal = IsTerminalMountCarDiscrete()
    runDQNTimeStep = RunTimeStep(actByTrainNetEpsilonGreedy, transit, getReward, isTerminal, addToMemory,
                 trainModels, minibatchSize, learningStartBufferSize)

    resetLowPos = -1
    resetHighPos = -0.8
    reset = ResetMountCarDiscrete(seed, resetLowPos, resetHighPos)
    maxTimeStep = 2000
    runEpisode = RunEpisode(reset, runDQNTimeStep, maxTimeStep)

    bufferSize = 3000
    maxEpisode = 50
    dqn = RunAlgorithm(runEpisode, bufferSize, maxEpisode)
    meanRewardList, trajectory, modelList = dqn(modelList)
    trainedModel = modelList[0]

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

    showDemo = True
    if showDemo:
        visualize = VisualizeMountCarDiscrete()
        visualize(trajectory[-50000: ])


if __name__ == "__main__":
    main()
