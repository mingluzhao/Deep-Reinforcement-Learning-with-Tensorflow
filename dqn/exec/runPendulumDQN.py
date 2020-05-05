from dqn.src.dqn import *
from RLframework.RLrun import *
from environment.gymEnv.pendulumEnv import *
import os
import matplotlib.pyplot as plt
from functionTools.loadSaveModel import saveToPickle, GetSavePath, saveVariables
os.environ['KMP_DUPLICATE_LIB_OK']='True'

seed = 1
env = gym.make('Pendulum-v0')
env = env.unwrapped

def main():
    stateDim = env.observation_space.shape[0]
    actionDim = 3
    buildModel = BuildModel(stateDim, actionDim)
    layersWidths = [32, 32]
    writer, model = buildModel(layersWidths)
    modelList = resetTargetParamToTrainParam([model])

    learningRate = 0.0001
    gamma = 0.99
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
    processAction = ProcessDiscretePendulumAction(actionDim)
    transit = TransitGymPendulum(processAction)
    getReward = RewardGymPendulum(angle_normalize, processAction)
    runDQNTimeStep = RunTimeStep(actByTrainNetEpsilonGreedy, transit, getReward, isTerminalGymPendulum, addToMemory,
                 trainModels, minibatchSize, learningStartBufferSize, observe)

    reset = ResetGymPendulum(seed, observe)
    maxTimeStep = 200
    runEpisode = RunEpisode(reset, runDQNTimeStep, maxTimeStep)

    bufferSize = 100000
    maxEpisode = 200
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
    trajectoryPath = os.path.join(dirName, '..', 'trajectory', 'pendulumDQNTrajectory.pickle')
    saveToPickle(trajectory, trajectoryPath)

    plotResult = True
    if plotResult:
        plt.plot(list(range(maxEpisode)), meanRewardList)
        plt.show()

    showDemo = False
    if showDemo:
        visualize = VisualizeGymPendulum()
        visualize(trajectory)


if __name__ == "__main__":
    main()
