import matplotlib.pyplot as plt
import gym
from collections import deque
import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))
sys.path.append(os.path.join(dirName, '..'))

from dqn.src.dqn import *
from dqn.src.policy import *
from RLframework.RLrun import resetTargetParamToTrainParam, UpdateParameters, SampleOneStep, SampleFromMemory,\
    LearnFromBuffer, RunTimeStep, RunEpisode, RunAlgorithm
from environment.gymEnv.pendulumEnv import *
from functionTools.loadSaveModel import saveToPickle, GetSavePath, saveVariables
os.environ['KMP_DUPLICATE_LIB_OK']='True'

seed = 1
env = gym.make('Pendulum-v0')
env = env.unwrapped

def main():
    stateDim = env.observation_space.shape[0]
    actionDim = 7
    buildModel = BuildModel(stateDim, actionDim)
    layersWidths = [30]
    writer, model = buildModel(layersWidths)

    learningRate = 0.001
    gamma = 0.99
    trainModelBySASRQ = TrainModelBySASRQ(learningRate, gamma, writer)

    paramUpdateInterval = 300
    updateParameters = UpdateParameters(paramUpdateInterval)
    model = resetTargetParamToTrainParam([model])[0]
    trainModels = TrainDQNModel(getTargetQValue, trainModelBySASRQ, updateParameters, model)

    epsilonMax = 0.9
    epsilonIncrease = 0.0001
    epsilonMin = 0
    bufferSize = 10000
    decayStartStep = bufferSize
    getEpsilon = GetEpsilon(epsilonMax, epsilonMin, epsilonIncrease, decayStartStep)

    actGreedyByModel = ActGreedyByModel(getTrainQValue, model)
    actRandom = ActRandom(actionDim)
    actByTrainNetEpsilonGreedy = ActByTrainNetEpsilonGreedy(getEpsilon, actGreedyByModel, actRandom)

    minibatchSize = 128
    learningStartBufferSize = minibatchSize
    sampleFromMemory = SampleFromMemory(minibatchSize)
    learnFromBuffer = LearnFromBuffer(learningStartBufferSize, sampleFromMemory, trainModels)

    processAction = ProcessDiscretePendulumAction(actionDim)
    transit = TransitGymPendulum(processAction)
    getReward = RewardGymPendulum(angle_normalize, processAction)
    sampleOneStep = SampleOneStep(transit, getReward)

    runDQNTimeStep = RunTimeStep(actByTrainNetEpsilonGreedy, sampleOneStep, learnFromBuffer, observe)

    reset = ResetGymPendulum(seed)
    maxTimeStep = 200
    runEpisode = RunEpisode(reset, runDQNTimeStep, maxTimeStep, isTerminalGymPendulum)

    maxEpisode = 400
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
