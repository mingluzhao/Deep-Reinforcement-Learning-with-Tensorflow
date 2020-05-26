import os
import pandas as pd
import pylab as plt
import gym
import sys
from collections import deque, OrderedDict
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))
sys.path.append(os.path.join(dirName, '..'))

from dqn.src.dqn import *
from dqn.src.policy import *
from RLframework.RLrun import resetTargetParamToTrainParam, UpdateParameters, SampleOneStep, SampleFromMemory,\
    LearnFromBuffer, RunTimeStep, RunEpisode, RunAlgorithm
from functionTools.loadSaveModel import GetSavePath, saveVariables, saveToPickle
from environment.gymEnv.pendulumEnv import TransitGymPendulum, RewardGymPendulum, isTerminalGymPendulum, \
    observe, angle_normalize, ResetGymPendulum, ProcessDiscretePendulumAction

ENV_NAME = 'Pendulum-v0'
env = gym.make(ENV_NAME)
env = env.unwrapped
seed = 1

stateDim = env.observation_space.shape[0]
actionDim = env.action_space.shape[0]
actionHigh = env.action_space.high
actionLow = env.action_space.low
actionBound = (actionHigh - actionLow) / 2


class EvaluateNoiseAndMemorySize:
    def __init__(self, getSavePath, saveModel = True):
        self.getSavePath = getSavePath
        self.saveModel = saveModel

    def __call__(self, df):
        actionDim = df.index.get_level_values('actionDim')[0]
        epsilonIncrease = df.index.get_level_values('epsilonIncrease')[0]

        stateDim = env.observation_space.shape[0]
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

        timeStep = list(range(len(meanRewardList)))
        resultSe = pd.Series({time: reward for time, reward in zip(timeStep, meanRewardList)})

        return resultSe


def main():
    independentVariables = OrderedDict()
    independentVariables['actionDim'] = [3, 7, 11, 15]
    independentVariables['epsilonIncrease'] = [1e-5, 1e-4, 2e-4, 1e-3]

    # independentVariables['actionDim'] = [3, 15]
    # independentVariables['epsilonIncrease'] = [1e-5, 1e-3]

    modelSaveDirectory = "../trainedDQNModels"
    modelSaveExtension = '.ckpt'
    getSavePath = GetSavePath(modelSaveDirectory, modelSaveExtension)

    evaluate = EvaluateNoiseAndMemorySize(getSavePath, saveModel= False)

    levelNames = list(independentVariables.keys())
    levelValues = list(independentVariables.values())
    levelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=levelIndex)
    resultDF = toSplitFrame.groupby(levelNames).apply(evaluate)


    nCols = len(independentVariables['actionDim'])
    nRows = len(independentVariables['epsilonIncrease'])
    numplot = 1
    axs = []
    figure = plt.figure(figsize=(12, 10))
    for keyCol, outterSubDf in resultDF.groupby('epsilonIncrease'):
        for keyRow, innerSubDf in outterSubDf.groupby('actionDim'):
            subplot = figure.add_subplot(nRows, nCols, numplot)
            axs.append(subplot)
            numplot += 1
            plt.ylim([-1600, 50])
            innerSubDf.T.plot(ax=subplot)

    dirName = os.path.dirname(__file__)
    plotPath = os.path.join(dirName, '..', 'plots')
    plt.savefig(os.path.join(plotPath, 'pendulumEvaluation'))
    plt.show()


if __name__ == "__main__":
    main()
