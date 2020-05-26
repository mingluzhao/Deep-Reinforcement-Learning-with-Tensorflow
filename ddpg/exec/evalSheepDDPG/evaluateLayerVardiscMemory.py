import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))
sys.path.append(os.path.join(dirName, '..'))

from collections import deque, OrderedDict
import matplotlib.pyplot as plt
import pandas as pd

from src.ddpg import actByPolicyTrain, actByPolicyTarget, evaluateCriticTarget, getActionGradients, \
    BuildActorModel, BuildCriticModel, TrainCriticBySASRQ, TrainCritic, TrainActorFromGradients, TrainActorOneStep, \
    TrainActor, TrainDDPGModels
from RLframework.RLrun import resetTargetParamToTrainParam, UpdateParameters, SampleOneStep, SampleFromMemory,\
    LearnFromBuffer, RunTimeStep, RunEpisode, RunAlgorithm
from src.policy import ActDDPGOneStep
from functionTools.loadSaveModel import GetSavePath, saveToPickle
from environment.noise.noise import GetExponentialDecayGaussNoise
from environment.chasingEnv.reward import RewardSheepWithBoundaryHeuristics, GetBoundaryPunishment, RewardFunctionCompete
from environment.chasingEnv.chasingPolicy import HeatSeekingContinuousDeterministicPolicy
from environment.chasingEnv.envNoPhysics import Reset, TransitForNoPhysics, getIntendedNextState, StayWithinBoundary, \
    TransitWithSingleWolf, GetAgentPosFromState, IsTerminal


learningRateCritic = 0.01
gamma = 0.95
tau = 0.01
learningRateActor = 0.01
minibatchSize = 32
maxTimeStep = 100  #
maxEpisode = 20

numAgents = 2
stateDim = numAgents * 2
actionLow = -1
actionHigh = 1
actionBound = (actionHigh - actionLow) / 2
actionDim = 2


class EvaluateNoiseAndMemorySize:
    def __init__(self, getSavePath, saveModel=True):
        self.getSavePath = getSavePath
        self.saveModel = saveModel

    def __call__(self, df):
        varianceDiscount = df.index.get_level_values('varianceDiscount')[0]
        bufferSize = df.index.get_level_values('bufferSize')[0]
        layerWidth = df.index.get_level_values('layerWidth')[0]
        print('buffer: ', bufferSize, ', layers: ', layerWidth, ', varDiscount: ', varianceDiscount)

        buildActorModel = BuildActorModel(stateDim, actionDim, actionBound)
        actorWriter, actorModel = buildActorModel(layerWidth)

        buildCriticModel = BuildCriticModel(stateDim, actionDim)
        criticWriter, criticModel = buildCriticModel(layerWidth)

        trainCriticBySASRQ = TrainCriticBySASRQ(learningRateCritic, gamma, criticWriter)
        trainCritic = TrainCritic(actByPolicyTarget, evaluateCriticTarget, trainCriticBySASRQ)

        trainActorFromGradients = TrainActorFromGradients(learningRateActor, actorWriter)
        trainActorOneStep = TrainActorOneStep(actByPolicyTrain, trainActorFromGradients, getActionGradients)
        trainActor = TrainActor(trainActorOneStep)

        paramUpdateInterval = 1
        updateParameters = UpdateParameters(paramUpdateInterval, tau)

        modelList = [actorModel, criticModel]
        actorModel, criticModel = resetTargetParamToTrainParam(modelList)
        trainModels = TrainDDPGModels(updateParameters, trainActor, trainCritic, actorModel, criticModel)

        noiseInitVariance = 1
        noiseDecayStartStep = bufferSize
        getNoise = GetExponentialDecayGaussNoise(noiseInitVariance, varianceDiscount, noiseDecayStartStep)
        actOneStepWithNoise = ActDDPGOneStep(actionLow, actionHigh, actByPolicyTrain, actorModel, getNoise)

        learningStartBufferSize = minibatchSize
        sampleFromMemory = SampleFromMemory(minibatchSize)
        learnFromBuffer = LearnFromBuffer(learningStartBufferSize, sampleFromMemory, trainModels)

        sheepId = 0
        wolfId = 1
        getSheepXPos = GetAgentPosFromState(sheepId)
        getWolfXPos = GetAgentPosFromState(wolfId)

        wolfSpeed = 2
        wolfPolicy = HeatSeekingContinuousDeterministicPolicy(getWolfXPos, getSheepXPos, wolfSpeed)
        xBoundary = (0, 20)
        yBoundary = (0, 20)
        stayWithinBoundary = StayWithinBoundary(xBoundary, yBoundary)
        physicalTransition = TransitForNoPhysics(getIntendedNextState, stayWithinBoundary)
        transit = TransitWithSingleWolf(physicalTransition, wolfPolicy)

        sheepAliveBonus = 0 / maxTimeStep
        sheepTerminalPenalty = -20

        killzoneRadius = 1
        isTerminal = IsTerminal(getWolfXPos, getSheepXPos, killzoneRadius)
        getBoundaryPunishment = GetBoundaryPunishment(xBoundary, yBoundary, sheepIndex=0, punishmentVal=10)
        rewardSheep = RewardFunctionCompete(sheepAliveBonus, sheepTerminalPenalty, isTerminal)
        getReward = RewardSheepWithBoundaryHeuristics(rewardSheep, getIntendedNextState, getBoundaryPunishment,
                                                      getSheepXPos)
        sampleOneStep = SampleOneStep(transit, getReward)

        runDDPGTimeStep = RunTimeStep(actOneStepWithNoise, sampleOneStep, learnFromBuffer)

        reset = Reset(xBoundary, yBoundary, numAgents)
        runEpisode = RunEpisode(reset, runDDPGTimeStep, maxTimeStep, isTerminal)

        ddpg = RunAlgorithm(runEpisode, maxEpisode)

        replayBuffer = deque(maxlen=int(bufferSize))
        meanRewardList, trajectory = ddpg(replayBuffer)

        timeStep = list(range(len(meanRewardList)))
        resultSe = pd.Series({time: reward for time, reward in zip(timeStep, meanRewardList)})

        return resultSe


def main():
    independentVariables = OrderedDict()
    # independentVariables['bufferSize'] = [1000, 5000, 10000, 20000]
    # independentVariables['layerWidth'] = [[32], [32, 32], [64, 64, 64], [128, 128, 128, 128]]
    # independentVariables['varianceDiscount'] = [0.995, 0.9995, 0.99995]

    independentVariables['bufferSize'] = [1000, 5000]
    independentVariables['layerWidth'] = [(32, 32), (32, 32, 32)]
    independentVariables['varianceDiscount'] = [0.995, 0.9995]

    modelSaveDirectory = "../trainedDDPGModels"
    modelSaveExtension = '.ckpt'
    getSavePath = GetSavePath(modelSaveDirectory, modelSaveExtension)

    evaluate = EvaluateNoiseAndMemorySize(getSavePath, saveModel=False)

    levelNames = list(independentVariables.keys())
    levelValues = list(independentVariables.values())
    levelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=levelIndex)
    resultDF = toSplitFrame.groupby(levelNames).apply(evaluate)
    resultPath = os.path.join(dirName, '..', '..', 'plots', str(list(independentVariables.keys())) + '.pickle')
    saveToPickle(resultDF, resultPath)

    nCols = len(independentVariables['bufferSize'])
    nRows = len(independentVariables['layerWidth'])
    numplot = 1
    axs = []
    figure = plt.figure(dpi= 200)
    figure.set_size_inches(7.5, 6)
    figure.suptitle('buffer: ' + str(independentVariables['bufferSize'])+
                    ', layers: '+ str(independentVariables['layerWidth']) +
                    ', varDiscount: '+ str(independentVariables['varianceDiscount']), fontsize = 8)

    for layerWidth, outterSubDf in resultDF.groupby('layerWidth'):
        for bufferSize, innerSubDf in outterSubDf.groupby('bufferSize'):
            subplot = figure.add_subplot(nRows, nCols, numplot)
            axs.append(subplot)
            numplot += 1
            plt.ylim([-800, 400])
            innerSubDf.T.plot(ax=subplot)

            subplot.set_title('layerWidth = ' + str(layerWidth) + ' bufferSize = ' + str(bufferSize), fontsize=5)
            subplot.set_ylabel('MeanEpsReward', fontsize=5)
            subplot.set_xlabel('Episode', fontsize=5)
            subplot.tick_params(axis='both', which='major', labelsize=5)
            subplot.tick_params(axis='both', which='minor', labelsize=5)
            plt.legend(loc='best', prop={'size': 5})


    plotPath = os.path.join(dirName, '..', '..', 'plots')
    plt.savefig(os.path.join(plotPath, str(list(independentVariables.keys()))))
    plt.show()


if __name__ == '__main__':
    main()

