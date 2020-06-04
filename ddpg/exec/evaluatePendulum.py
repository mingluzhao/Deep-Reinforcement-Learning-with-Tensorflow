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

from src.ddpg import actByPolicyTrain, actByPolicyTarget, evaluateCriticTarget, getActionGradients, \
    BuildActorModel, BuildCriticModel, TrainCriticBySASRQ, TrainCritic, TrainActorFromGradients, TrainActorOneStep, \
    TrainActor, TrainDDPGModels
from RLframework.RLrun import resetTargetParamToTrainParam, UpdateParameters, SampleOneStep, SampleFromMemory,\
    LearnFromBuffer, RunTimeStep, RunEpisode, RunAlgorithm
from src.policy import ActDDPGOneStep
from functionTools.loadSaveModel import GetSavePath, saveVariables

from environment.noise.noise import GetExponentialDecayGaussNoise
from environment.gymEnv.pendulumEnv import TransitGymPendulum, RewardGymPendulum, isTerminalGymPendulum, \
    observe, angle_normalize, ResetGymPendulum

ENV_NAME = 'Pendulum-v0'
env = gym.make(ENV_NAME)
env = env.unwrapped
seed = 1

class EvaluateNoiseAndMemorySize:
    def __init__(self, fixedParameters, getSavePath, saveModel = True):
        self.fixedParameters = fixedParameters
        self.getSavePath = getSavePath
        self.saveModel = saveModel

    def __call__(self, df):
        noiseVariance = df.index.get_level_values('noiseInitVariance')[0]
        memorySize = df.index.get_level_values('memorySize')[0]

        buildActorModel = BuildActorModel(self.fixedParameters['stateDim'], self.fixedParameters['actionDim'],
                                          self.fixedParameters['actionBound'])
        actorWriter, actorModel = buildActorModel(self.fixedParameters['actorLayerWidths'])

        buildCriticModel = BuildCriticModel(self.fixedParameters['stateDim'], self.fixedParameters['actionDim'])
        criticWriter, criticModel = buildCriticModel(self.fixedParameters['criticLayerWidths'])

        trainCriticBySASRQ = TrainCriticBySASRQ(self.fixedParameters['learningRateCritic'], self.fixedParameters['gamma'], criticWriter)
        trainCritic = TrainCritic(actByPolicyTarget, evaluateCriticTarget, trainCriticBySASRQ)

        trainActorFromGradients = TrainActorFromGradients(self.fixedParameters['learningRateActor'], actorWriter)
        trainActorOneStep = TrainActorOneStep(actByPolicyTrain, trainActorFromGradients, getActionGradients)
        trainActor = TrainActor(trainActorOneStep)

        updateParameters = UpdateParameters(self.fixedParameters['paramUpdateInterval'], self.fixedParameters['tau'])

        modelList = [actorModel, criticModel]
        actorModel, criticModel = resetTargetParamToTrainParam(modelList)
        trainModels = TrainDDPGModels(updateParameters, trainActor, trainCritic, actorModel, criticModel)

        getNoise = GetExponentialDecayGaussNoise(noiseVariance, self.fixedParameters['varianceDiscount'], self.fixedParameters['noiseDecayStartStep'])
        actOneStepWithNoise = ActDDPGOneStep(self.fixedParameters['actionLow'], self.fixedParameters['actionHigh'],
                                             actByPolicyTrain, actorModel, getNoise)

        sampleFromMemory = SampleFromMemory(self.fixedParameters['batchSize'])
        learnFromBuffer = LearnFromBuffer(self.fixedParameters['learningStartStep'], sampleFromMemory, trainModels)

        transit = TransitGymPendulum()
        getReward = RewardGymPendulum(angle_normalize)
        sampleOneStep = SampleOneStep(transit, getReward)

        runDDPGTimeStep = RunTimeStep(actOneStepWithNoise, sampleOneStep, learnFromBuffer, observe)

        reset = ResetGymPendulum(seed)
        runEpisode = RunEpisode(reset, runDDPGTimeStep, self.fixedParameters['maxRunSteps'], isTerminalGymPendulum)

        ddpg = RunAlgorithm(runEpisode, self.fixedParameters['maxEpisode'])

        replayBuffer = deque(maxlen=int(memorySize))
        meanRewardList, trajectory = ddpg(replayBuffer)

        trainedActorModel, trainedCriticModel = trainModels.getTrainedModels()

        timeStep = list(range(len(meanRewardList)))
        resultSe = pd.Series({time: reward for time, reward in zip(timeStep, meanRewardList)})

        if self.saveModel:
            actorParameters = {'ActorMemorySize': memorySize, 'NoiseVariance': noiseVariance }
            criticParameters ={'CriticMemorySize': memorySize, 'NoiseVariance': noiseVariance }
            actorPath = self.getSavePath(actorParameters)
            criticPath = self.getSavePath(criticParameters)
            with trainedActorModel.as_default():
                saveVariables(trainedActorModel, actorPath)
            with trainedCriticModel.as_default():
                saveVariables(trainedCriticModel, criticPath)

        return resultSe


def main():
    stateDim = env.observation_space.shape[0]
    actionDim = env.action_space.shape[0]
    actionHigh = env.action_space.high
    actionLow = env.action_space.low
    actionBound = (actionHigh - actionLow)/2
    
    fixedParameters = OrderedDict()
    fixedParameters['batchSize'] = 128
    fixedParameters['learningRateActor'] = 0.001
    fixedParameters['learningRateCritic'] = 0.001
    fixedParameters['maxEpisode'] = 200
    fixedParameters['maxRunSteps'] = 200
    fixedParameters['tau'] = 0.01
    fixedParameters['gamma'] = 0.9
    fixedParameters['actorLayerWidths'] = [30]
    fixedParameters['criticLayerWidths'] = [30]
    fixedParameters['stateDim'] = stateDim
    fixedParameters['actionDim'] = actionDim
    fixedParameters['actionBound'] = actionBound
    fixedParameters['actionHigh'] = actionHigh
    fixedParameters['actionLow'] = actionLow
    fixedParameters['paramUpdateInterval'] = 1
    fixedParameters['varianceDiscount'] = .9995
    fixedParameters['noiseDecayStartStep'] = 10000
    fixedParameters['learningStartStep'] = fixedParameters['batchSize']

    independentVariables = OrderedDict()
    independentVariables['noiseInitVariance'] = [1, 2, 3, 4]
    independentVariables['memorySize'] = [1000, 5000, 10000, 20000]

    modelSaveDirectory = "../trainedDDPGModels"
    modelSaveExtension = '.ckpt'
    getSavePath = GetSavePath(modelSaveDirectory, modelSaveExtension)

    evaluate = EvaluateNoiseAndMemorySize(fixedParameters, getSavePath, saveModel= False)


    levelNames = list(independentVariables.keys())
    levelValues = list(independentVariables.values())
    levelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=levelIndex)
    resultDF = toSplitFrame.groupby(levelNames).apply(evaluate)


    nCols = len(independentVariables['noiseInitVariance'])
    nRows = len(independentVariables['memorySize'])
    numplot = 1
    axs = []
    figure = plt.figure(figsize=(12, 10))
    for keyCol, outterSubDf in resultDF.groupby('memorySize'):
        for keyRow, innerSubDf in outterSubDf.groupby('noiseInitVariance'):
            subplot = figure.add_subplot(nRows, nCols, numplot)
            axs.append(subplot)
            numplot += 1
            plt.ylim([-1600, 50])
            innerSubDf.T.plot(ax=subplot)

            subplot.set_ylabel('MeanEpsReward', fontsize=5)
            subplot.set_xlabel('Episode', fontsize=5)
            subplot.tick_params(axis='both', which='major', labelsize=5)
            subplot.tick_params(axis='both', which='minor', labelsize=5)

    dirName = os.path.dirname(__file__)
    plotPath = os.path.join(dirName, '..', 'plots')
    plt.savefig(os.path.join(plotPath, 'pendulumEvaluation'))
    plt.show()


if __name__ == "__main__":
    main()
