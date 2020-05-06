import pandas as pd
import pylab as plt
from collections import OrderedDict
from src.ddpg import *
from RLframework.RLrun import *
from src.policy import ActDDPGOneStepWithNoise
from functionTools.loadSaveModel import *
import os
from environment.noise.noise import GetExponentialDecayGaussNoise
from environment.gymEnv.pendulumEnv import *

os.environ['KMP_DUPLICATE_LIB_OK']='True'

ENV_NAME = 'Pendulum-v0'
env = gym.make(ENV_NAME)
env = env.unwrapped
seed = 1

class EvaluateNoiseAndMemorySize:
    def __init__(self, actorModel, criticModel, getNoiseWithDiffVar, actInGymWithDiffNoise,
                 runDDPGTimeStepWithDiffNoise, runEpisodeWithDiffNoise, ddpgWithDiffNoiseAndDiffMemorySize,
                 getSavePath, saveModel = True):
        self.modelList = [actorModel, criticModel]
        self.getNoiseWithDiffVar = getNoiseWithDiffVar
        self.actInGymWithDiffNoise = actInGymWithDiffNoise
        self.runDDPGTimeStepWithDiffNoise = runDDPGTimeStepWithDiffNoise
        self.runEpisodeWithDiffNoise = runEpisodeWithDiffNoise
        self.ddpgWithDiffNoiseAndDiffMemorySize = ddpgWithDiffNoiseAndDiffMemorySize
        self.getSavePath = getSavePath
        self.saveModel = saveModel

    def __call__(self, df):
        noiseVariance = df.index.get_level_values('noiseInitVariance')[0]
        memorySize = df.index.get_level_values('memorySize')[0]

        getNoise = self.getNoiseWithDiffVar(noiseVariance)
        actInGymWithNoise = self.actInGymWithDiffNoise(getNoise)
        runDDPGTimeStep = self.runDDPGTimeStepWithDiffNoise(actInGymWithNoise)
        runDDPGEpisode = self.runEpisodeWithDiffNoise(runDDPGTimeStep)
        ddpg = self.ddpgWithDiffNoiseAndDiffMemorySize(memorySize, runDDPGEpisode)

        meanRewardList, trajectory, modelList = ddpg(self.modelList)
        trainedActorModel, trainedCriticModel = modelList

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
    fixedParameters['learningRateActor'] = 1e-4
    fixedParameters['learningRateCritic'] = 1e-4
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

    buildActorModel = BuildActorModel(fixedParameters['stateDim'], fixedParameters['actionDim'], fixedParameters['actionBound'])
    actorWriter, actorModel = buildActorModel(fixedParameters['actorLayerWidths'])

    buildCriticModel = BuildCriticModel(fixedParameters['stateDim'], fixedParameters['actionDim'])
    criticWriter, criticModel = buildCriticModel(fixedParameters['criticLayerWidths'])

    trainCriticBySASRQ = TrainCriticBySASRQ(fixedParameters['learningRateCritic'], fixedParameters['gamma'], criticWriter)
    trainCritic = TrainCritic(actByPolicyTarget, evaluateCriticTarget, trainCriticBySASRQ)

    trainActorFromGradients = TrainActorFromGradients(fixedParameters['learningRateActor'], actorWriter)
    trainActorOneStep = TrainActorOneStep(actByPolicyTrain, trainActorFromGradients, getActionGradients)
    trainActor = TrainActor(trainActorOneStep)

    updateParameters = UpdateParameters(fixedParameters['paramUpdateInterval'], fixedParameters['tau'])
    trainModels = TrainDDPGModels(updateParameters, trainActor, trainCritic)


    getNoiseWithDiffVar = lambda var: GetExponentialDecayGaussNoise(var, fixedParameters['varianceDiscount'], fixedParameters['noiseDecayStartStep'])
    actOneStepWithDiffNoise = lambda getNoiseWithDiffVar: ActDDPGOneStepWithNoise(fixedParameters['actionLow'], fixedParameters['actionHigh'], actByPolicyTrain, getNoiseWithDiffVar)

    transit = TransitGymPendulum()
    getReward = RewardGymPendulum(angle_normalize)
    runDDPGTimeStepWithDiffNoise = lambda actWithNoise: RunTimeStep(actWithNoise, transit, getReward, isTerminalGymPendulum, addToMemory,
                 trainModels, fixedParameters['batchSize'], fixedParameters['learningStartStep'], observe)

    reset = ResetGymPendulum(seed, observe)
    runEpisodeWithDiffNoise = lambda runDDPGTimeStepWithDiffNoise: RunEpisode(reset, runDDPGTimeStepWithDiffNoise, fixedParameters['maxRunSteps'])

    ddpgWithDiffNoiseAndDiffMemorySize = lambda memorySize, runEpisodeWithDiffNoise: RunAlgorithm(runEpisodeWithDiffNoise, memorySize, fixedParameters['maxEpisode'])

    modelSaveDirectory = "../trainedDDPGModels"
    modelSaveExtension = '.ckpt'
    getSavePath = GetSavePath(modelSaveDirectory, modelSaveExtension)

    modelList = [actorModel, criticModel]
    actorModel, criticModel = resetTargetParamToTrainParam(modelList)

    evaluate = EvaluateNoiseAndMemorySize(actorModel, criticModel, getNoiseWithDiffVar, actOneStepWithDiffNoise,
                 runDDPGTimeStepWithDiffNoise, runEpisodeWithDiffNoise, ddpgWithDiffNoiseAndDiffMemorySize,
                 getSavePath, saveModel=True)

    env.close()


    ####################################################################################################################################

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

    dirName = os.path.dirname(__file__)
    plotPath = os.path.join(dirName, '..', 'demo')
    plt.savefig(os.path.join(plotPath, 'pendulumEvaluation'))
    plt.show()


if __name__ == "__main__":
    main()
