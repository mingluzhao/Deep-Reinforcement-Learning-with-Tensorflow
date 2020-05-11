import matplotlib.pyplot as plt
import gym
from collections import deque
import os
import sys
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
from functionTools.loadSaveModel import GetSavePath, saveVariables, saveToPickle

from environment.noise.noise import GetExponentialDecayGaussNoise
from environment.gymEnv.continousMountainCarEnv import IsTerminalMountCarContin, TransitGymMountCarContinuous, \
    RewardMountCarContin, ResetMountCarContin, VisualizeMountCarContin


maxEpisode = 300
maxTimeStep = 2000
learningRateActor = 0.001
learningRateCritic = 0.001
gamma = 0.9
tau=0.01
bufferSize = 200000
minibatchSize = 128
seed = 1

ENV_NAME = 'MountainCarContinuous-v0'
env = gym.make(ENV_NAME)
env = env.unwrapped

def main():
    stateDim = env.observation_space.shape[0]
    actionDim = env.action_space.shape[0]
    actionHigh = env.action_space.high
    actionLow = env.action_space.low
    actionBound = (actionHigh - actionLow)/2

    buildActorModel = BuildActorModel(stateDim, actionDim, actionBound)
    actorLayerWidths = [30]
    actorWriter, actorModel = buildActorModel(actorLayerWidths)

    buildCriticModel = BuildCriticModel(stateDim, actionDim)
    criticLayerWidths = [30]
    criticWriter, criticModel = buildCriticModel(criticLayerWidths)

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

    noiseInitVariance = 1  # control exploration
    varianceDiscount = .99995
    noiseDecayStartStep = bufferSize
    minVar = .1
    getNoise = GetExponentialDecayGaussNoise(noiseInitVariance, varianceDiscount, noiseDecayStartStep, minVar)
    actOneStepWithNoise = ActDDPGOneStep(actionLow, actionHigh, actByPolicyTrain, actorModel, getNoise)

    learningStartBufferSize = minibatchSize
    sampleFromMemory = SampleFromMemory(minibatchSize)
    learnFromBuffer = LearnFromBuffer(learningStartBufferSize, sampleFromMemory, trainModels)

    transit = TransitGymMountCarContinuous()
    isTerminal = IsTerminalMountCarContin()
    getReward = RewardMountCarContin(isTerminal)
    sampleOneStep = SampleOneStep(transit, getReward)

    runDDPGTimeStep = RunTimeStep(actOneStepWithNoise, sampleOneStep, learnFromBuffer)

    resetLow = -1
    resetHigh = 0.4
    reset = ResetMountCarContin(seed = None)
    runEpisode = RunEpisode(reset, runDDPGTimeStep, maxTimeStep, isTerminal)

    ddpg = RunAlgorithm(runEpisode, maxEpisode)
    replayBuffer = deque(maxlen=int(bufferSize))
    meanRewardList, trajectory = ddpg(replayBuffer)

    trainedActorModel, trainedCriticModel = trainModels.getTrainedModels()

# save Model
    modelIndex = 0
    actorFixedParam = {'actorModel': modelIndex}
    criticFixedParam = {'criticModel': modelIndex}
    parameters = {'env': ENV_NAME, 'Eps': maxEpisode, 'timeStep': maxTimeStep, 'batch': minibatchSize,
                  'gam': gamma, 'lrActor': learningRateActor, 'lrCritic': learningRateCritic,
                  'noiseVar': noiseInitVariance, 'varDiscout': varianceDiscount, 'resetLow': resetLow, 'High': resetHigh}

    modelSaveDirectory = "../trainedDDPGModels"
    modelSaveExtension = '.ckpt'
    getActorSavePath = GetSavePath(modelSaveDirectory, modelSaveExtension, actorFixedParam)
    getCriticSavePath = GetSavePath(modelSaveDirectory, modelSaveExtension, criticFixedParam)
    savePathActor = getActorSavePath(parameters)
    savePathCritic = getCriticSavePath(parameters)

    with actorModel.as_default():
        saveVariables(trainedActorModel, savePathActor)
    with criticModel.as_default():
        saveVariables(trainedCriticModel, savePathCritic)

    dirName = os.path.dirname(__file__)
    trajectoryPath = os.path.join(dirName, '..', 'trajectory', 'mountCarTrajectoryOriginalReset1.pickle')
    saveToPickle(trajectory, trajectoryPath)

# plots& plot
    showDemo = False
    if showDemo:
        visualize = VisualizeMountCarContin()
        visualize(trajectory)

    plotResult = True
    if plotResult:
        plt.plot(list(range(maxEpisode)), meanRewardList)
        plt.show()


if __name__ == '__main__':
    main()



