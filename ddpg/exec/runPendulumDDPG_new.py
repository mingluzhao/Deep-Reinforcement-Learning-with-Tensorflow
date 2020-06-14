import matplotlib.pyplot as plt
import gym
from collections import deque
import os
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from src.newddpg import actByPolicyTrain, BuildDDPGModels, TrainCritic, TrainActor, TrainDDPGModels, TrainCriticBySASR, \
    TrainActorFromState, reshapeBatchToGetSASR
from RLframework.RLrun_MultiAgent import resetTargetParamToTrainParam, UpdateParameters, SampleOneStep, SampleFromMemory,\
    LearnFromBuffer, RunTimeStep, RunEpisode, RunAlgorithm, SaveModel
from src.policy import ActDDPGOneStep
from functionTools.loadSaveModel import GetSavePath, saveVariables, saveToPickle

from environment.noise.noise import GetExponentialDecayGaussNoise
from environment.gymEnv.pendulumEnv import TransitGymPendulum, RewardGymPendulum, isTerminalGymPendulum, \
    observe, angle_normalize, VisualizeGymPendulum, ResetGymPendulum
import tensorflow as tf

maxEpisode = 200
maxTimeStep = 200
learningRateActor = 0.001
learningRateCritic = 0.001
gamma = 0.9
tau=0.01
bufferSize = 10000
minibatchSize = 128
seed = 1

ENV_NAME = 'Pendulum-v0'
env = gym.make(ENV_NAME)
env = env.unwrapped

def main():
    stateDim = env.observation_space.shape[0]
    actionDim = env.action_space.shape[0]
    actionHigh = env.action_space.high
    actionLow = env.action_space.low
    actionBound = (actionHigh - actionLow)/2

    actorWeightInit = tf.random_uniform_initializer(0, 0.03)
    actorBiasInit = tf.constant_initializer(0.01)
    criticWeightInit = tf.random_uniform_initializer(0, 0.01)
    cirticBiasInit = tf.constant_initializer(0.01)

    weightInitializerList = [actorWeightInit, actorBiasInit, criticWeightInit, cirticBiasInit]
    buildModel = BuildDDPGModels(stateDim, actionDim, weightInitializerList, actionBound)
    layerWidths = [30]
    writer, model = buildModel(layerWidths)

    trainCriticBySASR = TrainCriticBySASR(learningRateCritic, gamma, writer)
    trainCritic = TrainCritic(reshapeBatchToGetSASR, trainCriticBySASR)

    trainActorFromState = TrainActorFromState(learningRateActor, writer)
    trainActor = TrainActor(reshapeBatchToGetSASR, trainActorFromState)

    paramUpdateInterval = 1 #
    updateParameters = UpdateParameters(paramUpdateInterval, tau)

    trainModels = TrainDDPGModels(updateParameters, trainActor, trainCritic, model)

    noiseInitVariance = 3
    varianceDiscount = .9995
    noiseDecayStartStep = bufferSize
    getNoise = GetExponentialDecayGaussNoise(noiseInitVariance, varianceDiscount, noiseDecayStartStep)
    actOneStepWithNoise = ActDDPGOneStep(actionLow, actionHigh, actByPolicyTrain, model, getNoise)

    learningStartBufferSize = minibatchSize
    sampleFromMemory = SampleFromMemory(minibatchSize)
    learnFromBuffer = LearnFromBuffer(learningStartBufferSize, sampleFromMemory, trainModels)

    transit = TransitGymPendulum()
    getReward = RewardGymPendulum(angle_normalize)
    sampleOneStep = SampleOneStep(transit, getReward)

    runDDPGTimeStep = RunTimeStep(actOneStepWithNoise, sampleOneStep, learnFromBuffer, observe)

    reset = ResetGymPendulum(seed)
    runEpisode = RunEpisode(reset, runDDPGTimeStep, maxTimeStep, isTerminalGymPendulum)

    dirName = os.path.dirname(__file__)
    modelPath = os.path.join(dirName, '..', 'trainedDDPGModels', 'pendulum_newddpg')
    getTrainedModel = lambda: trainModels.getTrainedModels()
    modelSaveRate = 50
    saveModel = SaveModel(modelSaveRate, saveVariables, getTrainedModel, modelPath)

    ddpg = RunAlgorithm(runEpisode, maxEpisode, [saveModel])

    replayBuffer = deque(maxlen=int(bufferSize))
    meanRewardList, trajectory = ddpg(replayBuffer)

# plots& plot
    showDemo = False
    if showDemo:
        visualize = VisualizeGymPendulum()
        visualize(trajectory)

    plotResult = True
    if plotResult:
        plt.plot(list(range(maxEpisode)), meanRewardList)
        plt.show()


if __name__ == '__main__':
    main()

