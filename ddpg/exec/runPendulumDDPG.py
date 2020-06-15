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
from RLframework.RLrun_MultiAgent import resetTargetParamToTrainParam, UpdateParameters, SampleOneStep, SampleFromMemory,\
    LearnFromBuffer, RunTimeStep, RunEpisode, RunAlgorithm, SaveModel
from src.policy import ActDDPGOneStep
from functionTools.loadSaveModel import saveVariables, saveToPickle

from environment.noise.noise import GetExponentialDecayGaussNoise
from environment.gymEnv.pendulumEnv import TransitGymPendulum, RewardGymPendulum, isTerminalGymPendulum, \
    observe, angle_normalize, VisualizeGymPendulum, ResetGymPendulum

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

    noiseInitVariance = 3
    varianceDiscount = .9995
    noiseDecayStartStep = bufferSize
    getNoise = GetExponentialDecayGaussNoise(noiseInitVariance, varianceDiscount, noiseDecayStartStep)
    actOneStepWithNoise = ActDDPGOneStep(actionLow, actionHigh, actByPolicyTrain, actorModel, getNoise)

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
    modelPath = os.path.join(dirName, '..', 'trainedDDPGModels', 'pendulum')
    getTrainedModel = lambda: trainModels.actorModel
    modelSaveRate = 50
    saveModel = SaveModel(modelSaveRate, saveVariables, getTrainedModel, modelPath)

    ddpg = RunAlgorithm(runEpisode, maxEpisode, [saveModel])

    replayBuffer = deque(maxlen=int(bufferSize))
    meanRewardList, trajectory = ddpg(replayBuffer)

    dirName = os.path.dirname(__file__)
    trajectoryPath = os.path.join(dirName, '..', 'trajectory', 'pendulumTrajectory1.pickle')
    saveToPickle(trajectory, trajectoryPath)

# plots& plot
    showDemo = True
    if showDemo:
        visualize = VisualizeGymPendulum()
        visualize(trajectory)

    plotResult = True
    if plotResult:
        plt.plot(list(range(maxEpisode)), meanRewardList)
        plt.show()


if __name__ == '__main__':
    main()

