import matplotlib.pyplot as plt
import gym
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from src.ddpg import actByPolicyTrain, actByPolicyTarget, evaluateCriticTarget, getActionGradients, \
    BuildActorModel, BuildCriticModel, TrainCriticBySASRQ, TrainCritic, TrainActorFromGradients,\
    TrainActorOneStep, TrainActor, TrainDDPGModels
from RLframework.RLrun import resetTargetParamToTrainParam, addToMemory, UpdateParameters, RunTimeStep, \
    RunEpisode, RunAlgorithm
from src.policy import ActDDPGOneStepWithNoise
from environment.noise.noise import GetExponentialDecayGaussNoise
from environment.gymEnv.pendulumEnv import TransitGymPendulum, RewardGymPendulum, isTerminalGymPendulum, \
    observe, angle_normalize, VisualizeGymPendulum, ResetGymPendulum
from functionTools.loadSaveModel import GetSavePath, saveVariables, saveToPickle

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

    trainModels = TrainDDPGModels(updateParameters, trainActor, trainCritic)

    noiseInitVariance = 3  # control exploration
    varianceDiscount = .9995
    noiseDecayStartStep = bufferSize
    getNoise = GetExponentialDecayGaussNoise(noiseInitVariance, varianceDiscount, noiseDecayStartStep)
    actOneStepWithNoise = ActDDPGOneStepWithNoise(actionLow, actionHigh, actByPolicyTrain, getNoise)

    learningStartBufferSize = minibatchSize
    transit = TransitGymPendulum()
    getReward = RewardGymPendulum(angle_normalize)
    runDDPGTimeStep = RunTimeStep(actOneStepWithNoise, transit, getReward, isTerminalGymPendulum, addToMemory,
                 trainModels, minibatchSize, learningStartBufferSize, observe)

    reset = ResetGymPendulum(seed, observe)
    runEpisode = RunEpisode(reset, runDDPGTimeStep, maxTimeStep)

    ddpg = RunAlgorithm(runEpisode, bufferSize, maxEpisode)

    modelList = [actorModel, criticModel]
    modelList = resetTargetParamToTrainParam(modelList)
    meanRewardList, trajectory, trainedModelList = ddpg(modelList)

    trainedActorModel, trainedCriticModel = trainedModelList

# save Model
    modelIndex = 0
    actorFixedParam = {'actorModel': modelIndex}
    criticFixedParam = {'criticModel': modelIndex}
    parameters = {'env': ENV_NAME, 'Eps': maxEpisode, 'timeStep': maxTimeStep, 'batch': minibatchSize,
                  'gam': gamma, 'lrActor': learningRateActor, 'lrCritic': learningRateCritic,
                  'noiseVar': noiseInitVariance, 'varDiscout': varianceDiscount}

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
    trajectoryPath = os.path.join(dirName, '..', 'trajectory', 'pendulumTrajectory.pickle')
    saveToPickle(trajectory, trajectoryPath)

# demo& plot
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

