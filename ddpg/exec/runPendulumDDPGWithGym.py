import matplotlib.pyplot as plt
import gym
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from src.ddpg import actByPolicyTrain, actByPolicyTarget, evaluateCriticTarget, getActionGradients, \
    BuildActorModel, BuildCriticModel, TrainCriticBySASRQ, TrainCritic, TrainActorFromGradients,\
    TrainActorOneStep, TrainActor, TrainDDPGModels
from RLframework.RLrun import resetTargetParamToTrainParam, addToMemory, UpdateParameters, RunTimeStepEnv, \
    RunEpisode, RunAlgorithm
from src.policy import ActDDPGOneStepWithNoise
from environment.noise.noise import GetExponentialDecayGaussNoise


maxEpisode = 200
maxTimeStep = 200
learningRateActor = 0.001
learningRateCritic = 0.001
gamma = 0.9     # reward discount
tau=0.01
bufferSize = 10000
minibatchSize = 128

ENV_NAME = 'Pendulum-v0'
env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)


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
    runDDPGTimeStep = RunTimeStepEnv(actOneStepWithNoise, addToMemory, trainModels, minibatchSize, learningStartBufferSize, env)

    reset = lambda: env.reset()
    runEpisode = RunEpisode(reset, runDDPGTimeStep, maxTimeStep)

    ddpg = RunAlgorithm(runEpisode, bufferSize, maxEpisode)
    modelList = [actorModel, criticModel]
    modelList = resetTargetParamToTrainParam(modelList)
    meanRewardList, trajectory, trainedModelList = ddpg(modelList)

    trainedActorModel, trainedCriticModel = trainedModelList

    env.close()


if __name__ == '__main__':
    main()


# parameters [30], [30]:
#     cartPole example:
#         MYDDPG: mean episode reward:  -645, -578, -633
#         EXP: -622, -620, -616
#     chasing example:
#         MYDDPG: (1, -1)
#         EXP: output action always (-1, 1)
#
# parameters [20, 20], [100, 100]:
#     cartPole example:
#         my model: -515, -541, -534
#         exp model:
#     chasing example
#         my model: always [-1, -1]
#         exp model:



