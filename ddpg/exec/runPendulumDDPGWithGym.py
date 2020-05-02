import gym
from src.ddpg import *
import os
from src.policy import ActInGymWithNoise
from environment.noise.noise import GetExponentialDecayGaussNoise
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# hyperparameters
maxEpisode = 200
maxTimeStep = 200
learningRateActor = 0.001    # learning rate for actor
learningRateCritic = 0.001    # learning rate for critic
GAMMA = 0.9     # reward discount
tau=0.01
bufferSize = 10000
minibatchSize = 128

ENV_NAME = 'Pendulum-v0'
env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

def main():
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high

    buildActorModel = BuildActorModel(state_dim, action_dim, action_bound)
    actorLayerWidths = [30]
    actorWriter, actorModel = buildActorModel(actorLayerWidths)

    buildCriticModel = BuildCriticModel(state_dim, action_dim)
    criticLayerWidths = [30]
    criticWriter, criticModel = buildCriticModel(criticLayerWidths)

    trainCriticBySASRQ = TrainCriticBySASRQ(learningRateCritic, GAMMA, criticWriter)
    trainCritic = TrainCritic(actByPolicyTarget, evaluateCriticTarget, trainCriticBySASRQ)

    trainActorFromGradients = TrainActorFromGradients(learningRateActor, actorWriter)
    trainActorOneStep = TrainActorOneStep(actByPolicyTrain, trainActorFromGradients, getActionGradients)
    trainActor = TrainActor(trainActorOneStep)

    paramUpdateInterval = 1
    updateParameters = UpdateParameters(tau, paramUpdateInterval)
    updateModelsByMiniBatch = UpdateModelsByMiniBatch(updateParameters, trainActor, trainCritic)

    noiseInitVariance = 3  # control exploration
    varianceDiscount = .9995
    noiseDecayStartStep = bufferSize
    getNoise = GetExponentialDecayGaussNoise(noiseInitVariance, varianceDiscount, noiseDecayStartStep)
    actInGymWithNoise = ActInGymWithNoise(action_bound, actByPolicyTrain, getNoise, env)

    learningStartBufferSize = minibatchSize
    runDDPGTimeStep = RunDDPGTimeStep(actInGymWithNoise, addToMemory, updateModelsByMiniBatch, minibatchSize,
                 learningStartBufferSize, env, render=True)

    reset = lambda: env.reset()
    runEpisode = RunEpisode(reset, runDDPGTimeStep, maxTimeStep)

    ddpg = DDPG(runEpisode, bufferSize, maxEpisode)
    trainedActorModel, trainedCriticModel = ddpg(actorModel, criticModel)

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



