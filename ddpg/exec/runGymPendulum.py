import tensorflow as tf
import numpy as np
import gym
import time
from src.ddpg import *
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# class ActInGymWithNoise:
#     def __init__(self, var, varianceDiscount, actionRange, actByPolicyTrain):
#         self.var = var
#         self.varianceDiscount = varianceDiscount
#         self.actionRange = abs(actionRange)
#         self.actByPolicyTrain = actByPolicyTrain
#
#     def __call__(self, pointer, actorModel, state):
#         var = self.var
#         if pointer > MEMORY_CAPACITY:
#             var = self.var* self.varianceDiscount ** (pointer - MEMORY_CAPACITY)
#         stateBatch = np.asarray(state).reshape(1, -1)
#         actionPerfect = self.actByPolicyTrain(actorModel, stateBatch)[0]
#         action = np.clip(np.random.normal(actionPerfect, var), -self.actionRange, self.actionRange)
#         nextState, reward, terminal, info = env.step(action)
#
#         if pointer % MAX_EP_STEPS == 0:
#             print('episode: ', int(pointer/MAX_EP_STEPS),  'var:', round(var, 2))
#
#         return state, action, reward, nextState, terminal

class GetNoise:
    def __init__(self, noiseInitVariance, varianceDiscount):
        self.noiseInitVariance = noiseInitVariance
        self.varianceDiscount = varianceDiscount

    def __call__(self, runStep):
        var = self.noiseInitVariance
        if runStep > MEMORY_CAPACITY:
            var = self.noiseInitVariance* self.varianceDiscount ** (runStep - MEMORY_CAPACITY)
        noise = np.random.normal(0, var)
        if runStep % MAX_EP_STEPS == 0:
            print('episode: ', int(runStep/MAX_EP_STEPS),  'var:', round(var, 2))

        return noise


class ActInGymWithNoise:
    def __init__(self, actionRange, actByPolicyTrain, getNoise):
        self.actionRange = abs(actionRange)
        self.actByPolicyTrain = actByPolicyTrain
        self.getNoise = getNoise

    def __call__(self, runStep, actorModel, state):
        noise = self.getNoise(runStep)
        stateBatch = np.asarray(state).reshape(1, -1)
        actionPerfect = self.actByPolicyTrain(actorModel, stateBatch)[0]
        action = np.clip(noise + actionPerfect, -self.actionRange, self.actionRange)
        nextState, reward, terminal, info = env.step(action)

        return state, action, reward, nextState, terminal


####################  hyper parameters  ####################

np.random.seed(1)
tf.set_random_seed(1)

MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.9     # reward discount
tau=0.01
# REPLACEMENT = [
#     dict(name='soft', tau=0.01),
#     dict(name='hard', rep_iter_a=600, rep_iter_c=500)
# ][0]            # you can try different target replacement strategies
MEMORY_CAPACITY = 10000
BATCH_SIZE = 128

RENDER = True
OUTPUT_GRAPH = True
ENV_NAME = 'Pendulum-v0'


env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

def main():
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high

    buildActorModel = BuildActorModel(state_dim, action_dim, action_bound)
    actorTrainingLayerWidths = [30]
    # actorTrainingLayerWidths = [20, 20]
    actorTargetLayerWidths = actorTrainingLayerWidths
    actorWriter, actorModel = buildActorModel(actorTrainingLayerWidths, actorTargetLayerWidths)

    buildCriticModel = BuildCriticModel(state_dim, action_dim)
    criticTrainingLayerWidths = [30]
    # criticTrainingLayerWidths = [100, 100]
    criticTargetLayerWidths = criticTrainingLayerWidths
    criticWriter, criticModel = buildCriticModel(criticTrainingLayerWidths, criticTargetLayerWidths)

    trainCriticBySASRQ = TrainCriticBySASRQ(LR_C, GAMMA, criticWriter)
    trainCritic = TrainCritic(actByPolicyTarget, evaluateCriticTarget, trainCriticBySASRQ)

    trainActorFromGradients = TrainActorFromGradients(LR_A, actorWriter)
    trainActorOneStep = TrainActorOneStep(actByPolicyTrain, trainActorFromGradients, getActionGradients)
    trainActor = TrainActor(trainActorOneStep)

    paramUpdateInterval = 1
    updateParameters = UpdateParameters(tau, paramUpdateInterval)
    updateModelsByMiniBatch = UpdateModelsByMiniBatch(updateParameters, trainActor, trainCritic)

    noiseInitVariance = 3  # control exploration
    varianceDiscount = .9995
    getNoise = GetNoise(noiseInitVariance, varianceDiscount)
    actInGymWithNoise = ActInGymWithNoise(action_bound, actByPolicyTrain, getNoise)

    learningStartBufferSize = BATCH_SIZE
    runDDPGTimeStep = RunDDPGTimeStep(actInGymWithNoise, addToMemory, updateModelsByMiniBatch, BATCH_SIZE,
                 learningStartBufferSize, env)

    reset = lambda: env.reset()
    runEpisode = RunEpisode(reset, runDDPGTimeStep, MAX_EP_STEPS)

    ddpg = DDPG(runEpisode, MEMORY_CAPACITY, MAX_EPISODES)
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



