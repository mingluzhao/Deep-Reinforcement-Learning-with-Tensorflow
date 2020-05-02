import gym
from src.ddpg import *
from environment.noise.noise import *


class ActMountainWithNoise:
    def __init__(self, noise, actionRange, actByPolicyTrain):
        self.noise = noise
        self.actionRange = abs(actionRange)
        self.actByPolicyTrain = actByPolicyTrain

    def __call__(self, pointer, actorModel, state):
        stateBatch = np.asarray(state).reshape(1, -1)
        actionPerfect = self.actByPolicyTrain(actorModel, stateBatch)[0]
        addedNoise = self.noise.sample()
        action = actionPerfect + self.noise.epsilon * addedNoise
        nextState, reward, terminal, info = env.step(action)
        self.noise.update()

        return state, action, reward, nextState, terminal


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
#         return state, action, reward, nextState, terminal

# class ActInGymWithNoise:
#     def __init__(self, actionRange, getNoise, actByPolicyTrain):
#         self.actionRange = actionRange
#         self.getNoise = getNoise
#         self.actByPolicyTrain = actByPolicyTrain
#
#     def __call__(self, actorModel, state, oldNoise):
#         stateBatch = np.asarray(state).reshape(1, -1)
#         actionPerfect = self.actByPolicyTrain(actorModel, stateBatch)[0]
#         noise = self.getNoise(oldNoise, actionPerfect)
#         action = np.clip(actionPerfect + noise, -self.actionRange, self.actionRange)
#         nextState, reward, terminal, info = env.step(action)
#
#         return state, action, reward, nextState, terminal, noise

# class ActPendulumWithNoise:
#     def __init__(self, var, varianceDiscount, actionRange, actByPolicyTrain):
#         self.var = var
#         self.varianceDiscount = varianceDiscount
#         self.actionRange = abs(actionRange)
#         self.actByPolicyTrain = actByPolicyTrain
#
#     def __call__(self, actorModel, state):
#         var = self.var
#         stateBatch = np.asarray(state).reshape(1, -1)
#         actionPerfect = self.actByPolicyTrain(actorModel, stateBatch)[0]
#         action = np.clip(np.random.normal(actionPerfect, var), -self.actionRange, self.actionRange)
#         nextState, reward, terminal, info = env.step(action)
#
#         return state, action, reward, nextState, terminal

#####################  hyper parameters  ####################

MAX_EPISODES = 200
MAX_EP_STEPS = 1000
LR_A = 1e-2    # learning rate for actor
LR_C = 5e-3  # learning rate for critic
GAMMA = 0.99     # reward discount
tau=0.001
MEMORY_CAPACITY = 1000000
BATCH_SIZE = 64
learningStartBufferSize = 20000

RENDER = True
OUTPUT_GRAPH = True
# ENV_NAME = 'CartPole-v0'

ENV_NAME = 'MountainCarContinuous-v0'

env = gym.make(ENV_NAME)
env = env.unwrapped
seed = 14
env.seed(seed)

def main():
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high

    buildActorModel = BuildActorModel(state_dim, action_dim, action_bound)
    actorTrainingLayerWidths = [256, 128]
    actorTargetLayerWidths = actorTrainingLayerWidths
    actorWriter, actorModel = buildActorModel(actorTrainingLayerWidths, actorTargetLayerWidths)

    buildCriticModel = BuildCriticModel(state_dim, action_dim)
    criticTrainingLayerWidths = [256, 128]
    criticTargetLayerWidths = criticTrainingLayerWidths
    criticWriter, criticModel = buildCriticModel(criticTrainingLayerWidths, criticTargetLayerWidths)

    updateParameters = UpdateParameters(tau)

    trainCriticBySASRQ = TrainCriticBySASRQ(LR_C, GAMMA, criticWriter)
    trainCritic = TrainCritic(actByPolicyTarget, evaluateCriticTarget, trainCriticBySASRQ)

    trainActorFromGradients = TrainActorFromGradients(LR_A, actorWriter)
    trainActorOneStep = TrainActorOneStep(actByPolicyTrain, trainActorFromGradients, getActionGradients)
    trainActor = TrainActor(trainActorOneStep)

    updateModelsByMiniBatch = UpdateModelsByMiniBatch(trainActor, trainCritic)

    noise = OUNoise(action_dim, seed, mu=0., theta=0.15, sigma=0.2)
    actOneStepWithNoise = ActMountainWithNoise(noise, action_bound, actByPolicyTrain)
    rewardScalingFactor = 1
    addToMemory = AddToMemory(rewardScalingFactor)
    paramUpdateInterval = 1
    runDDPGTimeStep = RunDDPGTimeStep(actOneStepWithNoise, addToMemory, updateModelsByMiniBatch,
                                      updateParameters, BATCH_SIZE, learningStartBufferSize, paramUpdateInterval, env)

    reset = lambda: env.reset()
    runEpisode = RunEpisode(reset, runDDPGTimeStep, MAX_EP_STEPS)

    ddpg = DDPG(initializeMemory, runEpisode, MEMORY_CAPACITY, MAX_EPISODES)
    trainedActorModel, trainedCriticModel = ddpg(actorModel, criticModel)

    env.close()


if __name__ == '__main__':
    main()


