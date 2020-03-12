import numpy as np
from matplotlib import pyplot as plt

pts = np.random.multivariate_normal([0, 0], [[1,0],[0,30]], size=10000, check_valid='warn')

plt.scatter(pts[:, 0], pts[:, 1], s=1)
plt.xlim((-2,2))
plt.ylim((-30,30))
plt.show()


class DeterministicGaussianIsotropicPolicy:
    def __init__(self, theta, y):
        self.theta = theta
        self.y = y

    def __call__(self, state):
        covarianceMatrix = [[self.y, 0], [0, self.y]]
        pts = np.random.multivariate_normal([0, 0], covarianceMatrix, size=10000, check_valid='warn')




def ActionValueActorCritic(policyModel, qModel, state, rewardFunction, transitionFunction, phiFunction, maxStep, alpha, beta):
    s = 0
    theta = 0
    policy = policyModel(theta)
    action = policy(state)
    w = 0
    for step in range(maxStep):
        reward = rewardFunction(state, action)
        nextState = transitionFunction(state, action)
        nextAction = policy(nextState)
        QwFunction = qModel(w)
        TDerror = reward + QwFunction(nextState, nextAction) - QwFunction(state, action)
        theta = theta + alpha * QwFunction(state, action)
        w = w + beta * TDerror * phiFunction(state, action)
        action = nextAction
        state = nextState







class Memory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def __len__(self):
        return len(self.buffer)


import random


def sampleFromMemory(buffer, batchSize):
    state_batch = []
    action_batch = []
    reward_batch = []
    next_state_batch = []
    done_batch = []

    sampledBatch = random.sample(buffer, batchSize)
    # reshapedBatch = np.concatenate(sampledBatch)
    # state_batch, action_batch, reward_batch, next_state_batch, done_batch = list(zip(*reshapedBatch))

    for experience in sampledBatch:
        state, action, reward, next_state, done = experience
        state_batch.append(state)
        action_batch.append(action)
        reward_batch.append(reward)
        next_state_batch.append(next_state)
        done_batch.append(done)

    return state_batch, action_batch, reward_batch, next_state_batch, done_batch


def addToMemory(buffer, state, action, reward, nextState, done):
    experience = (state, action, np.array([reward]), nextState, done)
    buffer.append(experience)


class MemoryBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)


    def __len__(self):
        return len(self.buffer)

    for step in range(500):
        action = agent.get_action(state)
        action = noise.get_action(action, step)
        new_state, reward, done, _ = env.step(action)
        agent.memory.push(state, action, reward, new_state, done)

        if len(agent.memory) > batch_size:
            agent.update(batch_size)

        state = new_state
        episode_reward += reward