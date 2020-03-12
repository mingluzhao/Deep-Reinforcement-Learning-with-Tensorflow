import random
import numpy as np
from matplotlib import pyplot as plt
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
from torch.autograd import Variable
import gym
import torch.optim as optim


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
    return buffer

class GetMemoryBuffer:
    def __init__(self, addToMemory, addNoiseToAction):
        self.addToMemory = addToMemory
        self.addNoiseToAction = addNoiseToAction

    def __call__(self, memoryBuffer, addBufferTimes, policy, state, transit, rewardFunction):
        for step in range(addBufferTimes):
            intendedAction = policy(state)
            action = self.addNoiseToAction(intendedAction, step)
            nextState = transit(state, action)
            reward = rewardFunction(state, action)
            memoryBuffer = addToMemory(memoryBuffer, state, action, reward, nextState, done)
            return memoryBuffer

def initializeMemory(bufferSize):
    return deque(maxlen=bufferSize)

class SampleTrajectory():
    def __init__(self, maxTimeStep, transitionFunction, isTerminal):
        self.maxTimeStep = maxTimeStep
        self.transitionFunction = transitionFunction
        self.isTerminal = isTerminal

    def __call__(self, actor):
        oldState, action = None, None
        oldState = self.transitionFunction(oldState, action)
        trajectory = []

        for time in range(self.maxTimeStep):
            oldStateBatch = oldState.reshape(1, -1)
            actionBatch = actor(oldStateBatch)
            action = actionBatch[0]
            # actionBatch shape: batch * action Dimension; only keep action Dimention in shape
            newState = self.transitionFunction(oldState, action)
            trajectory.append((oldState, action))
            terminal = self.isTerminal(oldState)
            if terminal:
                break
            oldState = newState
        return trajectory

class AccumulateReward():
    def __init__(self, decay, rewardFunction):
        self.decay = decay
        self.rewardFunction = rewardFunction

    def __call__(self, trajectory):
        rewards = [self.rewardFunction(state, action) for state, action in trajectory]
        accumulateReward = lambda accumulatedReward, reward: self.decay * accumulatedReward + reward
        accumulatedRewards = np.array(
            [ft.reduce(accumulateReward, reversed(rewards[TimeT:])) for TimeT in range(len(rewards))])
        return accumulatedRewards

def approximateValue(stateBatch, criticModel):
    graph = criticModel.graph
    state_ = graph.get_tensor_by_name('inputs/state_:0')
    value_ = graph.get_tensor_by_name('outputs/value_/BiasAdd:0')
    valueBatch = criticModel.run(value_, feed_dict={state_: stateBatch})
    return valueBatch


def main():
    # Networks
    actor = Actor(self.num_states, hidden_size, self.num_actions)
    actor_target = Actor(self.num_states, hidden_size, self.num_actions)
    critic = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions)
    critic_target = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions)

    actorTargetParam = actorParam
    criticTargetParam = criticParam

    bufferSize = 1000
    replayBuffer = initializeMemory(bufferSize)

    maxEpisode = 1000

    for episode in range(maxEpisode):
        noise = initalizeNoise()
        state = reset()

        for step in range(stepNum):
            intendedAction = policy(state)
            action = addNoiseToAction(intendedAction, step)
            nextState = transit(state, action)
            reward = rewardFunction(state, action)
            memoryBuffer = addToMemory(replayBuffer, state, action, reward, nextState)

            state, action, reward, nextState = sampleFromMemory(memoryBuffer, minibatchSize)

            criticValue = getCriticValue(state, action, criticParams)

            nextTargetAction = actorTargetAct(nextState)
            nextCriticTargetValue = getCriticTargetValue(nextState, nextTargetAction)
            QupdateTarget = reward + gamma * nextCriticTargetValue
            criticUpdate = minimizeMSE(criticValue, QupdateTarget)

            action = actorAct(state)
            nextCriticValue = getCriticValue(state, action)
            actorUpdate = mean(gradientWrtActorParam(nextAction) * gradientWrtNextAction(nextCriticValue))
            # actorUpdate = -critic.forward(states, actor.forward(states)).mean()

            actorTargetParam = tau* actorParam + (1-tau) * actorTargetParam
            criticTargetParam = tau* criticParam + (1-tau) * criticTargetParam

