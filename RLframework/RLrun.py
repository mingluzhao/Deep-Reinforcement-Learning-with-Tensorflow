import numpy as np
from collections import deque
import random
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def resetTargetParamToTrainParam(modelList):
    updatedModels = []
    for model in modelList:
        graph = model.graph
        updateParam_ = graph.get_collection_ref("hardReplaceTargetParam_")[0]
        model.run(updateParam_)
        updatedModels.append(model)
    return updatedModels


class UpdateParameters:
    def __init__(self, paramUpdateInterval, tau = None):
        self.paramUpdateInterval = paramUpdateInterval
        self.tau = tau

    def __call__(self, model, runTime = None):
        if runTime is None or runTime % self.paramUpdateInterval == 0:
            graph = model.graph
            updateParam_ = graph.get_collection_ref("updateParam_")[0]
            if self.tau is not None:
                # soft replace
                tau_ = graph.get_collection_ref("tau_")[0]
                model.run(updateParam_, feed_dict={tau_: self.tau})
            else:
                # hard replace
                model.run(updateParam_)
        return model


def addToMemory(buffer, state, action, reward, nextState):
    experience = (state, action, reward, nextState)
    buffer.append(experience)
    return buffer



class RunTimeStep:
    def __init__(self, actOneStep, transit, getReward, isTerminal, addToMemory,
                 trainModels, minibatchSize, learningStartBufferSize, observe = None):
        self.actOneStep = actOneStep

        self.transit = transit
        self.getReward = getReward
        self.isTerminal = isTerminal
        self.observe = observe
        self.addToMemory = addToMemory
        self.trainModels = trainModels
        self.minibatchSize = minibatchSize
        self.learningStartBufferSize = learningStartBufferSize
        self.observe = observe

    def __call__(self, state, modelList, replayBuffer, trajectory):
        runTime = len(trajectory)
        observation = self.observe(state) if self.observe is not None else state
        observation = np.asarray(observation).reshape(1, -1)
        action = self.actOneStep(modelList, observation, runTime)

        nextState = self.transit(state, action)
        reward = self.getReward(state, action)
        terminal = self.isTerminal(nextState)
        nextObservation = self.observe(nextState) if self.observe is not None else nextState
        replayBuffer = self.addToMemory(replayBuffer, observation, action, reward, nextObservation)
        trajectory.append((state, action))

        if runTime >= self.learningStartBufferSize:
            miniBatch = random.sample(replayBuffer, self.minibatchSize)
            modelList = self.trainModels(modelList, miniBatch, runTime)

        return reward, nextState, modelList, replayBuffer, terminal, trajectory


class RunTimeStepEnv:
    def __init__(self, actOneStep, addToMemory, trainModels, minibatchSize, learningStartBufferSize, env):
        self.actOneStep = actOneStep
        self.addToMemory = addToMemory
        self.trainModels = trainModels
        self.minibatchSize = minibatchSize
        self.learningStartBufferSize = learningStartBufferSize
        self.env = env

    def __call__(self, state, modelList, replayBuffer, trajectory):
        self.env.render()
        runTime = len(trajectory)
        state = np.asarray(state).reshape(1, -1)
        action = self.actOneStep(modelList, state, runTime)
        nextState, reward, terminal, info = self.env.step(action)
        replayBuffer = self.addToMemory(replayBuffer, state, action, reward, nextState)
        trajectory.append((state, action))

        if runTime >= self.learningStartBufferSize:
            miniBatch = random.sample(replayBuffer, self.minibatchSize)
            modelList = self.trainModels(modelList, miniBatch, runTime)

        return reward, nextState, modelList, replayBuffer, terminal, trajectory


class RunEpisode:
    def __init__(self, reset, runTimeStep, maxTimeStep):
        self.reset = reset
        self.runTimeStep = runTimeStep
        self.maxTimeStep = maxTimeStep

    def __call__(self, modelList, replayBuffer, trajectory):
        state = self.reset()
        episodeReward = 0
        for timeStep in range(self.maxTimeStep):
            reward, state, modelList, replayBuffer, terminal, trajectory = \
                self.runTimeStep(state, modelList, replayBuffer, trajectory)
            if terminal:
                print('------------terminal-----------------')
                break
            episodeReward += reward
        print('episodeReward: ', episodeReward)
        return modelList, replayBuffer, episodeReward, trajectory


class RunAlgorithm:
    def __init__(self, runEpisode, bufferSize, maxEpisode, print = True):
        self.bufferSize = bufferSize
        self.runEpisode = runEpisode
        self.maxEpisode = maxEpisode
        self.print = print

    def __call__(self, modelList):
        replayBuffer = deque(maxlen=int(self.bufferSize))
        episodeRewardList = []
        meanRewardList = []
        trajectory = []
        for episode in range(self.maxEpisode):
            modelList, replayBuffer, episodeReward, trajectory = self.runEpisode(modelList, replayBuffer, trajectory)
            episodeRewardList.append(episodeReward)
            meanRewardList.append(np.mean(episodeRewardList))

            if print:
                print('episode', episode)
                if episode == self.maxEpisode - 1:
                    print('mean episode reward: ', int(np.mean(episodeRewardList)))

        return meanRewardList, trajectory, modelList



