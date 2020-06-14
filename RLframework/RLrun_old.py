import numpy as np
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
        self.runTime = 0

    def __call__(self, model):
        if self.runTime % self.paramUpdateInterval == 0:
            graph = model.graph
            updateParam_ = graph.get_collection_ref("updateParam_")[0]
            if self.tau is not None:
                tau_ = graph.get_collection_ref("tau_")[0]
                model.run(updateParam_, feed_dict={tau_: self.tau})
            else:
                model.run(updateParam_)
        self.runTime += 1

        return model


class SampleOneStep:
    def __init__(self, transit, getReward):
        self.transit = transit
        self.getReward = getReward

    def __call__(self, state, action):
        nextState = self.transit(state, action)
        reward = self.getReward(state, action, nextState)

        return reward, nextState


class SampleOneStepUsingGym:
    def __init__(self, env):
        self.env = env

    def __call__(self, state, action):
        nextState, reward, terminal, info = self.env.step(action)

        return reward, nextState


class SampleFromMemory:
    def __init__(self, minibatchSize):
        self.minibatchSize = minibatchSize

    def __call__(self, memoryBuffer):
        sample = random.sample(memoryBuffer, self.minibatchSize)
        return sample


class LearnFromBuffer:
    def __init__(self, learningStartBufferSize, sampleFromMemory, trainModels):
        self.learningStartBufferSize = learningStartBufferSize
        self.sampleFromMemory = sampleFromMemory
        self.trainModels = trainModels

    def __call__(self, replayBuffer, runTime):
        if runTime >= self.learningStartBufferSize:
            miniBatch = self.sampleFromMemory(replayBuffer)
            self.trainModels(miniBatch)


class RunTimeStep:
    def __init__(self, actOneStep, sampleOneStep, learnFromBuffer, observe = None):
        self.actOneStep = actOneStep
        self.sampleOneStep = sampleOneStep
        self.learnFromBuffer = learnFromBuffer
        self.observe = observe

    def __call__(self, state, replayBuffer, trajectory):
        runTime = len(trajectory)
        observation = self.observe(state) if self.observe is not None else state
        action = self.actOneStep(observation, runTime)
        reward, nextState = self.sampleOneStep(state, action)
        nextObservation = self.observe(nextState) if self.observe is not None else nextState
        replayBuffer.append((observation, action, reward, nextObservation))
        trajectory.append((state, action, reward, nextState))
        self.learnFromBuffer(replayBuffer, runTime)

        return reward, nextState, replayBuffer, trajectory


class RunEpisode:
    def __init__(self, reset, runTimeStep, maxTimeStep, isTerminal):
        self.reset = reset
        self.runTimeStep = runTimeStep
        self.maxTimeStep = maxTimeStep
        self.isTerminal = isTerminal

    def __call__(self, replayBuffer, trajectory):
        state = self.reset()
        episodeReward = 0
        for timeStep in range(self.maxTimeStep):
            reward, state, replayBuffer, trajectory = self.runTimeStep(state, replayBuffer, trajectory)
            episodeReward += reward
            terminal = self.isTerminal(state)
            if terminal:
                print('------------terminal----------------- timeStep: ', timeStep)
                break
        print('episodeReward: ', episodeReward, 'runSteps: ', len(trajectory))
        return replayBuffer, episodeReward, trajectory



class RunAlgorithm:
    def __init__(self, runEpisode, maxEpisode, print = True):
        self.runEpisode = runEpisode
        self.maxEpisode = maxEpisode
        self.print = print

    def __call__(self, replayBuffer):
        episodeRewardList = []
        meanRewardList = []
        trajectory = []
        for episode in range(self.maxEpisode):
            replayBuffer, episodeReward, trajectory = self.runEpisode(replayBuffer, trajectory)
            episodeRewardList.append(episodeReward)
            meanRewardList.append(np.mean(episodeRewardList))

            if print:
                print('EPISODE ', episode)
                print('mean episode reward', np.mean(episodeRewardList))
                if episode == self.maxEpisode - 1:
                    print('mean episode reward: ', int(np.mean(episodeRewardList)))

        return meanRewardList, trajectory
