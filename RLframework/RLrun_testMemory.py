import numpy as np
import random
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from collections import deque
import psutil

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
        sampleIndex = [random.randint(0, len(memoryBuffer) - 1) for _ in range(self.minibatchSize)]
        sample = [memoryBuffer[index] for index in sampleIndex]

        return sample


class LearnFromBuffer:
    def __init__(self, learningStartBufferSize, sampleFromMemory, trainModels, learnInterval = 1):
        self.learningStartBufferSize = learningStartBufferSize
        self.sampleFromMemory = sampleFromMemory
        self.trainModels = trainModels
        self.learnInterval = learnInterval
        self.getAgentBuffer = lambda buffer, id: [[bufferElement[id] for bufferElement in timeStepBuffer] for timeStepBuffer in buffer]

    def __call__(self, replayBuffer, runTime, agentID = None):
        if runTime >= self.learningStartBufferSize and runTime % self.learnInterval == 0:
            agentBuffer = self.getAgentBuffer(replayBuffer, agentID) if agentID is not None else replayBuffer
            miniBatch = self.sampleFromMemory(agentBuffer)
            self.trainModels(miniBatch)


class RunTimeStep:
    def __init__(self, actOneStep, sampleOneStep, learnFromBuffer, observe = None):
        self.actOneStep = actOneStep
        self.sampleOneStep = sampleOneStep
        self.learnFromBuffer = learnFromBuffer
        self.observe = observe
        self.runTime = 0

    def __call__(self, state, replayBuffer):
        observation = self.observe(state) if self.observe is not None else state
        action = self.actOneStep(observation, self.runTime)
        reward, nextState = self.sampleOneStep(state, action)
        nextObservation = self.observe(nextState) if self.observe is not None else nextState
        replayBuffer.append((observation, action, reward, nextObservation))

        isMultiAgent = isinstance(self.learnFromBuffer, list)
        if isMultiAgent:
            for id, agentLearn in enumerate(self.learnFromBuffer):
                agentLearn(replayBuffer, self.runTime, id)
        else:
            self.learnFromBuffer(replayBuffer, self.runTime)

        self.runTime += 1
        return reward, nextState, replayBuffer


class StartLearn:
    def __init__(self, learningStartBufferSize, learnInterval):
        self.learningStartBufferSize = learningStartBufferSize
        self.learnInterval = learnInterval

    def __call__(self, runTime):
        shouldStart = runTime >= self.learningStartBufferSize and runTime % self.learnInterval == 0
        return shouldStart


def getBuffer(bufferSize):
    replayBuffer = deque(maxlen=int(bufferSize))
    return replayBuffer


class RunEpisode:
    def __init__(self, reset, runTimeStep, maxTimeStep, isTerminal):
        self.reset = reset
        self.runTimeStep = runTimeStep
        self.maxTimeStep = maxTimeStep
        self.isTerminal = isTerminal

    def __call__(self, replayBuffer):
        state = self.reset()
        reward, state, replayBuffer = self.runTimeStep(state, replayBuffer)
        episodeReward = np.array(reward)

        for timeStep in range(self.maxTimeStep-1):
            reward, state, replayBuffer = self.runTimeStep(state, replayBuffer)
            episodeReward = episodeReward + np.array(reward)
            terminal = self.isTerminal(state)
            terminalCheck = (np.sum(np.array(terminal)) != 0)
            if terminalCheck:
                break
        return replayBuffer, episodeReward


class SaveModel:
    def __init__(self, modelSaveRate, saveVariables, getCurrentModel, modelSavePath, saveAllmodels = False):
        self.modelSaveRate = modelSaveRate
        self.saveVariables = saveVariables
        self.getCurrentModel = getCurrentModel
        self.epsNum = 0
        self.modelSavePath = modelSavePath
        self.saveAllmodels = saveAllmodels

    def __call__(self):
        self.epsNum += 1
        if self.epsNum % self.modelSaveRate == 0:
            modelSavePathToUse = self.modelSavePath + str(self.epsNum) + "eps" if self.saveAllmodels else self.modelSavePath
            model = self.getCurrentModel()
            with model.as_default():
                self.saveVariables(model, modelSavePathToUse)

class RunAlgorithm:
    def __init__(self, runEpisode, maxEpisode, saveModels, numAgents = 1, printEpsFrequency = 1000):
        self.runEpisode = runEpisode
        self.maxEpisode = maxEpisode
        self.saveModels = saveModels
        self.numAgents = numAgents
        self.printEpsFrequency = printEpsFrequency
        self.multiAgent = (self.numAgents > 1)

    def __call__(self, replayBuffer):
        episodeRewardList = []
        meanRewardList = []
        agentsEpsRewardList = [list() for agentID in range(self.numAgents)] if self.multiAgent else []
        memoryUsedRSS = []
        memoryUsedVMS = []

        for episodeID in range(self.maxEpisode):
            process = psutil.Process(os.getpid())
            memoryRSS = process.memory_info().rss/ 1024.0 / 1024.0
            memoryVMS = process.memory_info().vms/ 1024.0 / 1024.0
            memoryUsedRSS.append(memoryRSS)
            memoryUsedVMS.append(memoryVMS)

            replayBuffer, episodeReward = self.runEpisode(replayBuffer)
            [saveModel() for saveModel in self.saveModels] if self.multiAgent else self.saveModels()
            if self.multiAgent:
                episodeRewardList.append(np.sum(episodeReward))
                [agentRewardList.append(agentEpsReward) for agentRewardList, agentEpsReward in zip(agentsEpsRewardList, episodeReward)]
                meanRewardList.append(np.mean(episodeRewardList))

                if episodeID % self.printEpsFrequency == 0:
                    lastTimeSpanMeanReward = np.mean(episodeRewardList[-self.printEpsFrequency:])
                    print("episodes: {}, last {} eps mean episode reward: {}, agent mean reward: {}".format(
                        episodeID, self.printEpsFrequency, lastTimeSpanMeanReward,
                        [np.mean(rew[-self.printEpsFrequency:]) for rew in agentsEpsRewardList]))
            else:
                episodeRewardList.append(episodeReward)
                print('episode {}: mean eps reward {}'.format(len(episodeRewardList), np.mean(episodeRewardList)))

        return memoryUsedRSS, memoryUsedVMS



