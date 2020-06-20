import numpy as np
import random
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from collections import  deque

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
        # print(action)
        nextState = self.transit(state, action)
        reward = self.getReward(state, action, nextState)
        # print('state', state, 'action', action, 'reward', reward)

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


# class LearnFromBufferMultiagent:
#     def __init__(self, learningStartBufferSize, trainModelsFromBuffer, learnInterval = 1):
#         self.learningStartBufferSize = learningStartBufferSize
#         self.trainModelsFromBuffer = trainModelsFromBuffer
#         self.learnInterval = learnInterval
#
#     def __call__(self, replayBuffer, runTime):
#         if runTime >= self.learningStartBufferSize and runTime % self.learnInterval == 0:
#             self.trainModelsFromBuffer(replayBuffer)

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

        isMultiAgent = isinstance(self.learnFromBuffer, list)
        if isMultiAgent:
            for id, agentLearn in enumerate(self.learnFromBuffer):
                agentLearn(replayBuffer, runTime, id)
        else:
            self.learnFromBuffer(replayBuffer, runTime)

        return reward, nextState, replayBuffer, trajectory



class RunEpisode:
    def __init__(self, reset, runTimeStep, maxTimeStep, isTerminal):
        self.reset = reset
        self.runTimeStep = runTimeStep
        self.maxTimeStep = maxTimeStep
        self.isTerminal = isTerminal
        self.collisionCount = 0

    def __call__(self, replayBuffer, trajectory):
        state = self.reset()
        # numAgents = 1 if len(state.shape) == 1 else state.shape[0]
        reward, state, replayBuffer, trajectory = self.runTimeStep(state, replayBuffer, trajectory)
        episodeReward = np.array(reward)

        for timeStep in range(self.maxTimeStep):
            reward, state, replayBuffer, trajectory = self.runTimeStep(state, replayBuffer, trajectory)
            if any(reward) == 10:
                self.collisionCount +=1

            episodeReward = episodeReward + np.array(reward)
            terminal = self.isTerminal(state)
            terminalCheck = (np.sum(np.array(terminal)) != 0)
            if terminalCheck:
                break
        if self.collisionCount != 0:
            print(self.collisionCount)
        return replayBuffer, episodeReward, trajectory


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

    def __call__(self, replayBuffer):
        episodeRewardList = []
        meanRewardList = []
        lastTimeSpanMeanRewardList = []

        trajectory = []
        agentsEpsRewardList = [list() for agentID in range(self.numAgents)]

        for episode in range(self.maxEpisode):
            replayBuffer, episodeReward, trajectory = self.runEpisode(replayBuffer, trajectory)
            episodeRewardList.append(np.sum(episodeReward))
            [agentRewardList.append(agentEpsReward) for agentRewardList, agentEpsReward in zip(agentsEpsRewardList, episodeReward)]
            meanRewardList.append(np.mean(episodeRewardList))
            # print('eps', episode, 'reward', episodeReward, 'mean reward', np.mean(episodeRewardList))

            [saveModel() for saveModel in self.saveModels]

            if episode % self.printEpsFrequency == 0:
                lastTimeSpanMeanReward = np.mean(episodeRewardList[-self.printEpsFrequency:])
                lastTimeSpanMeanRewardList.append(lastTimeSpanMeanReward)

                print("steps: {}, episodes: {}, last {} eps mean episode reward: {}, agent episode reward: {}".format(
                    len(replayBuffer), len(episodeRewardList), self.printEpsFrequency, lastTimeSpanMeanReward,
                    [np.mean(rew[-self.printEpsFrequency:]) for rew in agentsEpsRewardList]))

        return meanRewardList, trajectory

