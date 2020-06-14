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

    def __call__(self, replayBuffer, runTime):
        if runTime >= self.learningStartBufferSize and runTime % self.learnInterval == 0:
            # print('learn')
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
        # print('state: ', state, ', action: ', action, ', reward: ', reward)

        nextObservation = self.observe(nextState) if self.observe is not None else nextState
        replayBuffer.append((observation, action, reward, nextObservation))
        trajectory.append((state, action, reward, nextState))
        self.learnFromBuffer(replayBuffer, runTime)

        return reward, nextState, replayBuffer, trajectory


def getBuffer(bufferSize):
    replayBuffer = deque(maxlen=int(bufferSize))
    return replayBuffer



class RunMultiAgentTimeStep:
    def __init__(self, actOneStep, sampleOneStep, learnFromBuffer, observe = None, multiagent = False):
        self.actOneStep = actOneStep
        self.sampleOneStep = sampleOneStep
        self.learnFromBuffer = learnFromBuffer
        self.observe = observe
        self.multiagent = multiagent
        self.getAgentBuffer = lambda buffer, id: [[bufferElement[id] for bufferElement in timeStepBuffer] for timeStepBuffer in buffer]


    def __call__(self, state, replayBuffer, trajectory):
        runTime = len(trajectory)
        observation = self.observe(state) if self.observe is not None else state
        action = self.actOneStep(observation, runTime)
        reward, nextState = self.sampleOneStep(state, action)
        # print("reward ", reward)

        nextObservation = self.observe(nextState) if self.observe is not None else nextState
        replayBuffer.append((observation, action, reward, nextObservation))
        trajectory.append((state, action, reward, nextState))

        # if self.multiagent:
        for id, agentLearn in enumerate(self.learnFromBuffer):
            agentBuffer = self.getAgentBuffer(replayBuffer, id)
            agentLearn(agentBuffer, runTime)

        # if runTime % 100 == 0:
        #     if self.multiagent:
        #         getAgentBuffer = lambda buffer, id: [[bufferElement[id] for bufferElement in timeStepBuffer] for timeStepBuffer in buffer]
        #         for id, agentLearn in enumerate(self.learnFromBuffer):
        #             agentBuffer = getAgentBuffer(replayBuffer, id)
        #             agentLearn(agentBuffer, runTime)
        #     else:
        #         self.learnFromBuffer(replayBuffer, runTime)

        return reward, nextState, replayBuffer, trajectory



class RunEpisode:
    def __init__(self, reset, runTimeStep, maxTimeStep, isTerminal):
        self.reset = reset
        self.runTimeStep = runTimeStep
        self.maxTimeStep = maxTimeStep
        self.isTerminal = isTerminal
        self.notTerminalCount = 0

    def __call__(self, replayBuffer, trajectory):
        state = self.reset()
        episodeReward = np.zeros(2)
        # episodeReward = 0
        for timeStep in range(self.maxTimeStep):
            reward, state, replayBuffer, trajectory = self.runTimeStep(state, replayBuffer, trajectory)
            episodeReward += np.array(reward)
            terminal = self.isTerminal(state)
            terminalCheck = (np.sum(np.array(terminal)) != 0)
            if terminalCheck:
                break

        return replayBuffer, episodeReward, trajectory


class SaveModel:
    def __init__(self, modelSaveRate, saveVariables, getCurrentModel, modelSavePath):
        self.modelSaveRate = modelSaveRate
        self.saveVariables = saveVariables
        self.getCurrentModel = getCurrentModel
        self.epsNum = 0
        self.modelSavePath = modelSavePath

    def __call__(self):
        if self.epsNum % self.modelSaveRate == 0:
            model = self.getCurrentModel()
            with model.as_default():
                self.saveVariables(model, self.modelSavePath)

        self.epsNum += 1



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
        trajectory = []
        agentsEpsRewardList = [list() for agentID in range(self.numAgents)]

        for episode in range(self.maxEpisode):
            replayBuffer, episodeReward, trajectory = self.runEpisode(replayBuffer, trajectory)
            episodeRewardList.append(np.sum(episodeReward))
            [agentRewardList.append(agentEpsReward) for agentRewardList, agentEpsReward in zip(agentsEpsRewardList, episodeReward)]
            # meanRewardList.append(np.mean(episodeRewardList))
            # print('eps reward', episodeReward, 'mean reward', np.mean(episodeRewardList))

            [saveModel() for saveModel in self.saveModels]


            if episode % self.printEpsFrequency == 0:
                lastTimeSpanMeanReward = np.mean(episodeRewardList[-self.printEpsFrequency:])
                meanRewardList.append(lastTimeSpanMeanReward)

                print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}".format(
                    len(replayBuffer), len(episodeRewardList), np.mean(episodeRewardList[-self.printEpsFrequency:]),
                    [np.mean(rew[-self.printEpsFrequency:]) for rew in agentsEpsRewardList]))

        return meanRewardList, trajectory

