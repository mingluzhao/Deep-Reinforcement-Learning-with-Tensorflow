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


# class UpdateParameters:
#     def __init__(self, paramUpdateInterval, tau = None):
#         self.paramUpdateInterval = paramUpdateInterval
#         self.tau = tau
#         self.runTime = 0
#
#     def __call__(self, model):
#         if self.runTime % self.paramUpdateInterval == 0:
#             graph = model.graph
#             updateParam_ = graph.get_collection_ref("updateParam_")[0]
#             if self.tau is not None:
#                 tau_ = graph.get_collection_ref("tau_")[0]
#                 model.run(updateParam_, feed_dict={tau_: self.tau})
#             else:
#                 model.run(updateParam_)
#         self.runTime += 1
#
#         return model

class UpdateParameters:
    def __init__(self, paramUpdateInterval, tau = None):
        self.paramUpdateInterval = paramUpdateInterval
        self.tau = tau
        self.runTime = 0

    def __call__(self, model):
        graph = model.graph
        updateParam_ = graph.get_collection_ref("updateParam_")[0]
        if self.tau is not None:
            tau_ = graph.get_collection_ref("tau_")[0]
            model.run(updateParam_, feed_dict={tau_: self.tau})
        else:
            model.run(updateParam_)

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
        # sample = random.sample(memoryBuffer, self.minibatchSize)

        return sample


class LearnFromBuffer:
    def __init__(self, learningStartBufferSize, sampleFromMemory, trainModels):
        self.learningStartBufferSize = learningStartBufferSize
        self.sampleFromMemory = sampleFromMemory
        self.trainModels = trainModels

    def __call__(self, replayBuffer, runTime):
        if runTime >= self.learningStartBufferSize:
            # print('learn-------------------------------------')
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
        print('state: ', state, ', action: ', action, ', reward: ', reward)

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
        self.notTerminalCount = 0

    def __call__(self, replayBuffer, trajectory):
        state = self.reset()
        episodeReward = np.zeros(2)
        for timeStep in range(self.maxTimeStep):
            reward, state, replayBuffer, trajectory = self.runTimeStep(state, replayBuffer, trajectory)
            episodeReward += np.array(reward)
            terminal = self.isTerminal(state)
            if terminal:
                break
        # print('episodeReward: ', episodeReward, 'runSteps: ', len(trajectory))

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
            episodeRewardList.append(np.sum(episodeReward))

            if print and episode % 1000 == 0:
                lastTimeSpanMeanReward = np.mean(episodeRewardList[-1000:])
                meanRewardList.append(lastTimeSpanMeanReward)
                print('episodes: ', episode, 'mean episode reward: ', lastTimeSpanMeanReward)

        return meanRewardList, trajectory



class RunMultiAgentTimeStep:
    def __init__(self, actOneStep, sampleOneStep, learnFromBuffer, multiagent = False, observe = None):
        self.actOneStep = actOneStep
        self.sampleOneStep = sampleOneStep
        self.learnFromBuffer = learnFromBuffer
        self.multiagent = multiagent
        self.observe = observe

    def __call__(self, state, replayBuffer, trajectory):
        runTime = len(trajectory)
        observation = self.observe(state) if self.observe is not None else state
        action = self.actOneStep(observation, runTime)
        reward, nextState = self.sampleOneStep(state, action)
        # print("reward ", reward)

        nextObservation = self.observe(nextState) if self.observe is not None else nextState
        replayBuffer.append((observation, action, reward, nextObservation))
        trajectory.append((state, action, reward, nextState))

        if runTime % 100 == 0:
            if self.multiagent:
                print('learn')
                getAgentBuffer = lambda buffer, id: [[bufferElement[id] for bufferElement in timeStepBuffer] for timeStepBuffer in buffer]
                for id, agentLearn in enumerate(self.learnFromBuffer):
                    agentBuffer = getAgentBuffer(replayBuffer, id)
                    agentLearn(agentBuffer, runTime)
            else:
                self.learnFromBuffer(replayBuffer, runTime)

        return reward, nextState, replayBuffer, trajectory


