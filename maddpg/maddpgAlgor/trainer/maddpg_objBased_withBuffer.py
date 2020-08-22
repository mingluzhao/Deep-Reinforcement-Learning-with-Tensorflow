import tensorflow as tf
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tensorflow.contrib.layers as layers
import maddpg.maddpgAlgor.common.tf_util as U
import random
from collections import deque


class BuildActorLayers:
    def __init__(self, layersWidths, actionDim, actionRange = 1):
        self.actionRange = actionRange
        self.actionDim = actionDim
        self.layersWidths = layersWidths

    def __call__(self, currentAgentState_, scope):
        with tf.variable_scope(scope):
            actorActivation_ = currentAgentState_

            for layerWidth in self.layersWidths:
                actorActivation_ = layers.fully_connected(actorActivation_, num_outputs= layerWidth, activation_fn=tf.nn.relu)

            actorActivation_ = layers.fully_connected(actorActivation_, num_outputs= self.actionDim, activation_fn= None)

            trainAction_ = tf.multiply(actorActivation_, self.actionRange, name='trainAction_')

            sampleNoise_ = tf.random_uniform(tf.shape(trainAction_))
            noisyAction_ = U.softmax(trainAction_ - tf.log(-tf.log(sampleNoise_)), axis=-1) # give this to q input

        return actorActivation_, trainAction_, noisyAction_


class BuildCriticLayers:
    def __init__(self, layersWidths):
        self.layersWidths = layersWidths

    def __call__(self, allAgentsStates_, allAgentsActions_, scope):
        with tf.variable_scope(scope):
            criticActivation_ = tf.concat(allAgentsStates_ + allAgentsActions_, axis=1)
            for layerWidth in self.layersWidths:
                criticActivation_ = layers.fully_connected(criticActivation_, num_outputs=layerWidth, activation_fn=tf.nn.relu)

            criticActivation_ = layers.fully_connected(criticActivation_, num_outputs=1, activation_fn=None)
        return criticActivation_


class Actor:
    def __init__(self, agentObsDim, buildActorLayers, tau, actorLR, session, agentID):
        self.buildActorLayers = buildActorLayers
        self.agentObsDim = agentObsDim
        self.tau = tau
        self.actorLR = actorLR
        self.gradNormClipping = 0.5
        self.session = session

        agentStr = 'Agent'+ str(agentID)

        with tf.variable_scope(agentStr):
            self.currentAgentState_ = tf.placeholder(dtype=tf.float32, shape=[None, self.agentObsDim])
            self.currentAgentNextState_ = tf.placeholder(dtype=tf.float32, shape=[None, self.agentObsDim])
            self.criticTrainActivation_ = tf.placeholder(dtype=tf.float32, shape=[None, 1])

            self.actorTrainActivation_, self.trainAction_, self.noisyTrainAction_ = self.buildActorLayers(self.currentAgentState_, scope = 'actorTrain')
            self.actorTargetActivation_, self.targetAction_, self.noisyTargetAction_ = self.buildActorLayers(self.currentAgentNextState_, scope = 'actorTarget')

            with tf.variable_scope("updateParameters"):
                actorTrainParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=agentStr + '/actorTrain')
                actorTargetParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=agentStr + '/actorTarget')
                self.actorUpdateParam_ = [actorTargetParams_[i].assign((1 - self.tau) * actorTargetParams_[i] + self.tau * actorTrainParams_[i]) for i in range(len(actorTargetParams_))]

            with tf.variable_scope("train"):
                trainQ = self.criticTrainActivation_[:, 0]
                pg_loss = -tf.reduce_mean(trainQ)
                p_reg = tf.reduce_mean(tf.square(self.actorTrainActivation_))
                actorLoss_ = pg_loss + p_reg * 1e-3

                actorOptimizer = tf.train.AdamOptimizer(self.actorLR, name='actorOptimizer')
                self.actorTrainOpt_ = U.minimize_and_clip(actorOptimizer, actorLoss_, actorTrainParams_, self.gradNormClipping)

    def actByTrainNoisy(self, agentObservation):
        agentAction = self.session.run(self.noisyTrainAction_, feed_dict = {self.currentAgentState_: agentObservation})
        return agentAction

    def actByTargetNoisy(self, agentNextObservation):
        agentAction = self.session.run(self.noisyTargetAction_, feed_dict = {self.currentAgentNextState_: agentNextObservation})
        return agentAction

    def train(self, agentObservation, criticTrainActivation):
        self.session.run(self.actorTrainOpt_, feed_dict = {self.currentAgentState_: agentObservation, self.criticTrainActivation_: criticTrainActivation})

    def updateParams(self):
        self.session.run(self.actorUpdateParam_)

    def actOneStep(self, agentState):
        agentState = np.asarray(agentState).reshape(1, -1)
        action = self.actByTrainNoisy(agentState)[0]
        return action



class Critic:
    def __init__(self, actionDim, numAgents, obsShapeList, buildCriticLayers, tau, criticLR, gamma, session, agentID):
        self.actionDim = actionDim
        self.numAgents = numAgents
        self.obsShapeList = obsShapeList
        self.gradNormClipping = 0.5
        self.buildCriticLayers = buildCriticLayers
        self.tau = tau
        self.criticLR = criticLR
        self.session = session
        self.gamma = gamma

        agentStr = 'Agent'+ str(agentID)

        with tf.variable_scope(agentStr):
            self.allAgentsStates_ = [tf.placeholder(dtype=tf.float32, shape=[None, agentObsDim]) for agentObsDim in self.obsShapeList]
            self.allAgentsNextStates_ = [tf.placeholder(dtype=tf.float32, shape=[None, agentObsDim]) for agentObsDim in self.obsShapeList]
            self.allAgentsActions_ = [tf.placeholder(dtype=tf.float32, shape=[None, self.actionDim]) for i in range(self.numAgents)]
            self.allAgentsNextActions_ = [tf.placeholder(dtype=tf.float32, shape=[None, self.actionDim]) for i in range(self.numAgents)]
            self.agentReward_ = tf.placeholder(tf.float32, [None, 1])

            self.criticTrainActivation_ = self.buildCriticLayers(self.allAgentsStates_, self.allAgentsActions_, scope='criticTrain')
            self.criticTargetActivation_ = self.buildCriticLayers(self.allAgentsNextStates_, self.allAgentsNextActions_, scope='criticTarget')

            with tf.variable_scope("updateParameters"):
                criticTrainParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=agentStr + '/criticTrain')
                criticTargetParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=agentStr + '/criticTarget')
                self.criticUpdateParam_ = [criticTargetParams_[i].assign((1 - self.tau) * criticTargetParams_[i] + self.tau * criticTrainParams_[i]) for i in range(len(criticTargetParams_))]

            with tf.variable_scope("train"):
                yi_ = self.agentReward_ + self.gamma * self.criticTargetActivation_
                yi_ = yi_[:, 0]
                qTrain = self.criticTrainActivation_[:, 0]
                criticLoss_ = tf.reduce_mean(tf.square(qTrain - yi_))

                criticOptimizer = tf.train.AdamOptimizer(self.criticLR, name='criticOptimizer')
                self.crticTrainOpt_ = U.minimize_and_clip(criticOptimizer, criticLoss_, criticTrainParams_, self.gradNormClipping)

    def getActivationWithAgentTrainAction(self, allAgentsStatesBatch, allAgentsActionsBatch, agentAction, agentID):
        criticInputActionList = allAgentsActionsBatch
        criticInputActionList[agentID] = agentAction
        stateDict = {agentState_: agentState for agentState_, agentState in zip(self.allAgentsStates_, allAgentsStatesBatch)}
        actionDict = {agentAction_: agentAction for agentAction_, agentAction in zip(self.allAgentsActions_, criticInputActionList)}
        criticActiv = self.session.run(self.criticTrainActivation_, feed_dict={**stateDict, **actionDict})
        return criticActiv

    def train(self, agentReward, allAgentsStatesBatch, allAgentsActionsBatch, allAgentsNextStateBatch, allAgentsNextStateTargetActions):
        agentReward = np.array(agentReward).reshape(-1, 1)
        stateDict = {agentState_: agentState for agentState_, agentState in zip(self.allAgentsStates_, allAgentsStatesBatch)}
        actionDict = {agentAction_: agentAction for agentAction_, agentAction in zip(self.allAgentsActions_, allAgentsActionsBatch)}
        nextStateDict = {agentNextState_: agentNextState for agentNextState_, agentNextState in zip(self.allAgentsNextStates_, allAgentsNextStateBatch)}
        nextActionDict = {agentNextAction_: agentNextAction for agentNextAction_, agentNextAction in zip(self.allAgentsNextActions_, allAgentsNextStateTargetActions)}
        self.session.run(self.crticTrainOpt_, feed_dict={**stateDict, **nextStateDict, **nextActionDict, **actionDict, self.agentReward_: agentReward})

    def updateParams(self):
        self.session.run(self.criticUpdateParam_)


class MADDPGAgent:
    def __init__(self, actor, critic, buffer, agentID, startLearn):
        self.actor = actor
        self.critic = critic
        self.buffer = buffer
        self.agentID = agentID
        self.startLearn = startLearn

        self.runTime = 0

    def experience(self, observation, action, reward, nextObservation):
        self.buffer.add(observation, action, reward, nextObservation)

    def train(self, agents):
        self.runTime += 1
        if not self.startLearn(self.runTime):
            return

        sampleIndex = self.buffer.getSampleIndex()

        allAgentsStatesBatch = []
        allAgentsNextStateBatch = []
        allAgentsActionsBatch = []

        for agent in agents:
            obs, act, rew, obs_next = agent.buffer.getSampleFromIndex(sampleIndex)
            allAgentsStatesBatch.append(obs)
            allAgentsNextStateBatch.append(obs_next)
            allAgentsActionsBatch.append(act)
        agentObservation, agentActionFromBatch, agentReward, agentNextObservation = self.buffer.getSampleFromIndex(sampleIndex)

        # train q net
        allAgentsNextStateTargetActions =[agent.actor.actByTargetNoisy(agentNextObservation) for agent, agentNextObservation in zip(agents, allAgentsNextStateBatch)]
        self.critic.train(agentReward, allAgentsStatesBatch, allAgentsActionsBatch, allAgentsNextStateBatch, allAgentsNextStateTargetActions)

        agentAction = self.actor.actByTrainNoisy(agentObservation)
        criticTrainActivation = self.critic.getActivationWithAgentTrainAction(allAgentsStatesBatch, allAgentsActionsBatch, agentAction, self.agentID)
        self.actor.train(agentObservation, criticTrainActivation)

        self.actor.updateParams()
        self.critic.updateParams()


class MemoryBuffer(object):
    def __init__(self, size, minibatchSize):
        self.size = size
        self.buffer = self.reset()
        self.minibatchSize = minibatchSize

    def getSampleIndex(self):
        sampleIndex = [random.randint(0, len(self.buffer) - 1) for _ in range(self.minibatchSize)]
        return sampleIndex

    def getSampleFromIndex(self, sampleIndex):
        sample = [self.buffer[index] for index in sampleIndex]
        obs, act, rew, obs_next = list(zip(*sample))
        return obs, act, rew, obs_next

    def reset(self):
        return deque(maxlen=int(self.size))

    def add(self, observation, action, reward, nextObservation):
        self.buffer.append((observation, action, reward, nextObservation))


class MADDPG:
    def __init__(self, agents, observe, sampleOneStep, reset, maxTimeStep, maxEpisode, saveModel):
        self.agents = agents
        self.observe = observe
        self.sampleOneStep = sampleOneStep
        self.reset = reset
        self.maxEpisode = maxEpisode
        self.maxTimeStep = maxTimeStep
        self.saveModel = saveModel
        self.printEpsFrequency = 1000
        self.numAgents = len(agents)

    def resetBuffer(self):
        for agent in self.agents:
            agent.buffer.reset()

    def allActOneStep(self, observation):
        actions = [agent.actor.actOneStep(agentObs) for agent, agentObs in zip(self.agents, observation)]
        return actions

    def act(self, state):
        observation = self.observe(state)
        actions = self.allActOneStep(observation)
        rewards, nextState = self.sampleOneStep(state, actions)
        nextObservation = self.observe(nextState)
        return observation, actions, rewards, nextObservation, nextState

    def runTraining(self):
        episodeRewardList = []
        meanRewardList = []
        agentsEpsRewardList = [list() for agentID in range(self.numAgents)]

        for episodeID in range(self.maxEpisode):
            epsReward = np.zeros(len(self.agents))
            state = self.reset()
            for timeStep in range(self.maxTimeStep):
                observation, actions, rewards, nextObservation, state = self.act(state)
                for i, agent in enumerate(self.agents):
                    agent.experience(observation[i], actions[i], rewards[i], nextObservation[i])

                for agent in self.agents:
                    agent.train(self.agents)

                epsReward += np.array(rewards)
            self.saveModel()

            episodeRewardList.append(np.sum(epsReward))
            [agentRewardList.append(agentEpsReward) for agentRewardList, agentEpsReward in zip(agentsEpsRewardList, epsReward)]
            meanRewardList.append(np.mean(episodeRewardList))

            if episodeID % self.printEpsFrequency == 0:
                lastTimeSpanMeanReward = np.mean(episodeRewardList[-self.printEpsFrequency:])

                print("episode: {}, last {} eps mean episode reward: {}, agent mean reward: {}".format(
                    episodeID, self.printEpsFrequency, lastTimeSpanMeanReward,
                    [np.mean(rew[-self.printEpsFrequency:]) for rew in agentsEpsRewardList]))

        return


class SaveModel:
    def __init__(self, modelSaveRate, saveVariables, modelSavePath, sess, saveAllmodels = False):
        self.modelSaveRate = modelSaveRate
        self.saveVariables = saveVariables
        self.epsNum = 1
        self.modelSavePath = modelSavePath
        self.saveAllmodels = saveAllmodels
        self.sess = sess

    def __call__(self):
        self.epsNum += 1
        if self.epsNum % self.modelSaveRate == 0:
            modelSavePathToUse = self.modelSavePath + str(self.epsNum) + "eps" if self.saveAllmodels else self.modelSavePath
            with self.sess.as_default():
                self.saveVariables(self.sess, modelSavePathToUse)


