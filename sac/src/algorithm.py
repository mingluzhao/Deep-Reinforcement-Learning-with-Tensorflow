import tensorflow as tf
import numpy as np
import random
from collections import deque


class LearnFromBuffer:
    def __init__(self, learningStartBufferSize, trainModels, learnInterval = 1):
        self.learningStartBufferSize = learningStartBufferSize
        self.trainModels = trainModels
        self.learnInterval = learnInterval

    def __call__(self, miniBatch, runTime):
        if runTime >= self.learningStartBufferSize and runTime % self.learnInterval == 0:
            self.trainModels(miniBatch)


class MemoryBuffer(object):
    def __init__(self, size, minibatchSize):
        self.size = size
        self.buffer = self.reset()
        self.minibatchSize = minibatchSize

    def reset(self):
        return deque(maxlen=int(self.size))

    def add(self, observation, action, reward, nextObservation):
        self.buffer.append((observation, action, reward, nextObservation))

    def sample(self):
        if len(self.buffer) < self.minibatchSize:
            return []
        sampleIndex = [random.randint(0, len(self.buffer) - 1) for _ in range(self.minibatchSize)]
        sample = [self.buffer[index] for index in sampleIndex]

        return sample


EPS = 1e-8
def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)


class BuildQNet:
    def __init__(self, hyperparamDict):
        self.qNetWeightInit = hyperparamDict['qNetWeightInit']
        self.qNetBiasInit = hyperparamDict['qNetBiasInit']
        self.qNetActivFunctionList = hyperparamDict['qNetActivFunction']
        self.qNetLayersWidths = hyperparamDict['qNetLayersWidths']

    def __call__(self, statesInput_, actionsInput_, scope):
        with tf.variable_scope(scope):
            layerNum = len(self.qNetLayersWidths)
            net = tf.concat([statesInput_, actionsInput_], axis=1)
            for i in range(layerNum):
                layerWidth = self.qNetLayersWidths[i]
                activFunction = self.qNetActivFunctionList[i]
                net = tf.layers.dense(net, layerWidth, activation=activFunction, kernel_initializer=self.qNetWeightInit,
                                      bias_initializer=self.qNetBiasInit)
            out = tf.layers.dense(net, 1, activation = None, kernel_initializer=self.qNetWeightInit, bias_initializer=self.qNetBiasInit)

        return out

def apply_squashing_func(mu, pi, logp_pi):
    logp_pi -= tf.reduce_sum(2*(np.log(2) - pi - tf.nn.softplus(-2*pi)), axis=1)
    mu = tf.tanh(mu)
    pi = tf.tanh(pi)
    return mu, pi, logp_pi


class BuildPolicyNet:
    def __init__(self, hyperparamDict,actionRange):
        self.policyWeightInit = hyperparamDict['policyWeightInit']
        self.policyBiasInit = hyperparamDict['policyBiasInit']
        self.policyActivFunctionList = hyperparamDict['policyActivFunction']
        self.policyLayersWidths = hyperparamDict['policyLayersWidths']
        self.policyMuWeightInit = hyperparamDict['policyMuWeightInit']
        self.policySDWeightInit = hyperparamDict['policySDWeightInit']
        self.policyMuBiasInit = hyperparamDict['policyMuBiasInit']
        self.policySDBiasInit = hyperparamDict['policySDBiasInit']
        self.muActivationFunc = hyperparamDict['muActivationFunc']

        self.policySDlow = hyperparamDict['policySDlow']
        self.policySDhigh = hyperparamDict['policySDhigh']

        self.actionRange = actionRange
        self.actionLow, self.actionHigh = actionRange

    def __call__(self, stateInput_, actionDim, scope):
        with tf.variable_scope(scope):
            layerNum = len(self.policyLayersWidths)
            net = stateInput_
            for i in range(layerNum):
                layerWidth = self.policyLayersWidths[i]
                activFunction = self.policyActivFunctionList[i]
                net = tf.layers.dense(net, layerWidth, activation = activFunction, kernel_initializer = self.policyWeightInit, bias_initializer= self.policyBiasInit)

            mu_ = tf.layers.dense(net, actionDim, activation = self.muActivationFunc, kernel_initializer=self.policyMuWeightInit, bias_initializer = self.policyMuBiasInit, name='mu_')
            logSigmaRaw_ = tf.layers.dense(net, actionDim, activation = None, kernel_initializer=self.policySDWeightInit, bias_initializer = self.policySDBiasInit, name='sigma_')
            logSigma_ = tf.clip_by_value(logSigmaRaw_, self.policySDlow, self.policySDhigh)

            sigma_ = tf.exp(logSigma_)
            pi_ = mu_ + tf.random_normal(tf.shape(mu_)) * sigma_
            logpi_ = gaussian_likelihood(pi_, mu_, logSigma_)

            # squash
            self.muRaw_, self.piRaw_, self.logPi_ = apply_squashing_func(mu_, pi_, logpi_)

            #scale
            self.mu_ = self.muRaw_ * self.actionHigh
            self.action_ = self.piRaw_ * self.actionHigh

        return self.mu_, self.action_, self.logPi_


class SACAgent:
    def __init__(self, buildQNet, buildPolicyNet, numStateSpace, actionDim, session, hyperparamDict, agentID=None):
        self.buildQNet = buildQNet
        self.buildPolicyNet = buildPolicyNet
        self.numStateSpace = numStateSpace
        self.actionDim = actionDim

        self.qNetLR = hyperparamDict['qNetLR']
        self.policyNetLR = hyperparamDict['policyNetLR']
        self.tau = hyperparamDict['tau']
        self.gamma = hyperparamDict['gamma']
        self.alpha = hyperparamDict['alpha']

        self.session = session
        self.scope = 'Agent' + str(agentID) if agentID is not None else ''
        self.states_ = tf.placeholder(tf.float32, [None, self.numStateSpace], name='states_')
        self.actions_ = tf.placeholder(tf.float32, [None, self.actionDim], name='actions_')
        self.nextStates_ = tf.placeholder(tf.float32, [None, self.numStateSpace], name='nextStates_')
        self.reward_ = tf.placeholder(tf.float32, [None, 1], name='reward_')

        with tf.variable_scope('train'):
            self.q1TrainOutput_ = self.buildQNet(self.states_, self.actions_, scope = 'q1Train')
            self.q2TrainOutput_ = self.buildQNet(self.states_, self.actions_, scope = 'q2Train')
            self.mu_, self.actionOutput_, self.logPi_ = self.buildPolicyNet(self.states_, self.actionDim, scope = 'policyTrain')

        with tf.variable_scope('train', reuse= True):
            self.q1TrainGivenAction_ = self.buildQNet(self.states_, self.actionOutput_, scope = 'q1Train')
            self.q2TrainGivenAction_ = self.buildQNet(self.states_, self.actionOutput_, scope = 'q2Train')
            self.muNext_, self.actionOutputNext_, self.logPiNext_ = self.buildPolicyNet(self.nextStates_, self.actionDim, scope = 'policyTrain')

            self.minQTrainGivenAction_ = tf.minimum(self.q1TrainGivenAction_, self.q2TrainGivenAction_)

        with tf.variable_scope('target'):
            self.q1Target_ = self.buildQNet(self.nextStates_, self.actionOutputNext_, scope = 'q1Target')
            self.q2Target_ = self.buildQNet(self.nextStates_, self.actionOutputNext_, scope = 'q2Target')

            self.minQTargetGivenAction_ = tf.minimum(self.q1Target_, self.q2Target_)

        with tf.variable_scope('trainLoss'):
            self.policyLoss_ = tf.reduce_mean(self.alpha * self.logPi_ - self.minQTrainGivenAction_)

            self.qTargetOfUpdate_ = tf.stop_gradient(self.reward_ + self.gamma * (self.minQTargetGivenAction_ - self.alpha * self.logPiNext_))
            self.q1Loss_ = 0.5 * tf.reduce_mean((self.qTargetOfUpdate_ - self.q1TrainOutput_) ** 2)
            self.q2Loss_ = 0.5 * tf.reduce_mean((self.qTargetOfUpdate_ - self.q2TrainOutput_) ** 2)
            self.qLoss_ = self.q1Loss_ + self.q2Loss_

        with tf.variable_scope("trainParameters"):
            self.policyParam_ = [x for x in tf.global_variables() if 'train/policyTrain' in x.name]
            self.qTrainParam_ = [x for x in tf.global_variables() if 'train/q' in x.name]
            self.qTargetParam_ = [x for x in tf.global_variables() if 'target/q' in x.name]

        with tf.variable_scope('trainNetworks'):
            self.policyOptimizer = tf.train.AdamOptimizer(self.policyNetLR, name='policyOptimizer')
            self.policyOpt_ = self.policyOptimizer.minimize(self.policyLoss_, var_list=self.policyParam_)

            self.qOptimizer = tf.train.AdamOptimizer(self.qNetLR, name='qOptimizer')
            self.qOpt_ = self.qOptimizer.minimize(self.qLoss_, var_list=self.qTrainParam_)

        with tf.variable_scope('replaceParams'):
            self.updateParam_ = tf.group([tf.assign(targetParam, (1 - self.tau) * targetParam + self.tau * trainParam) for targetParam, trainParam in zip(self.qTargetParam_, self.qTrainParam_)])
            self.hardReplaceParam_ = tf.group([tf.assign(targetParam, trainParam) for targetParam, trainParam in zip(self.qTargetParam_, self.qTrainParam_)])

    def reset(self):
        self.session.run(self.hardReplaceParam_)

    def act(self, stateBatch, determ = False):
        action_ = self.mu_ if determ else self.actionOutput_
        action = self.session.run(action_, feed_dict = {self.states_: stateBatch})
        return action

    def train(self, stateBatch, actionBatch, nextStateBatch, rewardBatch):
        # policy train before value train
        self.session.run(self.policyOpt_, feed_dict = {self.states_: stateBatch, self.actions_: actionBatch, self.reward_: rewardBatch, self.nextStates_: nextStateBatch})
        self.session.run(self.qOpt_, feed_dict = {self.states_: stateBatch, self.actions_: actionBatch, self.reward_: rewardBatch, self.nextStates_: nextStateBatch})

    def updateParameters(self):
        self.session.run(self.updateParam_)



def reshapeBatchToGetSASR(miniBatch):
    states, actions, rewards, nextStates = list(zip(*miniBatch))
    stateBatch = np.asarray(states).reshape(len(miniBatch), -1)
    actionBatch = np.asarray(actions).reshape(len(miniBatch), -1)
    nextStateBatch = np.asarray(nextStates).reshape(len(miniBatch), -1)
    rewardBatch = np.asarray(rewards).reshape(len(miniBatch), -1)

    return stateBatch, actionBatch, nextStateBatch, rewardBatch


class TrainSoftACOneStep:
    def __init__(self, sacAgent, reshapeBatchToGetSASR, policyUpdateInterval):
        self.sacAgent = sacAgent
        self.reshapeBatchToGetSASR = reshapeBatchToGetSASR
        self.learnTime = 0
        self.policyUpdateInterval = policyUpdateInterval

    def __call__(self, miniBatch):
        stateBatch, actionBatch, nextStateBatch, rewardBatch = self.reshapeBatchToGetSASR(miniBatch)
        self.sacAgent.train(stateBatch, actionBatch, nextStateBatch, rewardBatch)

        if self.learnTime % self.policyUpdateInterval == 0:
            self.sacAgent.updateParameters()

        self.learnTime += 1


class TrainSoftAC:
    def __init__(self, maxEpisode, maxTimeStep, memoryBuffer, actOneStep, learnFromBuffer, env, saveModel):
        self.maxEpisode = maxEpisode
        self.maxTimeStep = maxTimeStep
        self.memoryBuffer = memoryBuffer
        self.actOneStep = actOneStep
        self.env = env
        self.learnFromBuffer = learnFromBuffer
        self.saveModel = saveModel
        self.runTime = 0

    def __call__(self):
        episodeRewardList = []
        for episodeID in range(self.maxEpisode):
            state = self.env.reset()
            state = state.reshape(1, -1)
            epsReward = 0

            for timeStep in range(self.maxTimeStep):
                action = self.actOneStep(state)
                nextState, reward, terminal, info = self.env.step(action)
                nextState = nextState.reshape(1, -1)
                self.memoryBuffer.add(state, action, reward, nextState)
                epsReward += reward

                miniBatch= self.memoryBuffer.sample()
                self.learnFromBuffer(miniBatch, self.runTime)
                state = nextState
                self.runTime += 1

                if terminal:
                    break

            self.saveModel()
            episodeRewardList.append(epsReward)
            last100EpsMeanReward = np.mean(episodeRewardList[-1000: ])
            if episodeID % 1 == 0:
                print('episode: {}, last 1000eps mean reward: {}, last eps reward: {} with {} steps'.format(episodeID, last100EpsMeanReward, epsReward, timeStep))

        return episodeRewardList



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






























