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


class BuildValueNet:
    def __init__(self, hyperparamDict):
        self.valueNetWeightInit = hyperparamDict['valueNetWeightInit']
        self.valueNetBiasInit = hyperparamDict['valueNetBiasInit']
        self.valueNetActivFunctionList = hyperparamDict['valueNetActivFunction']
        self.valueNetLayersWidths = hyperparamDict['valueNetLayersWidths']

    def __call__(self, statesInput_, scope):
        with tf.variable_scope(scope):
            layerNum = len(self.valueNetLayersWidths)
            net = statesInput_
            for i in range(layerNum):
                layerWidth = self.valueNetLayersWidths[i]
                activFunction = self.valueNetActivFunctionList[i]
                net = tf.layers.dense(net, layerWidth, activation=activFunction, kernel_initializer=self.valueNetWeightInit, bias_initializer=self.valueNetBiasInit)
            out = tf.layers.dense(net, 1, kernel_initializer=self.valueNetWeightInit, bias_initializer=self.valueNetBiasInit)

        return out


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
            out = tf.layers.dense(net, 1, kernel_initializer=self.qNetWeightInit, bias_initializer=self.qNetBiasInit)

        return out


class BuildPolicyNet:
    def __init__(self, hyperparamDict):
        self.policyWeightInit = hyperparamDict['policyWeightInit']
        self.policyBiasInit = hyperparamDict['policyBiasInit']
        self.policyActivFunctionList = hyperparamDict['policyActivFunction']
        self.policyLayersWidths = hyperparamDict['policyLayersWidths']
        self.policyMuWeightInit = hyperparamDict['policyMuWeightInit']
        self.policySDWeightInit = hyperparamDict['policySDWeightInit']

        self.policySDlow = hyperparamDict['policySDlow']
        self.policySDhigh = hyperparamDict['policySDhigh']

    def __call__(self, stateInput_, actionDim, scope):
        with tf.variable_scope(scope):
            layerNum = len(self.policyLayersWidths)
            net = stateInput_
            for i in range(layerNum):
                layerWidth = self.policyLayersWidths[i]
                activFunction = self.policyActivFunctionList[i]
                net = tf.layers.dense(net, layerWidth, activation = activFunction, kernel_initializer = self.policyWeightInit, bias_initializer= self.policyBiasInit)

            mu_ = tf.layers.dense(net, actionDim, kernel_initializer=self.policyMuWeightInit, name='mu_')
            logSigmaRaw_ = tf.layers.dense(net, actionDim, kernel_initializer=self.policySDWeightInit, name='sigma_')
            logSigma_ = tf.clip_by_value(logSigmaRaw_, self.policySDlow, self.policySDhigh)

        return mu_, logSigma_


# class DoubleQNet:
#     def __init__(self, buildQNet, numStateSpace, actionDim, session, hyperparamDict, agentID=None):
#         self.buildQNet = buildQNet
#         self.numStateSpace = numStateSpace
#         self.actionDim = actionDim
#
#         self.qNetLR = hyperparamDict['qNetLR']
#         self.tau = hyperparamDict['tau']
#         self.gamma = hyperparamDict['gamma']
#
#         self.rewardScale = hyperparamDict['rewardScale']
#
#         self.session = session
#         self.scope = 'Agent' + str(agentID) if agentID is not None else ''
#
#         with tf.variable_scope(self.scope):
#             self.states_ = tf.placeholder(tf.float32, [None, self.numStateSpace], name='states_')
#             self.actions_ = tf.placeholder(tf.float32, [None, self.actionDim], name='actions_')
#             self.qTarget_ = tf.placeholder(tf.float32, [None, 1], name='qTarget_')
#
#             q1TrainOutput_ = self.buildQNet(self.states_, self.actions_, scope = 'q1Train')
#             q1TargetOutput_ = self.buildQNet(self.states_, self.actions_, scope = 'q1Target')
#
#             q2TrainOutput_ = self.buildQNet(self.states_, self.actions_, scope = 'q2Train')
#             q2TargetOutput_ = self.buildQNet(self.states_, self.actions_, scope = 'q2Target')
#
#             with tf.variable_scope("updateParameters"):
#                 q1TrainParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + 'q1Train')
#                 q1TargetParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + 'q1Target')
#                 self.q1UpdateParam_ = [tf.assign((1 - self.tau) * targetParam + self.tau * trainParam, targetParam) for
#                                        trainParam, targetParam in zip(q1TrainParams_, q1TargetParams_)]
#                 self.q1HardReplaceTargetParam_ = [tf.assign(trainParam, targetParam) for trainParam, targetParam in
#                                                   zip(q1TrainParams_, q1TargetParams_)]
#
#                 q2TrainParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + 'q2Train')
#                 q2TargetParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + 'q2Target')
#                 self.q2UpdateParam_ = [tf.assign((1- self.tau)* targetParam + self.tau* trainParam, targetParam) for trainParam, targetParam in
#                                                       zip(q2TrainParams_, q2TargetParams_)]
#
#                 self.q2HardReplaceTargetParam_ = [tf.assign(trainParam, targetParam) for trainParam, targetParam in
#                                                       zip(q2TrainParams_, q2TargetParams_)]
#
#             with tf.variable_scope("trainDoubleQNet"):
#                 self.q1Loss_ = tf.losses.mean_squared_error(self.qTarget_, q1TrainOutput_)
#                 self.q2Loss_ = tf.losses.mean_squared_error(self.qTarget_, q2TrainOutput_)
#
#                 self.q1Optimizer = tf.train.AdamOptimizer(self.qNetLR, name='q1Optimizer')
#                 self.q2Optimizer = tf.train.AdamOptimizer(self.qNetLR, name='q2Optimizer')
#
#                 self.q1TrainOpt_ = self.q1Optimizer.minimize(self.q1Loss_, var_list=q1TrainParams_)
#                 self.q2TrainOpt_ = self.q2Optimizer.minimize(self.q2Loss_, var_list=q2TrainParams_)
#
#                 self.minQ_ = tf.minimum(q1TrainOutput_, q2TrainOutput_)
#
#             self.session.run([self.q1HardReplaceTargetParam_, self.q2HardReplaceTargetParam_])
#
#     def train(self, stateBatch, actionBatch, rewardBatch, valueTarget, done):
#         qTarget = rewardBatch* self.rewardScale + (1- done)* self.gamma * valueTarget
#         self.session.run([self.q1TrainOpt_, self.q2TrainOpt_], feed_dict={self.states_: stateBatch, self.actions_: actionBatch, self.qTarget_: qTarget})
#
#     def updateParameters(self):
#         self.session.run([self.q1UpdateParam_, self.q2UpdateParam_])
#
#     def getMinQ(self, stateBatch, actionBatch):
#         minQ = self.session.run(self.minQ_, feed_dict = {self.states_: stateBatch, self.actions_: actionBatch})
#         return minQ
class DoubleQNet:
    def __init__(self, buildQNet, numStateSpace, actionDim, session, hyperparamDict, agentID=None):
        self.buildQNet = buildQNet
        self.numStateSpace = numStateSpace
        self.actionDim = actionDim

        self.qNetLR = hyperparamDict['qNetLR']
        self.gamma = hyperparamDict['gamma']

        self.rewardScale = hyperparamDict['rewardScale']

        self.session = session
        self.scope = 'Agent' + str(agentID) if agentID is not None else ''
        self.states_ = tf.placeholder(tf.float32, [None, self.numStateSpace], name='states_')
        self.actions_ = tf.placeholder(tf.float32, [None, self.actionDim], name='actions_')

        self.reward_ = tf.placeholder(tf.float32, [None, 1], name='reward_')
        # self.done_ = tf.placeholder(tf.float32, [None, 1], name='done_')
        self.nextValueTarget_ = tf.placeholder(tf.float32, [None, 1], name='nextValueTarget_')
        # self.qTarget_ = tf.placeholder(tf.float32, [None, 1], name='qTarget_')

        with tf.variable_scope(self.scope):
            self.q1TrainOutput_ = self.buildQNet(self.states_, self.actions_, scope = 'q1Train')
            self.q2TrainOutput_ = self.buildQNet(self.states_, self.actions_, scope = 'q2Train')

            with tf.variable_scope("qNetParameters"):
                q1TrainParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + 'q1Train')
                q2TrainParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + 'q2Train')

            with tf.variable_scope("doubleQNetTrain"):
                # self.qTarget_ = self.reward_* self.rewardScale + (1 - self.done_)* self.gamma * self.nextValueTarget_
                self.qTarget_ = tf.stop_gradient(self.reward_ * self.rewardScale + self.gamma * self.nextValueTarget_)

                self.q1Loss_ = tf.losses.mean_squared_error(self.qTarget_, self.q1TrainOutput_)
                self.q2Loss_ = tf.losses.mean_squared_error(self.qTarget_, self.q2TrainOutput_)

                self.q1Optimizer = tf.train.AdamOptimizer(self.qNetLR, name='q1Optimizer')
                self.q2Optimizer = tf.train.AdamOptimizer(self.qNetLR, name='q2Optimizer')

                self.q1TrainOpt_ = self.q1Optimizer.minimize(self.q1Loss_, var_list=q1TrainParams_)
                self.q2TrainOpt_ = self.q2Optimizer.minimize(self.q2Loss_, var_list=q2TrainParams_)

                self.minQ_ = tf.minimum(self.q1TrainOutput_, self.q2TrainOutput_)

    def train(self, stateBatch, actionBatch, rewardBatch, nextValueTarget):
        self.session.run([self.q1TrainOpt_, self.q2TrainOpt_], feed_dict={
            self.states_: stateBatch, self.actions_: actionBatch, self.reward_: rewardBatch,
            self.nextValueTarget_: nextValueTarget})

    def getMinQ(self, stateBatch, actionBatch):
        minQ = self.session.run(self.minQ_, feed_dict = {self.states_: stateBatch, self.actions_: actionBatch})
        return minQ


class ValueNet:
    def __init__(self, buildValueNet, numStateSpace, actionDim, session, hyperparamDict, agentID=None):
        self.buildValueNet = buildValueNet
        self.numStateSpace = numStateSpace
        self.actionDim = actionDim

        self.valueNetLR = hyperparamDict['valueNetLR']
        self.tau = hyperparamDict['tau']

        self.rewardScale = hyperparamDict['rewardScale']

        self.session = session
        self.scope = 'Agent' + str(agentID) if agentID is not None else ''

        self.states_ = tf.placeholder(tf.float32, [None, self.numStateSpace], name='states_')
        self.nextStates_ = tf.placeholder(tf.float32, [None, self.numStateSpace], name='nextStates_')

        with tf.variable_scope(self.scope):
            self.trainValue_ = self.buildValueNet(self.states_, scope='trainValueNet')
            self.targetValue_ = self.buildValueNet(self.nextStates_, scope='targetValueNet')

            with tf.variable_scope("valueNetUpdateParameters"):
                trainValueParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + 'trainValueNet')
                targetValueParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + 'targetValueNet')
                self.valueNetUpdateParam_ = [targetValueParams_[i].assign(
                    (1 - self.tau) * targetValueParams_[i] + self.tau * trainValueParams_[i]) for i in
                    range(len(targetValueParams_))]

                self.hardReplaceTargetParam_ = [tf.assign(trainParam, targetParam) for trainParam, targetParam in
                                                  zip(trainValueParams_, targetValueParams_)]

            with tf.variable_scope("valueNetTrain"):
                self.minQ_ = tf.placeholder(tf.float32, [None, 1], name='minQ_')
                self.logPi_ = tf.placeholder(tf.float32, [None, 1], name='logPi_')
                self.valueTargetOfUpdate_ = tf.stop_gradient(self.minQ_ - self.logPi_)
                self.valueLoss_ = tf.losses.mean_squared_error(self.valueTargetOfUpdate_, self.trainValue_)
                self.valueOptimizer = tf.train.AdamOptimizer(self.valueNetLR, name='valueOptimizer')
                self.valueOpt_ = self.valueOptimizer.minimize(self.valueLoss_, var_list=trainValueParams_)

    def hardReplaceTargetParam(self):
        self.session.run(self.hardReplaceTargetParam_)

    def train(self, stateBatch, minQ, logPi):
        self.session.run(self.valueOpt_, feed_dict = {self.states_: stateBatch, self.minQ_: minQ, self.logPi_: logPi})

    def updateParams(self):
        self.session.run(self.valueNetUpdateParam_)

    def getTargetValue(self, nextStateBatch):
        targetValue = self.session.run(self.targetValue_, feed_dict = {self.nextStates_: nextStateBatch})
        return targetValue


def gaussian_likelihood(noisyAction_, mu_, logSigma_):
    """
    Helper to computer log likelihood of a gaussian.
    Here we assume this is a Diagonal Gaussian.

    :param input_: (tf.Tensor)
    :param mu_: (tf.Tensor)
    :param log_std: (tf.Tensor)
    :return: (tf.Tensor)
    """
    EPS = 1e-6 # prevent division by 0 or log0
    pre_sum = -0.5 * (((noisyAction_ - mu_) / (tf.exp(logSigma_) + EPS)) ** 2 + 2 * logSigma_ + np.log(2 * np.pi))
    return tf.reduce_sum(pre_sum, axis=1)


def apply_squashing_func(mu_, pi_, logp_pi):
    """
    Squash the output of the Gaussian distribution and account for that in the log probability
    The squashed mean is also returned for using deterministic actions.

    :param mu_: (tf.Tensor) Mean of the gaussian
    :param pi_: (tf.Tensor) Output of the policy before squashing
    :param logp_pi: (tf.Tensor) Log probability before squashing
    :return: ([tf.Tensor])
    """
    # Squash the output
    deterministic_policy = tf.tanh(mu_)
    policy = tf.tanh(pi_)
    # OpenAI Variation:
    # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
    # logp_pi -= tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - policy ** 2, lower=0, upper=1) + EPS), axis=1)
    # Squash correction (from original implementation)
    logp_pi -= tf.reduce_sum(tf.log(1 - policy ** 2 + EPS), axis=1)
    return deterministic_policy, policy, logp_pi

class PolicyNet(object):
    def __init__(self, buildPolicyNet, numStateSpace, actionDim, session, hyperparamDict, actionRange, agentID = None):
        self.buildPolicyNet = buildPolicyNet
        self.numStateSpace = numStateSpace
        self.actionDim = actionDim
        self.actionRange = actionRange

        self.policyNetLR = hyperparamDict['policyNetLR']
        self.muActivationFunc = hyperparamDict['muActivationFunc']
        self.epsilon = hyperparamDict['epsilon']
        self.actionLow, self.actionHigh = actionRange

        self.session = session
        self.scope = 'Agent'+ str(agentID) if agentID is not None else ''

        self.states_ = tf.placeholder(tf.float32, [None, self.numStateSpace], name='states_')

        with tf.variable_scope(self.scope):
            self.mu_, self.logSigma_ = self.buildPolicyNet(self.states_, actionDim, scope='policyNet')

            with tf.variable_scope("actionOutput"):
                sigma_ = tf.exp(self.logSigma_)
                normal_dist = tf.distributions.Normal(self.mu_, sigma_)

                # self.muZ_ = tf.squeeze(normal_dist.sample(1), axis=0)
                self.muZ_ = normal_dist.sample()
                self.action_ = self.muActivationFunc(self.muZ_)

                # logPi_ = normal_dist.log_prob(self.muZ_) - tf.reduce_sum(tf.log(1 - tf.pow(self.action_, 2) + self.epsilon), axis= 1)
                # self.logPi_ = tf.reduce_sum(logPi_, keep_dims= True)


                self.logPi_ = normal_dist.log_prob(self.muZ_) - tf.log(1 - tf.pow(self.action_, 2) + self.epsilon)

                self.scaledAction_ = self.action_ #* (self.actionHigh - self.actionLow)/ 2.0 + (self.actionHigh + self.actionLow)/ 2.0 # --------?

            with tf.variable_scope("policyNetParameters"):
                self.policyParam_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + 'policyNet')

            with tf.variable_scope("policyNetTrain"):
                self.minQ_ = tf.placeholder(tf.float32, [None, 1], name='minQ_')
                self.policyLoss = tf.reduce_mean(self.logPi_ - self.minQ_)
                self.policyOptimizer = tf.train.AdamOptimizer(self.policyNetLR, name='policyOptimizer')
                self.policyOpt_ = self.policyOptimizer.minimize(self.policyLoss, var_list=self.policyParam_)


        # with tf.variable_scope(self.scope):
        #     self.mu_, self.logSigma_ = self.buildPolicyNet(self.states_, numStateSpace, actionDim, actionRange,
        #                                                    scope='policyNet')
        #
        #     with tf.variable_scope("actionOutput"):
        #         sigma_ = tf.exp(self.logSigma_)
        #         normal_dist = tf.distributions.Normal(self.mu_, sigma_)
        #
        #         self.action_ = tf.squeeze(normal_dist.sample(1), axis=0)
        #         self.logPi_ = gaussian_likelihood(self.action_, self.mu_, self.logSigma_)
        #
        #         self.detAction_ = self.muActivationFunc(self.mu_)
        #         self.probAction_= self.muActivationFunc(self.action_)
        #         logp_pi = self.logPi_ - tf.reduce_sum(tf.log(1 - self.probAction_ ** 2 + EPS), axis=1)
        #
        #
        #         self.scaledAction_ = self.action_ * (self.actionHigh - self.actionLow) / 2.0 + (
        #                     self.actionHigh + self.actionLow) / 2.0  # --------?
        #
        #     with tf.variable_scope("policyNetParameters"):
        #         self.policyParam_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + 'policyNet')
        #
        #     with tf.variable_scope("policyNetTrain"):
        #         self.minQ_ = tf.placeholder(tf.float32, [None, 1], name='minQ_')
        #         self.policyLoss = tf.reduce_mean(self.logPi_ - self.minQ_)
        #         self.policyOptimizer = tf.train.AdamOptimizer(self.policyNetLR, name='policyOptimizer')
        #         self.policyOpt_ = self.policyOptimizer.minimize(self.policyLoss, var_list=self.policyParam_)

    def act(self, stateBatch):
        action = self.session.run(self.action_, feed_dict = {self.states_: stateBatch})
        return action

    def getLogPi(self, stateBatch):
        logPi = self.session.run(self.logPi_, feed_dict = {self.states_: stateBatch})
        return logPi

    def train(self, stateBatch, minQ):
        self.session.run(self.policyOpt_, feed_dict = {self.states_: stateBatch, self.minQ_: minQ})

    def actWithRescaling(self, stateBatch):
        action = self.session.run(self.scaledAction_, feed_dict = {self.states_: stateBatch})
        return action


def reshapeBatchToGetSASR(miniBatch):
    states, actions, rewards, nextStates = list(zip(*miniBatch))
    stateBatch = np.asarray(states).reshape(len(miniBatch), -1)
    actionBatch = np.asarray(actions).reshape(len(miniBatch), -1)
    nextStateBatch = np.asarray(nextStates).reshape(len(miniBatch), -1)
    rewardBatch = np.asarray(rewards).reshape(len(miniBatch), -1)

    return stateBatch, actionBatch, nextStateBatch, rewardBatch


# class TrainSoftACOneStep:
#     def __init__(self, policyNet, valueNet, qNet, reshapeBatchToGetSASR, policyUpdateInterval):
#         self.policyNet = policyNet
#         self.valueNet = valueNet
#         self.qNet = qNet
#         self.reshapeBatchToGetSASR = reshapeBatchToGetSASR
#         self.learnTime = 0
#         self.policyUpdateInterval = policyUpdateInterval
#
#     def __call__(self, miniBatch):
#         stateBatch, actionBatch, nextStateBatch, rewardBatch = self.reshapeBatchToGetSASR(miniBatch)
#         minQ = self.qNet.getMinQ(stateBatch, actionBatch)
#         logPi = self.policyNet.getLogPi(stateBatch)
#         valueTarget = self.valueNet.getTargetValue(nextStateBatch)
#
#         self.valueNet.train(stateBatch, minQ, logPi)
#         self.qNet.train(stateBatch, actionBatch, rewardBatch, valueTarget)
#
#         if self.learnTime % self.policyUpdateInterval == 0:
#             sampledAction = self.policyNet.act(stateBatch)
#             minQWithSampledAction = self.qNet.getMinQ(stateBatch, sampledAction)
#             self.policyNet.train(stateBatch, minQWithSampledAction)
#
#             self.valueNet.updateParams()
#
#         self.learnTime += 1
class TrainSoftACOneStep:
    def __init__(self, policyNet, valueNet, qNet, reshapeBatchToGetSASR, policyUpdateInterval):
        self.policyNet = policyNet
        self.valueNet = valueNet
        self.qNet = qNet
        self.reshapeBatchToGetSASR = reshapeBatchToGetSASR
        self.learnTime = 0
        self.policyUpdateInterval = policyUpdateInterval

    def __call__(self, miniBatch):
        stateBatch, actionBatch, nextStateBatch, rewardBatch = self.reshapeBatchToGetSASR(miniBatch)

        nextSampledAction = self.policyNet.act(nextStateBatch)
        nextMinQ = self.qNet.getMinQ(nextStateBatch, nextSampledAction)
        nextLogPi = self.policyNet.getLogPi(nextStateBatch)
        nextValueTarget = self.valueNet.getTargetValue(nextStateBatch)

        self.valueNet.train(stateBatch, nextMinQ, nextLogPi)

        self.qNet.train(stateBatch, actionBatch, rewardBatch, nextValueTarget)

        if self.learnTime % self.policyUpdateInterval == 0:
            sampledAction = self.policyNet.act(stateBatch)
            minQWithSampledAction = self.qNet.getMinQ(stateBatch, sampledAction)
            self.policyNet.train(stateBatch, minQWithSampledAction)

            self.valueNet.updateParams()

        self.learnTime += 1

class ActOneStep:
    def __init__(self, policyNet):
        self.policyNet = policyNet

    def __call__(self, state):
        state = np.asarray(state).reshape(1, -1)
        action = self.policyNet.actWithRescaling(state)
        return action


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






























