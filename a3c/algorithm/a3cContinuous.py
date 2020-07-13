import tensorflow as tf
import numpy as np
import time

class GlobalNet(object):
    def __init__(self, stateDim, actionDim, hyperparamDict):
        self.stateDim = stateDim
        self.actionDim = actionDim
        self.actorActivFunction = hyperparamDict['actorActivFunction']
        self.actorMuOutputActiv = hyperparamDict['actorMuOutputActiv']
        self.actorSigmaOutputActiv = hyperparamDict['actorSigmaOutputActiv']
        self.criticActivFunction = hyperparamDict['criticActivFunction']
        self.weightInit = hyperparamDict['weightInit']
        self.actorLayersWidths = hyperparamDict['actorLayersWidths']
        self.criticLayersWidths = hyperparamDict['criticLayersWidths']
        self.actorLR = hyperparamDict['actorLR']
        self.criticLR = hyperparamDict['criticLR']
        self.scope = 'global'

        with tf.variable_scope(self.scope):
            with tf.variable_scope("inputs"):
                self.states_ = tf.placeholder(tf.float32, [None, self.stateDim], name='states_')

            with tf.variable_scope('actorNet'):
                for numUnits in self.actorLayersWidths:
                    actorTrainActivation_ = tf.layers.dense(self.states_, numUnits, self.actorActivFunction, kernel_initializer=self.weightInit)
                mu_ = tf.layers.dense(actorTrainActivation_, self.actionDim, self.actorMuOutputActiv, kernel_initializer=self.weightInit, name='mu_')
                sigma_ = tf.layers.dense(actorTrainActivation_, self.actionDim, self.actorSigmaOutputActiv, kernel_initializer=self.weightInit, name='sigma_')

            with tf.variable_scope('criticNet'):
                for numUnits in self.criticLayersWidths:
                    criticTrainActivation_ = tf.layers.dense(self.states_, numUnits, self.criticActivFunction, kernel_initializer=self.weightInit)
                value_ = tf.layers.dense(criticTrainActivation_, 1, kernel_initializer=self.weightInit)  # state value

            with tf.variable_scope('parameters'):
                self.actorParam_ = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/actorNet')
                self.criticParam_ = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/criticNet')


class WorkerNet(object):
    def __init__(self, stateDim, actionDim, hyperparamDict, actionRange, scope, globalModel, session):
        self.stateDim = stateDim
        self.actionDim = actionDim
        self.actorActivFunction = hyperparamDict['actorActivFunction']
        self.actorMuOutputActiv = hyperparamDict['actorMuOutputActiv']
        self.actorSigmaOutputActiv = hyperparamDict['actorSigmaOutputActiv']
        self.criticActivFunction = hyperparamDict['criticActivFunction']
        self.weightInit = hyperparamDict['weightInit']
        self.actorLayersWidths = hyperparamDict['actorLayersWidths']
        self.criticLayersWidths = hyperparamDict['criticLayersWidths']
        self.actorLR = hyperparamDict['actorLR']
        self.criticLR = hyperparamDict['criticLR']
        self.entropyBeta = hyperparamDict['entropyBeta']

        self.actionLow, self.actionHigh = actionRange
        self.globalScope = 'global'
        self.scope = scope
        self.session = session

        with tf.variable_scope(self.scope):
            with tf.variable_scope("inputs"):
                self.states_ = tf.placeholder(tf.float32, [None, self.stateDim], name='states_')
                self.action_ = tf.placeholder(tf.float32, [None, self.actionDim], name='actions_')
                self.valueTarget_ = tf.placeholder(tf.float32, [None, 1], 'valueTarget_')

            with tf.variable_scope('actorNet'):
                for numUnits in self.actorLayersWidths:
                    actorTrainActivation_ = tf.layers.dense(self.states_, numUnits, self.actorActivFunction, kernel_initializer=self.weightInit)

                mu_ = tf.layers.dense(actorTrainActivation_, self.actionDim, self.actorMuOutputActiv, kernel_initializer=self.weightInit, name='mu_')
                sigma_ = tf.layers.dense(actorTrainActivation_, self.actionDim, self.actorSigmaOutputActiv, kernel_initializer=self.weightInit, name='sigma_')

            with tf.variable_scope('criticNet'):
                for numUnits in self.criticLayersWidths:
                    criticTrainActivation_ = tf.layers.dense(self.states_, numUnits, self.criticActivFunction, kernel_initializer=self.weightInit)

                self.value_ = tf.layers.dense(criticTrainActivation_, 1, kernel_initializer=self.weightInit, name='value_')  # state value

            with tf.variable_scope('adjustParams'):
                mu_ = mu_ * self.actionHigh
                sigma_ = sigma_ + 1e-4

            with tf.variable_scope('output'):
                normal_dist = tf.distributions.Normal(mu_, sigma_)
                self.actionOutput_ = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=[0, 1]), self.actionLow, self.actionHigh)

            with tf.variable_scope('loss'):
                tdError_ = tf.subtract(self.valueTarget_, self.value_, name='tdError_')
                criticLoss_ = tf.reduce_mean(tf.square(tdError_))

                log_prob = normal_dist.log_prob(self.action_)
                actionVal_ = log_prob * tf.stop_gradient(tdError_)
                entropy = normal_dist.entropy()  # encourage exploration
                actionValWithEntropy_ = self.entropyBeta * entropy + actionVal_
                actorLoss_ = tf.reduce_mean(-actionValWithEntropy_)

            with tf.variable_scope('parameters'):
                self.actorParam_ = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope= self.scope + '/actorNet')
                self.criticParam_ = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope= self.scope + '/criticNet')

            with tf.variable_scope('gradients'):
                self.actorGradients_ = tf.gradients(actorLoss_, self.actorParam_)
                self.criticGradients_ = tf.gradients(criticLoss_, self.criticParam_)

            with tf.variable_scope("trainOpt"):
                self.actorOpt_ = tf.train.RMSPropOptimizer(self.actorLR, name=self.scope + '/RMSPropOptActor')
                self.criticOpt_ = tf.train.RMSPropOptimizer(self.criticLR, name= self.scope + '/RMSPropOptCritic')

        with tf.variable_scope('sync'):
            with tf.variable_scope("pullFromGlobal"):
                self.pullActorParams_ = [workerParam.assign(globalParam) for workerParam, globalParam in zip(self.actorParam_, globalModel.actorParam_)]
                self.pullCriticParams_ = [workerParam.assign(globalParam) for workerParam, globalParam in zip(self.criticParam_, globalModel.criticParam_)]

            with tf.variable_scope('pushToGlobal'):
                self.updateActor_ = self.actorOpt_.apply_gradients(zip(self.actorGradients_, globalModel.actorParam_))
                self.updateCritic_ = self.criticOpt_.apply_gradients(zip(self.criticGradients_, globalModel.criticParam_))


    def act(self, state):
        state = state[np.newaxis, :]
        action = self.session.run(self.actionOutput_, feed_dict={self.states_: state})
        return action

    def pullFromGlobalNet(self):
        self.session.run([self.pullActorParams_, self.pullCriticParams_])

    def pushToGlobalNet(self, state, action, valueTarget):
        self.session.run([self.updateActor_, self.updateCritic_], feed_dict= {self.states_: state, self.action_: action, self.valueTarget_: valueTarget})

    def getBootStrapValue(self, state):
        state = state.reshape(1, -1)
        value = self.session.run(self.value_, feed_dict={self.states_: state})
        return value


class SampleOneStep:
    def __init__(self, transit, getReward):
        self.transit = transit
        self.getReward = getReward

    def __call__(self, state, action):
        nextState = self.transit(state, action)
        reward = self.getReward(state, action, nextState)

        return reward, nextState


class GetValueTargetList:
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, valueFromBootStrap, rewardBuffer):
        valueTarget = valueFromBootStrap
        valueTargetList = []
        for reward in rewardBuffer[::-1]:  # reverse buffer r
            valueTarget = reward + self.gamma * valueTarget
            valueTargetList.append(valueTarget)
        valueTargetList.reverse()
        return valueTargetList


class Count(object):
    def __init__(self):
        self.count = 1

    def __call__(self):
        self.count += 1

class GlobalReward:
    def __init__(self):
        self.reward = []

    def addReward(self, rew):
        self.reward.append(rew)

class Experience:
    def __init__(self, isTerminal, sampleOneStep, observe = None):
        self.isTerminal = isTerminal
        self.sampleOneStep = sampleOneStep
        self.observe = observe
        self.resetBuffer()

    def resetBuffer(self):
        self.stateBuffer = []
        self.actionsBuffer = []
        self.rewardBuffer = []
        self.nextStateBuffer = []

    def getNewExperience(self, workerNet, state):
        observation = self.observe(state) if self.observe is not None else state
        action = workerNet.act(observation)
        reward, nextState = self.sampleOneStep(state, action)
        done = self.isTerminal(nextState)
        nextObservation = self.observe(nextState) if self.observe is not None else nextState
        self.stateBuffer.append(observation)
        self.actionsBuffer.append(action)
        self.nextStateBuffer.append(nextObservation)
        self.rewardBuffer.append((reward + 8) / 8)  # for pendulum v0 only


class A3CWorker:
    def __init__(self, globalMaxT, coord, reset, getValueTargetList, isTerminal,
                 maxTimeStepPerEps, sampleOneStep, globalCount, globalReward, updateInterval, workerNet, saveModel, pendulum = False, observe = None):
        self.globalMaxT = globalMaxT
        self.coord = coord
        self.globalCount = globalCount
        self.reset = reset
        self.maxTimeStepPerEps = maxTimeStepPerEps
        self.getValueTargetList = getValueTargetList
        self.isTerminal = isTerminal
        self.maxTimeStepPerEps = maxTimeStepPerEps
        self.sampleOneStep = sampleOneStep
        self.globalCount = globalCount
        self.observe = observe
        self.globalReward = globalReward
        self.updateInterval = updateInterval
        self.workerTotalSteps = 1
        self.workerNet = workerNet
        self.pendulum = pendulum
        self.saveModel = saveModel

    def work(self):
        stateBuffer = []
        actionsBuffer = []
        rewardBuffer = []
        nextStateBuffer = []

        while not self.coord.should_stop() and self.globalCount.count < self.globalMaxT:
            state = self.reset()
            epsReward = 0

            for timeStep in range(self.maxTimeStepPerEps):
                observation = self.observe(state) if self.observe is not None else state
                action = self.workerNet.act(observation)
                reward, nextState = self.sampleOneStep(state, action)
                done = self.isTerminal(nextState)
                nextObservation = self.observe(nextState) if self.observe is not None else nextState

                stateBuffer.append(observation)
                actionsBuffer.append(action)
                nextStateBuffer.append(nextObservation)
                if self.pendulum:
                    rewardBuffer.append((reward +8)/8) # for pendulum v0 only
                else:
                    rewardBuffer.append(reward)

                done = True if timeStep == (self.maxTimeStepPerEps - 1) or done else False
                epsReward += reward

                if self.workerTotalSteps % self.updateInterval == 0 or done:
                    lastState = nextStateBuffer[-1]
                    valueFromBootStrap = 0 if done else self.workerNet.getBootStrapValue(lastState)
                    valueTargetList = self.getValueTargetList(valueFromBootStrap, rewardBuffer)
                    stateBatch, actionBatch, valueTargetBatch = np.vstack(stateBuffer), np.vstack(actionsBuffer), np.vstack(valueTargetList)

                    self.workerNet.pushToGlobalNet(stateBatch, actionBatch, valueTargetBatch)
                    self.workerNet.pullFromGlobalNet()

                    stateBuffer = []
                    actionsBuffer = []
                    rewardBuffer = []
                    nextStateBuffer = []

                state = nextState
                self.workerTotalSteps += 1

                if done:
                    self.globalCount()
                    self.saveModel()
                    if len(self.globalReward.reward) == 0:
                        self.globalReward.addReward(epsReward)
                    else:
                        self.globalReward.addReward(0.9 * self.globalReward.reward[-1] + 0.1 * epsReward)

                    print(self.workerNet.scope, "Ep:", self.globalCount.count, "| Ep_r: %i" % self.globalReward.reward[-1])
                    # print(self.workerNet.scope, "Ep:", self.globalCount.count, "| Ep_r: %i" % epsReward)
                    break

class A3CWorkerUsingGym:
    def __init__(self, globalMaxT, coord, getValueTargetList, env, maxTimeStepPerEps, globalCount, globalReward, updateInterval, workerNet, saveModel, pendulum = False):
        self.globalMaxT = globalMaxT
        self.coord = coord
        self.globalCount = globalCount
        self.env = env
        self.maxTimeStepPerEps = maxTimeStepPerEps
        self.getValueTargetList = getValueTargetList
        self.maxTimeStepPerEps = maxTimeStepPerEps
        self.globalCount = globalCount
        self.globalReward = globalReward
        self.updateInterval = updateInterval
        self.workerTotalSteps = 1
        self.workerNet = workerNet
        self.saveModel = saveModel
        self.pendulum = pendulum

    def work(self):
        stateBuffer = []
        actionsBuffer = []
        rewardBuffer = []
        nextStateBuffer = []

        while not self.coord.should_stop() and self.globalCount.count < self.globalMaxT:
            state = self.env.reset()
            epsReward = 0

            for timeStep in range(self.maxTimeStepPerEps):
                action = self.workerNet.act(state)
                nextState, reward, done, info = self.env.step(action)
                if done:
                    print(self.workerNet.scope, 'done', ' rew', reward)

                stateBuffer.append(state)
                actionsBuffer.append(action)
                nextStateBuffer.append(nextState)
                if self.pendulum:
                    rewardBuffer.append((reward +8)/8) # for pendulum v0 only
                    done = True if timeStep == (self.maxTimeStepPerEps - 1) else False # for pendulum v0 only
                else:
                    rewardBuffer.append(reward)

                epsReward += reward

                if self.workerTotalSteps % self.updateInterval == 0 or done:
                    lastState = nextStateBuffer[-1]
                    valueFromBootStrap = 0 if done else self.workerNet.getBootStrapValue(lastState)
                    valueTargetList = self.getValueTargetList(valueFromBootStrap, rewardBuffer)
                    stateBatch, actionBatch, valueTargetBatch = np.vstack(stateBuffer), np.vstack(actionsBuffer), np.vstack(valueTargetList)

                    self.workerNet.pushToGlobalNet(stateBatch, actionBatch, valueTargetBatch)
                    self.workerNet.pullFromGlobalNet()

                    stateBuffer = []
                    actionsBuffer = []
                    rewardBuffer = []
                    nextStateBuffer = []

                state = nextState
                self.workerTotalSteps += 1

                if done:
                    self.globalCount()
                    self.saveModel()
                    # if len(self.globalReward.reward) == 0:
                    #     self.globalReward.addReward(epsReward)
                    # else:
                    #     self.globalReward.addReward(0.9 * self.globalReward.reward[-1] + 0.1 * epsReward)

                    # print(self.workerNet.scope, "Ep:", self.globalCount.count, "| Ep_r: %i" % self.globalReward.reward[-1])
                    print(self.workerNet.scope, "Ep:", self.globalCount.count, "| Ep_r: %i" % epsReward)
                    break


# class Update:
#     def __init__(self, buffer, updateInterval, getValueTargetList):
#         self.buffer = buffer
#         self.updateInterval = updateInterval
#         self.getValueTargetList = getValueTargetList
#
#     def __call__(self, workerTotalSteps, done, workerNet):
#         if workerTotalSteps % self.updateInterval == 0 or done:
#             lastState = self.buffer.nextStateBuffer[-1]
#             valueFromBootStrap = 0 if done else workerNet.getBootStrapValue(lastState)
#             valueTargetList = self.getValueTargetList(valueFromBootStrap, self.buffer.rewardBuffer)
#             stateBatch, actionBatch, valueTargetBatch = np.vstack(self.buffer.stateBuffer), np.vstack(self.buffer.actionsBuffer), np.vstack(
#                 valueTargetList)
#
#             workerNet.pushToGlobalNet(stateBatch, actionBatch, valueTargetBatch)
#             workerNet.pullFromGlobalNet()
#
#             self.buffer.reset()
#         else:
#             return
#
#
# class Buffer:
#     def __init__(self, isTerminal, sampleOneStep, observe=None):
#         self.isTerminal = isTerminal
#         self.sampleOneStep = sampleOneStep
#         self.observe = observe
#         self.resetBuffer()
#
#     def resetBuffer(self):
#         self.stateBuffer = []
#         self.actionsBuffer = []
#         self.rewardBuffer = []
#         self.nextStateBuffer = []
#
#     def getNewExperience(self, workerNet, state):
#         observation = self.observe(state) if self.observe is not None else state
#         action = workerNet.act(observation)
#         reward, nextState = self.sampleOneStep(state, action)
#         done = self.isTerminal(nextState)
#         nextObservation = self.observe(nextState) if self.observe is not None else nextState
#         self.stateBuffer.append(observation)
#         self.actionsBuffer.append(action)
#         self.nextStateBuffer.append(nextObservation)
#         self.rewardBuffer.append((reward + 8) / 8)  # for pendulum v0 only
#
#         return reward, nextState
#
#
# class A3CWorker:
#     def __init__(self, globalMaxT, coord, reset, update, buffer,
#                  maxTimeStepPerEps, globalCount, globalReward, workerNet, observe = None):
#         self.globalMaxT = globalMaxT
#         self.coord = coord
#         self.globalCount = globalCount
#         self.reset = reset
#         self.update = update
#         self.maxTimeStepPerEps = maxTimeStepPerEps
#         self.buffer = buffer
#         self.maxTimeStepPerEps = maxTimeStepPerEps
#         self.globalCount = globalCount
#         self.observe = observe
#         self.globalReward = globalReward
#         self.workerTotalSteps = 1
#         self.workerNet = workerNet
#
#     def work(self):
#         while not self.coord.should_stop() and self.globalCount.count < self.globalMaxT:
#             state = self.reset()
#             epsReward = 0
#
#             for timeStep in range(self.maxTimeStepPerEps):
#                 reward, nextState = self.buffer.getNewExperience(self.workerNet, state)
#                 epsReward += reward
#                 done = True if timeStep == (self.maxTimeStepPerEps - 1) else False # for pendulum v0 only
#                 self.update(self.workerTotalSteps, done, self.workerNet)
#                 state = nextState
#                 self.workerTotalSteps += 1
#
#                 if done:
#                     self.globalCount()
#                     if len(self.globalReward.reward) == 0:
#                         self.globalReward.addReward(epsReward)
#                     else:
#                         self.globalReward.addReward(0.9 * self.globalReward.reward[-1] + 0.1 * epsReward)
#
#                     print(self.workerNet.scope, "Ep:", self.globalCount.count, "| Ep_r: %i" % self.globalReward.reward[-1])
#                     break
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