import tensorflow as tf
import numpy as np


class GlobalNet(object):
    def __init__(self, stateDim, actionDim, actionRange, actorLayersWidths, criticLayersWidths):
        self.stateDim = stateDim
        self.actionDim = actionDim
        self.actorActivFunction = tf.nn.relu6
        self.actorMuOutputActiv = tf.nn.tanh
        self.actorSigmaOutputActiv = tf.nn.softplus
        self.w_init = tf.random_normal_initializer(0., .1)
        self.actionLow, self.actionHigh = actionRange
        self.entropyBeta = 0.01
        self.actorLayersWidths = actorLayersWidths
        self.criticLayersWidths = criticLayersWidths
        self.actorLR = 0.0001
        self.criticLR = 0.001
        self.scope = 'global'

        self.buildNet()

    def buildNet(self):
        with tf.variable_scope(self.scope):
            with tf.variable_scope("inputs"):
                self.states_ = tf.placeholder(tf.float32, [None, self.stateDim], name='states_')

            with tf.variable_scope('actorNet'):
                for numUnits in self.actorLayersWidths:
                    actorTrainActivation_ = tf.layers.dense(self.states_, numUnits, self.actorActivFunction, kernel_initializer=self.w_init)
                mu_ = tf.layers.dense(actorTrainActivation_, self.actionDim, self.actorMuOutputActiv, kernel_initializer=self.w_init, name='mu_')
                sigma_ = tf.layers.dense(actorTrainActivation_, self.actionDim, self.actorSigmaOutputActiv, kernel_initializer=self.w_init, name='sigma_')

            with tf.variable_scope('criticNet'):
                for numUnits in self.criticLayersWidths:
                    criticTrainActivation_ = tf.layers.dense(self.states_, numUnits, self.actorActivFunction, kernel_initializer=self.w_init)
                value_ = tf.layers.dense(criticTrainActivation_, 1, kernel_initializer=self.w_init)  # state value

            with tf.variable_scope('parameters'):
                self.actorParam_ = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/actorNet')
                self.criticParam_ = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/criticNet')

            with tf.variable_scope("trainOpt"):
                self.actorOpt_ = tf.train.RMSPropOptimizer(self.actorLR, name=self.scope + '/RMSPropOptActor')
                self.criticOpt_ = tf.train.RMSPropOptimizer(self.criticLR, name= self.scope + '/RMSPropOptCritic')



# class A3C(object):
#     def __init__(self, stateDim, actionDim, actionRange, scope, actorLayersWidths, criticLayersWidths, globalModel, session):
#         self.stateDim = stateDim
#         self.actionDim = actionDim
#         self.actorActivFunction = tf.nn.relu6
#         self.actorMuOutputActiv = tf.nn.tanh
#         self.actorSigmaOutputActiv = tf.nn.softplus
#         self.w_init = tf.random_normal_initializer(0., .1)
#         self.actionLow, self.actionHigh = actionRange
#         self.entropyBeta = 0.01
#         self.actorLayersWidths = actorLayersWidths
#         self.criticLayersWidths = criticLayersWidths
#
#         self.globalModel = globalModel
#
#         self.actorLR = 0.0001
#         self.criticLR = 0.001
#         self.globalScope = 'global'
#         self.scope = scope
#         self.buildNet()
#         self.session = session
#
#     def buildNet(self):
#         with tf.variable_scope(self.scope):
#             with tf.variable_scope("inputs"):
#                 self.states_ = tf.placeholder(tf.float32, [None, self.stateDim], name='states_')
#                 self.action_ = tf.placeholder(tf.float32, [None, self.actionDim], name='actions_')
#                 self.valueTarget_ = tf.placeholder(tf.float32, [None, 1], 'valueTarget_')
#
#             with tf.variable_scope('actorNet'):
#                 for numUnits in self.actorLayersWidths:
#                     actorTrainActivation_ = tf.layers.dense(self.states_, numUnits, self.actorActivFunction,
#                                                             kernel_initializer=self.w_init)
#
#                 mu_ = tf.layers.dense(actorTrainActivation_, self.actionDim, self.actorMuOutputActiv,
#                                       kernel_initializer=self.w_init, name='mu_')
#                 sigma_ = tf.layers.dense(actorTrainActivation_, self.actionDim, self.actorSigmaOutputActiv,
#                                          kernel_initializer=self.w_init, name='sigma_')
#
#             with tf.variable_scope('criticNet'):
#                 for numUnits in self.criticLayersWidths:
#                     criticTrainActivation_ = tf.layers.dense(self.states_, numUnits, self.actorActivFunction,
#                                                              kernel_initializer=self.w_init)
#
#                 self.value_ = tf.layers.dense(criticTrainActivation_, 1, kernel_initializer=self.w_init, name='value_')  # state value
#
#             with tf.variable_scope('parameters'):
#                 self.actorParam_ = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope= self.scope + '/actorNet')
#                 self.criticParam_ = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope= self.scope + '/criticNet')
#
#             with tf.variable_scope('adjustParams'):
#                 self.mu_ = mu_ * self.actionLow
#                 self.sigma_ = sigma_ + 1e-4
#
#             with tf.variable_scope('output'):
#                 normal_dist = tf.distributions.Normal(mu_, sigma_)
#                 self.actionOutput_ = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=[0, 1]), self.actionLow, self.actionHigh)
#
#             with tf.variable_scope('loss'):
#                 tdError_ = tf.subtract(self.valueTarget_, self.value_, name='tdError_')
#                 criticLoss_ = tf.reduce_mean(tf.square(tdError_))
#
#                 log_prob = normal_dist.log_prob(self.action_)
#                 actionVal_ = log_prob * tf.stop_gradient(tdError_)
#                 entropy = normal_dist.entropy()  # encourage exploration
#                 actionValWithEntropy_ = self.entropyBeta * entropy + actionVal_
#                 actorLoss_ = tf.reduce_mean(-actionValWithEntropy_)
#
#             with tf.variable_scope('gradients'):
#                 self.actorGradients_ = tf.gradients(actorLoss_, self.actorParam_)
#                 self.criticGradients_ = tf.gradients(criticLoss_, self.criticParam_)
#
#
#     def act(self, state):
#         state = state[np.newaxis, :]
#         states_ = self.states_
#         actionOutput_ = self.actionOutput_
#         action = self.session.run(actionOutput_, feed_dict={states_: state})
#         return action
#
#     def pullFromGlobalNet(self):
#         workerActorParams_ = self.actorParam_
#         workerCriticParams_ = self.criticParam_
#
#         globalActorParams_ = self.globalModel.actorParam_
#         globalCriticParams_ = self.globalModel.criticParam_
#
#         pullActorParams_ = [workerParam.assign(globalParam) for workerParam, globalParam in zip(workerActorParams_, globalActorParams_)]
#         pullCriticParams_ = [workerParam.assign(globalParam) for workerParam, globalParam in zip(workerCriticParams_, globalCriticParams_)]
#
#         self.session.run([pullActorParams_, pullCriticParams_])
#
#     def pushToGlobalNet(self, state, action, valueTarget):
#         actorOpt_ = self.globalModel.actorOpt_
#         criticOpt_ = self.globalModel.criticOpt_
#
#         actorGradients_ = self.actorGradients_
#         criticGradients_ = self.criticGradients_
#         globalActorParams_ = self.globalModel.actorParam_
#         globalCriticParams_ = self.globalModel.criticParam_
#
#         updateActor_ = actorOpt_.apply_gradients(zip(actorGradients_, globalActorParams_))
#         updateCritic_ = criticOpt_.apply_gradients(zip(criticGradients_, globalCriticParams_))
#
#         states_ = self.states_
#         action_ = self.action_
#         valueTarget_ = self.valueTarget_
#
#         self.session.run([updateActor_, updateCritic_], feed_dict= {states_: state, action_: action, valueTarget_: valueTarget})
#
#     def getBootStrapValue(self, state):
#         states_ = self.states_
#         value_ = self.value_
#         self.session.run(value_, feed_dict={states_: state})

class A3C(object):
    def __init__(self, stateDim, actionDim, actionRange, scope, actorLayersWidths, criticLayersWidths, globalModel, session):
        self.stateDim = stateDim
        self.actionDim = actionDim
        self.actorActivFunction = tf.nn.relu6
        self.actorMuOutputActiv = tf.nn.tanh
        self.actorSigmaOutputActiv = tf.nn.softplus
        self.w_init = tf.random_normal_initializer(0., .1)
        self.actionLow, self.actionHigh = actionRange
        self.entropyBeta = 0.01
        self.actorLayersWidths = actorLayersWidths
        self.criticLayersWidths = criticLayersWidths

        self.globalModel = globalModel

        self.actorLR = 0.0001
        self.criticLR = 0.001
        self.globalScope = 'global'
        self.scope = scope
        self.buildNet()
        self.session = session

    def buildNet(self):
        with tf.variable_scope(self.scope):
            with tf.variable_scope("inputs"):
                self.states_ = tf.placeholder(tf.float32, [None, self.stateDim], name='states_')
                self.action_ = tf.placeholder(tf.float32, [None, self.actionDim], name='actions_')
                self.valueTarget_ = tf.placeholder(tf.float32, [None, 1], 'valueTarget_')

            with tf.variable_scope('actorNet'):
                for numUnits in self.actorLayersWidths:
                    actorTrainActivation_ = tf.layers.dense(self.states_, numUnits, self.actorActivFunction,
                                                            kernel_initializer=self.w_init)

                mu_ = tf.layers.dense(actorTrainActivation_, self.actionDim, self.actorMuOutputActiv,
                                      kernel_initializer=self.w_init, name='mu_')
                sigma_ = tf.layers.dense(actorTrainActivation_, self.actionDim, self.actorSigmaOutputActiv,
                                         kernel_initializer=self.w_init, name='sigma_')

            with tf.variable_scope('criticNet'):
                for numUnits in self.criticLayersWidths:
                    criticTrainActivation_ = tf.layers.dense(self.states_, numUnits, self.actorActivFunction,
                                                             kernel_initializer=self.w_init)

                self.value_ = tf.layers.dense(criticTrainActivation_, 1, kernel_initializer=self.w_init, name='value_')  # state value

            with tf.variable_scope('parameters'):
                self.actorParam_ = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope= self.scope + '/actorNet')
                self.criticParam_ = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope= self.scope + '/criticNet')

            with tf.variable_scope('adjustParams'):
                self.mu_ = mu_ * self.actionLow
                self.sigma_ = sigma_ + 1e-4

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

            with tf.variable_scope('gradients'):
                self.actorGradients_ = tf.gradients(actorLoss_, self.actorParam_)
                self.criticGradients_ = tf.gradients(criticLoss_, self.criticParam_)


    def act(self, state):
        state = state[np.newaxis, :]
        states_ = self.states_
        actionOutput_ = self.actionOutput_
        action = self.session.run(actionOutput_, feed_dict={states_: state})
        return action

    def pullFromGlobalNet(self):
        workerActorParams_ = self.actorParam_
        workerCriticParams_ = self.criticParam_

        globalActorParams_ = self.globalModel.actorParam_
        globalCriticParams_ = self.globalModel.criticParam_

        pullActorParams_ = [workerParam.assign(globalParam) for workerParam, globalParam in zip(workerActorParams_, globalActorParams_)]
        pullCriticParams_ = [workerParam.assign(globalParam) for workerParam, globalParam in zip(workerCriticParams_, globalCriticParams_)]

        self.session.run([pullActorParams_, pullCriticParams_])

    def pushToGlobalNet(self, state, action, valueTarget):
        actorOpt_ = self.globalModel.actorOpt_
        criticOpt_ = self.globalModel.criticOpt_

        actorGradients_ = self.actorGradients_
        criticGradients_ = self.criticGradients_
        globalActorParams_ = self.globalModel.actorParam_
        globalCriticParams_ = self.globalModel.criticParam_

        updateActor_ = actorOpt_.apply_gradients(zip(actorGradients_, globalActorParams_))
        updateCritic_ = criticOpt_.apply_gradients(zip(criticGradients_, globalCriticParams_))

        states_ = self.states_
        action_ = self.action_
        valueTarget_ = self.valueTarget_

        self.session.run([updateActor_, updateCritic_], feed_dict= {states_: state, action_: action, valueTarget_: valueTarget})

    def getBootStrapValue(self, state):
        states_ = self.states_
        value_ = self.value_
        self.session.run(value_, feed_dict={states_: state})


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
        # rewardBuffer = self.getRewardBuffer(buffer)
        valueTargetList = []
        for reward in rewardBuffer[::-1]:  # reverse buffer r
            valueTarget = reward + self.gamma * valueFromBootStrap
            valueTargetList.append(valueTarget)
        valueTargetList.reverse()
        return valueTargetList


class Count:
    def __init__(self):
        self.count = 1

    def __call__(self):
        self.count += 1


class ActNStepsBuffer:
    def __init__(self, isTerminal, maxTimeStepPerEps, sampleOneStep, workerCount, globalCount, observe = None):
        self.isTerminal = isTerminal
        self.maxTimeStepPerEps = maxTimeStepPerEps
        self.sampleOneStep = sampleOneStep
        self.workerCount = workerCount
        self.globalCount = globalCount
        self.observe = observe

    def __call__(self, state, workerModel):
        stateBuffer = []
        actionsBuffer = []
        rewardBuffer = []

        for timeStep in range(self.maxTimeStepPerEps):
            observation = self.observe(state) if self.observe is not None else state
            action = workerModel.act(observation)
            reward, nextState = self.sampleOneStep(state, action)
            done = self.isTerminal(nextState)
            # buffer.append((state, action, reward, nextState, done))
            stateBuffer.append(observation)
            actionsBuffer.append(action)
            rewardBuffer.append(reward)
            self.workerCount()
            self.globalCount()

            if done:
                break
            else:
                state = nextState

        observation = self.observe(state) if self.observe is not None else state
        stateBuffer.append(observation)

        # for pendulum v0 only
        done = True

        return stateBuffer, actionsBuffer, rewardBuffer, done


class RunOneThreadEpisode:
    def __init__(self, reset, maxTimeStepPerEps, getValueTargetList, actNStepsBuffer):
        self.reset = reset
        self.maxTimeStepPerEps = maxTimeStepPerEps
        self.getValueTargetList = getValueTargetList
        self.actNStepsBuffer = actNStepsBuffer

    def __call__(self, workerModel):
        state = self.reset()
        stateBuffer, actionsBuffer, rewardBuffer, done = self.actNStepsBuffer(state, workerModel)
        lastState = stateBuffer[-1]
        valueFromBootStrap = 0 if done else workerModel.getBootStrapValue(lastState)

        valueTargetList = self.getValueTargetList(valueFromBootStrap, rewardBuffer)
        stateBatch, actionBatch, valueTargetBatch = np.vstack(stateBuffer), np.vstack(actionsBuffer), np.vstack(valueTargetList)

        workerModel.pushToGlobalNet(stateBatch, actionBatch, valueTargetBatch)
        workerModel.pullFromGlobalNet()

        return workerModel


class RunA3C:
    def __init__(self, runEpisode, globalMaxT, coord, globalCount):
        self.runEpisode = runEpisode
        self.globalMaxT = globalMaxT
        self.coord = coord
        self.globalCount = globalCount

    def __call__(self, workerMoel):
        globalSharedT = self.globalCount.count
        while not self.coord.should_stop() and globalSharedT < self.globalMaxT:
            workerMoel = self.runEpisode(workerMoel)

