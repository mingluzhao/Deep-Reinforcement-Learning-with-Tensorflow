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

        self.globalNetModel = self.buildNet()
        self.graph = self.globalNetModel.graph

    def buildNet(self):
        graph = tf.Graph()
        with tf.variable_scope(self.scope):
            with tf.variable_scope("inputs"):
                states_ = tf.placeholder(tf.float32, [None, self.stateDim], name='states_')
                tf.add_to_collection(self.scope + "/states_", states_)

            with tf.variable_scope('actorNet'):
                for numUnits in self.actorLayersWidths:
                    actorTrainActivation_ = tf.layers.dense(states_, numUnits, self.actorActivFunction, kernel_initializer=self.w_init)
                mu_ = tf.layers.dense(actorTrainActivation_, self.actionDim, self.actorMuOutputActiv, kernel_initializer=self.w_init, name='mu_')
                sigma_ = tf.layers.dense(actorTrainActivation_, self.actionDim, self.actorSigmaOutputActiv, kernel_initializer=self.w_init, name='sigma_')

                tf.add_to_collection(self.scope + "/mu_", mu_)
                tf.add_to_collection(self.scope + "/sigma_", sigma_)

            with tf.variable_scope('criticNet'):
                for numUnits in self.criticLayersWidths:
                    criticTrainActivation_ = tf.layers.dense(states_, numUnits, self.actorActivFunction, kernel_initializer=self.w_init)

                value_ = tf.layers.dense(criticTrainActivation_, 1, kernel_initializer=self.w_init)  # state value
                tf.add_to_collection(self.scope + "/value_", value_)

            with tf.variable_scope('parameters'):
                actorParam_ = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/actorNet')
                criticParam_ = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/criticNet')

                tf.add_to_collection(self.scope + "/actorParam_", actorParam_)
                tf.add_to_collection(self.scope + "/criticParam_", criticParam_)

            with tf.variable_scope("trainOpt"):
                actorOpt_ = tf.train.RMSPropOptimizer(self.actorLR, name='RMSPropOptActor')
                criticOpt_ = tf.train.RMSPropOptimizer(self.criticLR, name='RMSPropOptCritic')
                tf.add_to_collection(self.scope + "/actorOpt_", actorOpt_)
                tf.add_to_collection(self.scope + "/criticOpt_", criticOpt_)

            model = tf.Session()
            model.run(tf.global_variables_initializer())

        return model


class A3C(object):
    def __init__(self, stateDim, actionDim, actionRange, scope, actorLayersWidths, criticLayersWidths, globalModel):
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
        self.model = self.buildNet()
        self.graph = self.model.graph


    def act(self, state):
        state = state[np.newaxis, :]
        states_ = self.graph.get_collection_ref(self.scope + "/states_")[0]
        actionOutput_ = self.graph.get_collection_ref(self.scope + "/actionOutput_")[0]
        action = self.model.run(actionOutput_, feed_dict={states_: state})
        return action

    def buildNet(self):
        graph = tf.Graph()
    # with graph.as_default():
        with tf.variable_scope(self.scope):
            with tf.variable_scope("inputs"):
                states_ = tf.placeholder(tf.float32, [None, self.stateDim], name='states_')
                action_ = tf.placeholder(tf.float32, [None, self.actionDim], name='actions_')
                valueTarget_ = tf.placeholder(tf.float32, [None, 1], 'valueTarget_')

                tf.add_to_collection(self.scope + "/states_", states_)
                tf.add_to_collection(self.scope + "/action_", action_)
                tf.add_to_collection(self.scope + "/valueTarget_", valueTarget_)

            with tf.variable_scope('actorNet'):
                for numUnits in self.actorLayersWidths:
                    actorTrainActivation_ = tf.layers.dense(states_, numUnits, self.actorActivFunction,
                                                            kernel_initializer=self.w_init)

                mu_ = tf.layers.dense(actorTrainActivation_, self.actionDim, self.actorMuOutputActiv,
                                      kernel_initializer=self.w_init, name='mu_')
                sigma_ = tf.layers.dense(actorTrainActivation_, self.actionDim, self.actorSigmaOutputActiv,
                                         kernel_initializer=self.w_init, name='sigma_')

                tf.add_to_collection(self.scope + "/mu_", mu_)
                tf.add_to_collection(self.scope + "/sigma_", sigma_)

            with tf.variable_scope('criticNet'):
                for numUnits in self.criticLayersWidths:
                    criticTrainActivation_ = tf.layers.dense(states_, numUnits, self.actorActivFunction,
                                                             kernel_initializer=self.w_init)

                value_ = tf.layers.dense(criticTrainActivation_, 1, kernel_initializer=self.w_init, name='value_')  # state value
                tf.add_to_collection(self.scope + "/value_", value_)

            with tf.variable_scope('parameters'):
                actorParam_ = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope= self.scope + '/actorNet')
                criticParam_ = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope= self.scope + '/criticNet')

                tf.add_to_collection(self.scope + "/actorParam_", actorParam_)
                tf.add_to_collection(self.scope + "/criticParam_", criticParam_)

            with tf.variable_scope('adjustParams'):
                mu_ = mu_ * self.actionLow
                sigma_ = sigma_ + 1e-4

            with tf.variable_scope('output'):
                normal_dist = tf.distributions.Normal(mu_, sigma_)
                actionOutput_ = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=[0, 1]), self.actionLow, self.actionHigh)
                tf.add_to_collection(self.scope + "/actionOutput_", actionOutput_)

            with tf.variable_scope('loss'):
                tdError_ = tf.subtract(valueTarget_, value_, name='tdError_')
                criticLoss_ = tf.reduce_mean(tf.square(tdError_))

                log_prob = normal_dist.log_prob(action_)
                actionVal_ = log_prob * tf.stop_gradient(tdError_)
                entropy = normal_dist.entropy()  # encourage exploration
                actionValWithEntropy_ = self.entropyBeta * entropy + actionVal_
                actorLoss_ = tf.reduce_mean(-actionValWithEntropy_)

                tf.add_to_collection(self.scope + "/criticLoss_", criticLoss_)
                tf.add_to_collection(self.scope + "/actorLoss_", actorLoss_)

            with tf.variable_scope('gradients'):
                actorGradients_ = tf.gradients(actorLoss_, actorParam_)
                criticGradients_ = tf.gradients(criticLoss_, criticParam_)
                tf.add_to_collection(self.scope + "/actorGradients_", actorGradients_)
                tf.add_to_collection(self.scope + "/criticGradients_", criticGradients_)

        model = tf.Session()
        model.run(tf.global_variables_initializer())

        return model

    def pullFromGlobalNet(self):
        workerGraph = self.model.graph
        globalGraph = self.globalModel.graph

        workerActorParams_ = workerGraph.get_collection_ref(self.scope + "/actorParam_")[0]
        workerCriticParams_ = workerGraph.get_collection_ref(self.scope + "/criticParam_")[0]

        globalActorParams_ = globalGraph.get_collection_ref(self.globalScope + "/actorParam_")[0]
        globalCriticParams_ = globalGraph.get_collection_ref(self.globalScope + "/criticParam_")[0]

        pullActorParams_ = [workerParam.assign(globalParam) for workerParam, globalParam in zip(workerActorParams_, globalActorParams_)]
        pullCriticParams_ = [workerParam.assign(globalParam) for workerParam, globalParam in zip(workerCriticParams_, globalCriticParams_)]

        self.model.run([pullActorParams_, pullCriticParams_])

    def pushToGlobalNet(self, state, action, valueTarget):
        workerGraph = self.model.graph
        globalGraph = self.globalModel.graph

        actorOpt_ = globalGraph.get_collection_ref(self.globalScope + "/actorOpt_")[0]
        criticOpt_ = globalGraph.get_collection_ref(self.globalScope + "/criticOpt_")[0]

        actorGradients_ = workerGraph.get_collection_ref(self.scope + "/actorGradients_")[0]
        criticGradients_ = workerGraph.get_collection_ref(self.scope + "/criticGradients_")[0]
        globalActorParams_ = globalGraph.get_collection_ref(self.globalScope + "/actorParam_")[0]
        globalCriticParams_ = globalGraph.get_collection_ref(self.globalScope + "/criticParam_")[0]

        updateActor_ = actorOpt_.apply_gradients(zip(actorGradients_, globalActorParams_))
        updateCritic_ = criticOpt_.apply_gradients(zip(criticGradients_, globalCriticParams_))

        states_ = workerGraph.get_collection_ref(self.scope + "/states_")[0]
        action_ = workerGraph.get_collection_ref(self.scope + "/action_")[0]
        valueTarget_ = workerGraph.get_collection_ref(self.scope + "/valueTarget_")[0]

        self.model.run([updateActor_, updateCritic_], feed_dict= {states_: state, action_: action, valueTarget_: valueTarget})

    def getBootStrapValue(self, state):
        workerGraph = self.model.graph
        states_ = workerGraph.get_collection_ref(self.scope + "/states_")[0]
        value_ = workerGraph.get_collection_ref(self.scope + "/value_")[0]
        self.model.run(value_, feed_dict={states_: state})


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










