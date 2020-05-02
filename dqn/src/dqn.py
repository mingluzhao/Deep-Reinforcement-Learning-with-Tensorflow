import tensorflow as tf
import numpy as np
import random
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class ActByTrainNetEpsilonGreedy:
    def __init__(self, epsilonMax, epsilonMin, epsilonDecay, getTrainQValue):
        self.epsilonMax = epsilonMax
        self.epsilonMin = epsilonMin
        self.epsilonDecay = epsilonDecay
        self.getTrainQValue = getTrainQValue

    def __call__(self, modelList, stateBatch, runTime):
        model = modelList[0]
        epsilonResult = self.epsilonMax * self.epsilonDecay ** runTime
        epsilon = epsilonResult if epsilonResult > self.epsilonMin else self.epsilonMin

        trainQVal = self.getTrainQValue(model, stateBatch)

        if np.random.uniform() > epsilon:
            action = np.argmax(trainQVal)
        else:
            action = random.randrange(len(trainQVal[0]))
        action = action
        return [action]


def getTrainQValue(model, stateBatch):
    graph = model.graph
    states_ = graph.get_collection_ref("states_")[0]
    trainQ_ = graph.get_collection_ref("trainQ_")[0]
    trainQ = model.run(trainQ_, feed_dict={states_: stateBatch})
    return trainQ


def getTargetQValue(model, stateBatch):
    graph = model.graph
    states_ = graph.get_collection_ref("states_")[0]
    targetQ_ = graph.get_collection_ref("targetQ_")[0]
    targetQ = model.run(targetQ_, feed_dict={states_: stateBatch})
    return targetQ


def getTrainQValueForAction(model, stateBatch, actionBatch):
    graph = model.graph
    states_ = graph.get_collection_ref("states_")[0]
    actions_ = graph.get_collection_ref("actions_")[0]
    trainQForAction_ = graph.get_collection_ref("trainQForAction_")[0]
    trainQForAction = model.run(trainQForAction_, feed_dict={states_: stateBatch, actions_: actionBatch})
    return trainQForAction


class BuildModel:
    def __init__(self, numStateSpace, actionDim):
        self.numStateSpace = numStateSpace
        self.actionDim = actionDim

    def __call__(self, layersWidths, summaryPath="./tbdata"):
        print("Generating NN Model with layers: {}".format(layersWidths))
        graph = tf.Graph()
        with graph.as_default():
            with tf.name_scope("inputs"):
                states_ = tf.placeholder(tf.float32, [None, self.numStateSpace])
                rewards_ = tf.placeholder(tf.float32, [None, ])
                actions_ = tf.placeholder(tf.int32, [None, ])
                targetNextStateQ_ = tf.placeholder(tf.float32, [None, self.actionDim])

                tf.add_to_collection("states_", states_)
                tf.add_to_collection("rewards_", rewards_)
                tf.add_to_collection("actions_", actions_)
                tf.add_to_collection("targetNextStateQ_", targetNextStateQ_)

            with tf.name_scope("trainingParams"):
                learningRate_ = tf.constant(0, dtype=tf.float32)
                gamma_ = tf.constant(0, dtype=tf.float32)
                tf.add_to_collection("learningRate_", learningRate_)
                tf.add_to_collection("gamma_", gamma_)

            initWeight = tf.random_uniform_initializer(0, 0.3)
            initBias = tf.constant_initializer(0.1)
            with tf.variable_scope("trainHidden"):
                activation_ = states_
                for i in range(len(layersWidths)):
                    fcLayer = tf.layers.Dense(units=layersWidths[i], activation=tf.nn.relu, kernel_initializer=initWeight,
                                              bias_initializer=initBias, name="fc{}".format(i+1), trainable = True)
                    activation_ = fcLayer(activation_)

                    tf.add_to_collections(["weights", f"weight/{fcLayer.kernel.name}"], fcLayer.kernel)
                    tf.add_to_collections(["biases", f"bias/{fcLayer.bias.name}"], fcLayer.bias)
                    tf.add_to_collections(["activations", f"activation/{activation_.name}"], activation_)
                trainActivation_ = tf.identity(activation_)
                outputFCLayer = tf.layers.Dense(units=self.actionDim, activation= None, kernel_initializer=initWeight,
                                                bias_initializer=initBias,name="fc{}".format(len(layersWidths) + 1), trainable = True)
                trainActivationOutput_ = outputFCLayer(trainActivation_)

                tf.add_to_collections(["weights", f"weight/{outputFCLayer.kernel.name}"], outputFCLayer.kernel)
                tf.add_to_collections(["biases", f"bias/{outputFCLayer.bias.name}"], outputFCLayer.bias)
                tf.add_to_collections(["activations", f"activation/{trainActivationOutput_.name}"], trainActivationOutput_)

            with tf.variable_scope("targetHidden"):
                activation_ = states_
                for i in range(len(layersWidths)):
                    fcLayer = tf.layers.Dense(units=layersWidths[i], activation=tf.nn.relu, kernel_initializer=initWeight,
                                              bias_initializer=initBias, name="fc{}".format(i+1), trainable = True)
                    activation_ = fcLayer(activation_)

                    tf.add_to_collections(["weights", f"weight/{fcLayer.kernel.name}"], fcLayer.kernel)
                    tf.add_to_collections(["biases", f"bias/{fcLayer.bias.name}"], fcLayer.bias)
                    tf.add_to_collections(["activations", f"activation/{activation_.name}"], activation_)
                targetActivation_ = tf.identity(activation_)
                outputFCLayer = tf.layers.Dense(units=self.actionDim, activation= None, kernel_initializer=initWeight,
                                                bias_initializer=initBias,name="fc{}".format(len(layersWidths) + 1), trainable = True)
                targetActivationOutput_ = outputFCLayer(targetActivation_)

                tf.add_to_collections(["weights", f"weight/{outputFCLayer.kernel.name}"], outputFCLayer.kernel)
                tf.add_to_collections(["biases", f"bias/{outputFCLayer.bias.name}"], outputFCLayer.bias)
                tf.add_to_collections(["activations", f"activation/{targetActivationOutput_.name}"], targetActivationOutput_)

            with tf.name_scope("updateParameters"):
                trainParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='trainHidden')
                targetParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='targetHidden')
                updateParam_ = [tf.assign(trainParam, targetParam) for trainParam, targetParam in zip(trainParams_, targetParams_)]

                tf.add_to_collection("trainParams_", trainParams_)
                tf.add_to_collection("targetParams_", targetParams_)
                tf.add_to_collection("updateParam_", updateParam_)

                hardReplaceTargetParam_ = [tf.assign(trainParam, targetParam) for trainParam, targetParam in zip(trainParams_, targetParams_)]
                tf.add_to_collection("hardReplaceTargetParam_", hardReplaceTargetParam_)

            with tf.name_scope("output"):
                trainQ_ = tf.multiply(trainActivationOutput_, 1, name='trainQ_')
                targetQ_ = tf.multiply(targetActivationOutput_, 1, name='targetQ_')
                #
                a_indices = tf.stack([tf.range(tf.shape(actions_)[0], dtype=tf.int32), actions_], axis=1)
                trainQForAction_ = tf.gather_nd(params= trainQ_, indices=a_indices)

                # action_one_hot = tf.one_hot(actions_, self.actionDim, 1.0, 0.0, name='action_one_hot')
                # trainQForAction_ = tf.reduce_sum(trainQ_ * action_one_hot, reduction_indices=-1, name='trainQForAction_')

                tf.add_to_collection("trainQ_", trainQ_)
                tf.add_to_collection("targetQ_", targetQ_)
                tf.add_to_collection("trainQForAction_", trainQForAction_)

            with tf.variable_scope('q_target'):
                q_target = rewards_ + gamma_ * tf.reduce_max(targetNextStateQ_, axis=1)
                q_target = tf.stop_gradient(q_target)

            with tf.variable_scope('loss'):
                loss_ = tf.reduce_mean(tf.squared_difference(q_target,trainQForAction_,name='loss'))
                tf.add_to_collection("loss_", loss_)

            with tf.variable_scope('train'):
                trainOpt_ = tf.train.RMSPropOptimizer(learningRate_).minimize(loss_, name="trainOpt_")
                tf.add_to_collection("trainOpt_", trainOpt_)

            saver = tf.train.Saver(max_to_keep=None)
            tf.add_to_collection("saver", saver)

            model = tf.Session(graph=graph)
            model.run(tf.global_variables_initializer())

            writer = tf.summary.FileWriter('tensorBoard/dqn', graph= graph)
            tf.add_to_collection("writer", writer)




        return writer, model





class TrainModelBySASRQ:
    def __init__(self, learningRate, gamma, writer):
        self.learningRate = learningRate
        self.gamma = gamma
        self.writer = writer

    def __call__(self, model, stateBatch, actionBatch, rewardBatch, targetNextStateQ):
        graph = model.graph
        states_ = graph.get_collection_ref("states_")[0]
        targetNextStateQ_ = graph.get_collection_ref("targetNextStateQ_")[0]
        rewards_ = graph.get_collection_ref("rewards_")[0]
        actions_ = graph.get_collection_ref("actions_")[0]
        learningRate_ = graph.get_collection_ref("learningRate_")[0]
        gamma_ = graph.get_collection_ref("gamma_")[0]

        loss_ = graph.get_collection_ref("loss_")[0]
        trainOpt_ = graph.get_collection_ref("trainOpt_")[0]

        loss, trainOpt = model.run([loss_, trainOpt_],
                                   feed_dict={states_: stateBatch, targetNextStateQ_: targetNextStateQ, rewards_: rewardBatch,
                                              actions_: actionBatch, learningRate_: self.learningRate, gamma_: self.gamma})
        self.writer.flush()

        return loss, model


class TrainDQNModel:
    def __init__(self, getTargetQValue, trainModelBySASRQ, updateParameters):
        self.getTargetQValue = getTargetQValue
        self.trainModelBySASRQ = trainModelBySASRQ
        self.updateParameters = updateParameters

    def __call__(self, modelList, miniBatch, runTime):
        model = modelList[0] # single model to train
        states, actions, rewards, nextStates = list(zip(*miniBatch))
        stateBatch = np.asarray(states).reshape(len(miniBatch), -1)
        actionBatch = np.asarray(actions).reshape(len(miniBatch))
        nextStateBatch = np.asarray(nextStates).reshape(len(miniBatch), -1)
        rewardBatch = np.asarray(rewards).reshape(len(miniBatch))

        targetNextStateQ = self.getTargetQValue(model, nextStateBatch)
        loss, model = self.trainModelBySASRQ(model, stateBatch, actionBatch, rewardBatch, targetNextStateQ)
        model = self.updateParameters(model, runTime)

        return [model]

#
# def addToMemory(buffer, state, action, reward, nextState):
#     experience = (state, action, reward, nextState)
#     buffer.append(experience)
#     return buffer
#
#
# class RunDQNTimeStep:
#     def __init__(self, actByTrainNetEpsilonGreedy,
#                  transit, getReward, isTerminal,
#                  addToMemory, observe, trainModel, minibatchSize,
#                  learningStartBufferSize):
#         self.actByTrainNetEpsilonGreedy = actByTrainNetEpsilonGreedy
#
#         self.transit = transit
#         self.getReward = getReward
#         self.isTerminal = isTerminal
#         self.observe = observe
#         self.addToMemory = addToMemory
#         self.trainModel = trainModel
#         self.minibatchSize = minibatchSize
#         self.learningStartBufferSize = learningStartBufferSize
#
#     def __call__(self, state, model, replayBuffer, trajectory):
#         runTime = len(trajectory)
#
#         observation = self.observe(state)
#         observation = np.asarray(observation).reshape(1, -1)
#         action = self.actByTrainNetEpsilonGreedy(model, observation, runTime)
#
#         nextState = self.transit(state, action)
#         reward = self.getReward(state, action)
#         terminal = self.isTerminal(nextState)
#         trajectory.append((state, action))
#         nextObservation = self.observe(nextState)
#         replayBuffer = self.addToMemory(replayBuffer, observation, action, reward, nextObservation)
#
#         if runTime >= self.learningStartBufferSize:
#             miniBatch = random.sample(replayBuffer, self.minibatchSize)
#             model = self.trainModel(model, miniBatch, runTime)
#
#         return reward, nextState, model, replayBuffer, terminal, trajectory
#
#
# class RunEpisode:
#     def __init__(self, reset, runTimeStep, maxTimeStep):
#         self.reset = reset
#         self.runTimeStep = runTimeStep
#         self.maxTimeStep = maxTimeStep
#
#     def __call__(self, model, replayBuffer, trajectory):
#         state = self.reset()
#         episodeReward = 0
#         for timeStep in range(self.maxTimeStep):
#             reward, state, model, replayBuffer, terminal, trajectory = \
#                 self.runTimeStep(state, model, replayBuffer, trajectory)
#             if terminal:
#                 break
#             episodeReward += reward
#         print('episodeReward: ', episodeReward)
#         return model, replayBuffer, episodeReward, trajectory
#
#
# class DQN:
#     def __init__(self, runEpisode, bufferSize, maxEpisode, print = True):
#         self.bufferSize = bufferSize
#         self.runEpisode = runEpisode
#         self.maxEpisode = maxEpisode
#         self.print = print
#
#     def __call__(self, model):
#         replayBuffer = deque(maxlen=int(self.bufferSize))
#         episodeRewardList = []
#         meanRewardList = []
#         trajectory = []
#         for episode in range(self.maxEpisode):
#             model, replayBuffer, episodeReward, trajectory = self.runEpisode(model, replayBuffer, trajectory)
#             episodeRewardList.append(episodeReward)
#             meanRewardList.append(np.mean(episodeRewardList))
#
#             if print:
#                 print('episode', episode)
#                 if episode == self.maxEpisode - 1:
#                     print('mean episode reward: ', int(np.mean(episodeRewardList)))
#
#         return meanRewardList, trajectory, model
#














