import tensorflow as tf
import numpy as np
from collections import deque
import random


# def trainCritic(actorTargetModel, criticTargetModel, criticModel,
#                  nextStateBatch, rewardBatch, stateBatch,
#                  actionBatch):
#     actorTargetGraph = actorTargetModel.graph
#     state_ = actorTargetGraph.get_collection_ref("state_")
#     actionsOutput_ = actorTargetGraph.get_collection_ref("actionsOutput_")
#     actionsOutput = actorTargetModel.run(actionsOutput_, feed_dict = {state_: nextStateBatch})
#
#     criticTargetGraph = criticTargetModel.graph
#     state_ = criticTargetGraph.get_collection_ref("state_")
#     action_ = criticTargetGraph.get_collection_ref("action_")
#     qValue_ = criticTargetGraph.get_collection_ref("qValue_")
#     qValue = criticTargetGraph.run(qValue_, feed_dict={state_: nextStateBatch,
#                                                       action_: actionsOutput})
#
#     criticGraph = criticModel.graph
#     state_ = criticGraph.get_collection_ref("state_")
#     action_ = criticGraph.get_collection_ref("action_")
#     reward_ = criticGraph.get_collection_ref("reward_")
#     valueTarget_ = criticGraph.get_collection_ref("valueTarget_")
#
#     loss_ = criticGraph.get_collection_ref("loss_")
#     criticParams_ = criticGraph.get_collection_ref("criticCurrentModelParams_")
#     trainOpt_ = criticGraph.get_collection_ref("trainOpt_")
#
#     loss, criticParams, trainOpt = criticModel.run([loss_, criticParams_, trainOpt_],
#                                      feed_dict={state_: stateBatch,
#                                                 action_: actionBatch,
#                                                 reward_: rewardBatch,
#                                                 valueTarget_: qValue})
#
#     criticTrainParams_ = criticTargetGraph.get_collection_ref("criticTrainParams_")
#     replaceParam_ = criticTargetGraph.get_collection_ref("replaceParam_")
#     replaceParam = criticTargetModel.run(replaceParam_,
#                                          feed_dict = {criticTrainParams_: criticParams})
#
#     return loss, trainOpt, criticTargetModel
#
#
# def trainActor(actorModel, criticModel, stateBatch, actorTargetModel):
#     actorGraph = actorModel.graph
#     state_ = actorGraph.get_collection_ref("state_")
#     actionsOutput_ = actorGraph.get_collection_ref("actionsOutput_")
#     actionsOutput = actorModel.run(actionsOutput_, feed_dict = {state_: stateBatch})
#
#     criticGraph = criticModel.graph
#     state_ = criticGraph.get_collection_ref("state_")
#     action_ = criticGraph.get_collection_ref("action_")
#     actionGradients_ = criticGraph.get_collection_ref("actionGradients_")
#     actionGradients = criticModel.run(actionGradients_, feed_dict={state_: stateBatch,
#                                                                    action_: actionsOutput})
#
#     actionGradients_ = actorGraph.get_collection_ref("actionGradients_")
#     actorParams_ = actorGraph.get_collection_ref("actorCurrentModelParams_")
#     trainOpt_ = actorGraph.get_collection_ref("trainOpt_")
#     actorParams, trainOpt = actorModel.run([actorParams_, trainOpt_],
#                                            feed_dict={state_: stateBatch,
#                                                       actionGradients_: actionGradients})
#
#     actorTargetGraph = actorTargetModel.graph
#     actorTrainParams_ = actorTargetGraph.get_collection_ref("actorTrainParams_")
#     replaceParam_ = actorTargetGraph.get_collection_ref("replaceParam_")
#
#     replaceParam = actorTargetModel.run(replaceParam_, feed_dict = {actorTrainParams_: actorParams})
#
#     return trainOpt, actorTargetModel


def actByDeterministicPolicy(actorModel, stateBatch):
    actorGraph = actorModel.graph
    state_ = actorGraph.get_collection_ref("state_")
    actionsOutput_ = actorGraph.get_collection_ref("actionsOutput_")
    actionsOutput = actorModel.run(actionsOutput_, feed_dict={state_: stateBatch})
    return actionsOutput


def getCriticQValue(criticModel, stateBatch, actionsBatch):
    criticGraph = criticModel.graph
    state_ = criticGraph.get_collection_ref("state_")
    action_ = criticGraph.get_collection_ref("action_")
    qValue_ = criticGraph.get_collection_ref("qValue_")
    qValue = criticGraph.run(qValue_, feed_dict={state_: stateBatch, action_: actionsBatch})
    return qValue


def updateCriticParameter(criticTargetModel, criticParams):
    criticTargetGraph = criticTargetModel.graph
    criticTrainParams_ = criticTargetGraph.get_collection_ref("criticTrainParams_")
    replaceParam_ = criticTargetGraph.get_collection_ref("replaceParam_")
    criticTargetModel.run(replaceParam_, feed_dict={criticTrainParams_: criticParams})
    return criticTargetModel


class TrainCritic:
    def __init__(self, criticWriter, actByDeterministicPolicy, getCriticQValue, updateCriticParameter):
        self.criticWriter = criticWriter
        self.actByDeterministicPolicy = actByDeterministicPolicy
        self.getCriticQValue = getCriticQValue
        self.updateCriticParameter = updateCriticParameter

    def __call__(self, actorTargetModel, criticModel, criticTargetModel, nextStateBatch, stateBatch, actionBatch,
                 rewardBatch):
        targetNextActionBatch = self.actByDeterministicPolicy(actorTargetModel, nextStateBatch)
        targetQValue = self.getCriticQValue(criticTargetModel, nextStateBatch, targetNextActionBatch)

        criticGraph = criticModel.graph
        state_ = criticGraph.get_collection_ref("state_")
        action_ = criticGraph.get_collection_ref("action_")
        reward_ = criticGraph.get_collection_ref("reward_")
        valueTarget_ = criticGraph.get_collection_ref("valueTarget_")
        loss_ = criticGraph.get_collection_ref("loss_")
        criticParams_ = criticGraph.get_collection_ref("criticCurrentModelParams_")
        trainOpt_ = criticGraph.get_collection_ref("trainOpt_")

        criticLoss, criticParams, trainOpt = criticModel.run([loss_, criticParams_, trainOpt_],feed_dict={state_: stateBatch, action_: actionBatch, reward_: rewardBatch, valueTarget_: targetQValue})
        criticTargetModel = self.updateCriticParameter(criticTargetModel, criticParams)
        self.criticWriter.flush()

        return criticLoss, trainOpt, criticTargetModel


def getActionGradients(criticModel, stateBatch, actionsBatch):
    criticGraph = criticModel.graph
    state_ = criticGraph.get_collection_ref("state_")
    action_ = criticGraph.get_collection_ref("action_")
    actionGradients_ = criticGraph.get_collection_ref("actionGradients_")
    actionGradients = criticModel.run(actionGradients_, feed_dict={state_: stateBatch,
                                                                   action_: actionsBatch})
    return actionGradients


def updateActorParameter(actorTargetModel, actorParams):
    actorTargetGraph = actorTargetModel.graph
    actorTrainParams_ = actorTargetGraph.get_collection_ref("actorTrainParams_")
    replaceParam_ = actorTargetGraph.get_collection_ref("replaceParam_")
    actorTargetModel.run(replaceParam_, feed_dict={actorTrainParams_: actorParams})
    return actorTargetModel


class TrainActor:
    def __init__(self, actorWriter, actByDeterministicPolicy, getActionGradients, updateActorParameter):
        self.actorWriter = actorWriter
        self.actByDeterministicPolicy = actByDeterministicPolicy
        self.getActionGradients = getActionGradients
        self.updateActorParameter = updateActorParameter

    def __call__(self, actorModel, actorTargetModel, criticModel, stateBatch):
        actionsBatch = self.actByDeterministicPolicy(actorModel, stateBatch)
        actionGradients = self.getActionGradients(criticModel, stateBatch, actionsBatch)

        actorGraph = actorModel.graph
        state_ = actorGraph.get_collection_ref("state_")
        actionGradients_ = actorGraph.get_collection_ref("actionGradients_")
        actorParams_ = actorGraph.get_collection_ref("actorCurrentModelParams_")
        trainOpt_ = actorGraph.get_collection_ref("trainOpt_")
        actorParams, trainOpt = actorModel.run([actorParams_, trainOpt_], feed_dict={state_: stateBatch, actionGradients_: actionGradients})
        actorTargetModel = self.updateActorParameter(actorTargetModel, actorParams)
        self.actorWriter.flush()

        return trainOpt, actorTargetModel


class BuildActorModel:
    def __init__(self, numStateSpace, numActionSpace, numActorFC1Unit, learningRateActor, tau):
        self.numStateSpace = numStateSpace
        self.numActionSpace = numActionSpace
        self.numActorFC1Unit = numActorFC1Unit
        self.learningRateActor = learningRateActor
        self.tau = tau

    def __call__(self, trainable):
        actorGraph = tf.Graph()
        with actorGraph.as_default():
            with tf.name_scope("inputs"):
                state_ = tf.placeholder(tf.float32, [None, self.numStateSpace])
                actionGradients_ = tf.placeholder(tf.float32, [None, self.numActionSpace])
                actorTrainParams_ = tf.placeholder(tf.float32, [None, self.numStateSpace])

                tf.add_to_collection("state_", state_)
                tf.add_to_collection("actionGradients_", actionGradients_)
                tf.add_to_collection("actorTrainParams_", actorTrainParams_)

            with tf.name_scope("trainingParams"):
                learningRate_ = tf.constant(0, dtype=tf.float32)
                actionLossCoef_ = tf.constant(1, dtype=tf.float32)
                valueLossCoef_ = tf.constant(1, dtype=tf.float32)
                tf.add_to_collection("learningRate", learningRate_)
                tf.add_to_collection("lossCoefs", actionLossCoef_)
                tf.add_to_collection("lossCoefs", valueLossCoef_)

            with tf.name_scope("hidden"):
                initWeight_ = tf.random_uniform_initializer(-0.03, 0.03)
                initBias_ = tf.constant_initializer(0.01)
                fullyConnected1_ = tf.layers.dense(inputs=state_, units=self.numActorFC1Unit, activation=tf.nn.relu,
                                                   kernel_initializer=initWeight_, bias_initializer=initBias_,
                                                   name='fullyConnected1_', trainable=trainable)
                actionsOutput_ = tf.layers.dense(inputs=fullyConnected1_, units=self.numActionSpace,
                                                 activation=tf.nn.tanh,
                                                 kernel_initializer=initWeight_, bias_initializer=initBias_,
                                                 name='actions', trainable=trainable)
                tf.add_to_collection("actionsOutput_", actionsOutput_)

            with tf.name_scope("parameters"):
                actorCurrentModelParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='hidden')
                replaceParam_ = [tf.assign(targetParam, (1 - self.tau) * targetParam + self.tau * trainParam)
                                 for targetParam, trainParam in zip(actorCurrentModelParams_, actorTrainParams_)]

                tf.add_to_collection("actorCurrentModelParams_", actorCurrentModelParams_)
                tf.add_to_collection("replaceParam_", replaceParam_)

            with tf.name_scope("policyGradient"):
                actorParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='hidden')
                policyGradient_ = tf.gradients(ys=actionsOutput_, xs=actorParams_, grad_ys=actionGradients_)

                tf.add_to_collection("actorParams_", actorParams_)

            with tf.name_scope("train"):
                optimizer = tf.train.AdamOptimizer(learningRate_)
                trainOpt_ = optimizer.apply_gradients(list(zip(policyGradient_, actorParams_)))

                tf.add_to_collection("trainOpt_", trainOpt_)

            actorSaver = tf.train.Saver(tf.global_variables())
            actorInit = tf.global_variables_initializer()

        actorWriter = tf.summary.FileWriter('tensorBoard/actorOnlineDDPGtrain' + str(trainable), graph=actorGraph)
        actorModel = tf.Session(graph=actorGraph)
        actorModel.run(actorInit)
        return actorWriter, actorModel


class BuildCriticModel:
    def __init__(self, numStateSpace, numActionSpace, numCriticFC1Unit, learningRateCritic, tau, gamma):
        self.numStateSpace = numStateSpace
        self.numActionSpace = numActionSpace
        self.numCriticFC1Unit = numCriticFC1Unit
        self.learningRateCritic = learningRateCritic
        self.tau = tau
        self.gamma = gamma

    def __call__(self, trainable):
        criticGraph = tf.Graph()
        with criticGraph.as_default():
            with tf.name_scope("inputs"):
                state_ = tf.placeholder(tf.float32, [None, self.numStateSpace])
                action_ = tf.placeholder(tf.float32, [None, self.numActionSpace])
                reward_ = tf.placeholder(tf.float32, [None, self.numStateSpace])
                valueTarget_ = tf.placeholder(tf.float32, [None, 1])
                criticTrainParams_ = tf.placeholder(tf.float32, [None, self.numActionSpace])

                tf.add_to_collection("state_", state_)
                tf.add_to_collection("action_", action_)
                tf.add_to_collection("reward_", reward_)
                tf.add_to_collection("valueTarget_", valueTarget_)
                tf.add_to_collection("criticTrainParams_", criticTrainParams_)

            with tf.name_scope("trainingParams"):
                learningRate_ = tf.constant(0, dtype=tf.float32)
                actionLossCoef_ = tf.constant(1, dtype=tf.float32)
                valueLossCoef_ = tf.constant(1, dtype=tf.float32)
                tf.add_to_collection("learningRate", learningRate_)
                tf.add_to_collection("lossCoefs", actionLossCoef_)
                tf.add_to_collection("lossCoefs", valueLossCoef_)


            with tf.variable_scope('hidden'):
                initWeight_ = tf.random_uniform_initializer(-0.03, 0.03)
                initBias_ = tf.constant_initializer(0.01)

                w1_s = tf.get_variable('w1_s', [self.numStateSpace, self.numCriticFC1Unit], initializer=initWeight_,
                                       trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.numActionSpace, self.numCriticFC1Unit], initializer=initWeight_,
                                       trainable=trainable)
                b1 = tf.get_variable('b1', [1, self.numCriticFC1Unit], initializer=initBias_, trainable=trainable)
                layer1 = tf.nn.relu(tf.matmul(state_, w1_s) + tf.matmul(action_, w1_a) + b1)
                qValue_ = tf.layers.dense(inputs=layer1, units=1, activation=None, name='value',
                                          kernel_initializer=initWeight_, bias_initializer=initBias_,
                                          trainable=trainable)  # Q(s,a)
                tf.add_to_collection("qValue_", qValue_)

            with tf.name_scope("parameters"):
                criticCurrentModelParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='hidden')
                replaceParam_ = [tf.assign(targetParam, (1 - self.tau) * targetParam + self.tau * trainParam)
                                 for targetParam, trainParam in zip(criticCurrentModelParams_, criticTrainParams_)]

                tf.add_to_collection("criticCurrentModelParams_", criticCurrentModelParams_)
                tf.add_to_collection("replaceParam_", replaceParam_)

            with tf.name_scope("actionGradients"):
                actionGradients_ = tf.gradients(qValue_, action_)[0]  # = a_grads in morvan
                tf.add_to_collection("actionGradients_", actionGradients_)

            with tf.name_scope("outputs"):
                yi_ = reward_ + self.gamma * valueTarget_
                diff_ = tf.subtract(yi_, qValue_, name='diff_')
                loss_ = tf.reduce_mean(tf.square(diff_), name='loss_')
                tf.add_to_collection("loss_", loss_)

            with tf.name_scope("train"):
                trainOpt_ = tf.train.AdamOptimizer(learningRate_, name='adamOpt_').minimize(loss_)
                tf.add_to_collection("trainOpt_", trainOpt_)

            criticInit = tf.global_variables_initializer()

        criticWriter = tf.summary.FileWriter('tensorBoard/criticOnlineDDPGtrain' + str(trainable), graph=criticGraph)
        criticModel = tf.Session(graph=criticGraph)
        criticModel.run(criticInit)
        return criticWriter, criticModel


class AddActionNoise():
    def __init__(self, actionNoise, noiseDecay, actionLow, actionHigh):
        self.actionNoise = actionNoise
        self.noiseDecay = noiseDecay
        self.actionLow, self.actionHigh = actionLow, actionHigh

    def __call__(self, actionPerfect, episodeIndex):
        noisyAction = np.random.normal(actionPerfect, self.actionNoise * (self.noiseDecay ** episodeIndex))
        action = np.clip(noisyAction, self.actionLow, self.actionHigh)
        return action


class Memory():
    def __init__(self, memoryCapacity):
        self.memoryCapacity = memoryCapacity

    def __call__(self, replayBuffer, timeStep):
        replayBuffer.append(timeStep)
        if len(replayBuffer) > self.memoryCapacity:
            numDelete = len(replayBuffer) - self.memoryCapacity
            del replayBuffer[numDelete:]
        return replayBuffer


def sampleFromMemory(buffer, batchSize):
    state_batch = []
    action_batch = []
    reward_batch = []
    next_state_batch = []
    done_batch = []

    sampledBatch = random.sample(buffer, batchSize)
    # reshapedBatch = np.concatenate(sampledBatch)
    # state_batch, action_batch, reward_batch, next_state_batch, done_batch = list(zip(*reshapedBatch))

    for experience in sampledBatch:
        state, action, reward, next_state, done = experience
        state_batch.append(state)
        action_batch.append(action)
        reward_batch.append(reward)
        next_state_batch.append(next_state)
        done_batch.append(done)

    return state_batch, action_batch, reward_batch, next_state_batch, done_batch


def addToMemory(buffer, state, action, reward, nextState, done):
    experience = (state, action, np.array([reward]), nextState, done)
    buffer.append(experience)
    return buffer


# class GetMemoryBuffer:
#     def __init__(self, addToMemory, addNoiseToAction):
#         self.addToMemory = addToMemory
#         self.addNoiseToAction = addNoiseToAction
#
#     def __call__(self, memoryBuffer, addBufferTimes, policy, state, transit, rewardFunction):
#         for step in range(addBufferTimes):
#             intendedAction = policy(state)
#             action = self.addNoiseToAction(intendedAction, step)
#             nextState = transit(state, action)
#             reward = rewardFunction(state, action)
#             memoryBuffer = addToMemory(memoryBuffer, state, action, reward, nextState, done)
#             return memoryBuffer

def initializeMemory(bufferSize):
    return deque(maxlen=bufferSize)


class ActWithNoise:
    def __init__(self, actByDeterministicPolicy, addActionNoise):
        self.actByDeterministicPolicy = actByDeterministicPolicy
        self.addActionNoise = addActionNoise

    def __call__(self, actorModel, stateBatch, timeStep):
        actionPerfect = actByDeterministicPolicy(actorModel, stateBatch)
        action = self.addActionNoise(actionPerfect, timeStep)
        return action


class RunDDPGTimeStep:
    def __init__(self, actWithNoise, actOneStep, addToMemory, sampleFromMemory, trainCritic, trainActor, minibatchSize, numStateSpace):
        self.actWithNoise = actWithNoise
        self.actOneStep = actOneStep
        self.addToMemory = addToMemory
        self.sampleFromMemory = sampleFromMemory
        self.trainCritic = trainCritic
        self.trainActor = trainActor
        self.minibatchSize = minibatchSize
        self.numStateSpace = numStateSpace

    def __call__(self, timeStep, state, actorModel, actorTargetModel, criticTargetModel, criticModel, replayBuffer):
        action = self.actWithNoise(actorModel, state, timeStep)
        nextState, reward, terminal, info = self.actOneStep(action)
        replayBuffer = self.addToMemory(replayBuffer, state, action, reward, nextState, terminal)

        stateBatch, actionBatch, rewardBatch, nextStateBatch = self.sampleFromMemory(replayBuffer, self.minibatchSize)
        criticLoss, criticModel, criticTargetModel = self.trainCritic(actorTargetModel, criticTargetModel, criticModel, nextStateBatch, rewardBatch, stateBatch,actionBatch)
        actorModel, actorTargetModel = self.trainActor(actorModel, criticModel, stateBatch, actorTargetModel)

        return actorModel, actorTargetModel, criticTargetModel, criticModel, replayBuffer


class RunEpisode:
    def __init__(self, reset, runTimeStep, maxTimeStep):
        self.reset = reset
        self.runTimeStep = runTimeStep
        self.maxTimeStep = maxTimeStep

    def __call__(self, actorModel, actorTargetModel, criticTargetModel, criticModel, replayBuffer):
        state = self.reset()
        for timeStep in range(self.maxTimeStep):
            actorModel, actorTargetModel, criticTargetModel, criticModel, replayBuffer = \
                self.runTimeStep(timeStep, state, actorModel, actorTargetModel, criticTargetModel, criticModel, replayBuffer)

        return actorModel, actorTargetModel, criticTargetModel, criticModel, replayBuffer



class DDPG:
    def __init__(self, initializeMemory, runEpisode, bufferSize, maxEpisode):
        self.bufferSize = bufferSize
        self.initializeMemory = initializeMemory
        self.runEpisode = runEpisode
        self.maxEpisode = maxEpisode

    def __call__(self, actorModel, actorTargetModel, criticModel, criticTargetModel, trainActor, trainCritic):
        replayBuffer = self.initializeMemory(self.bufferSize)
        for episode in range(self.maxEpisode):
            actorModel, actorTargetModel, criticTargetModel, criticModel, replayBuffer = \
                self.runEpisode(actorModel, actorTargetModel, criticTargetModel, criticModel, replayBuffer)

        return actorModel, criticModel


# input


def main():
    # tf.set_random_seed(123)
    # np.random.seed(123)

    actionSpace = [[10, 0], [7, 7], [0, 10], [-7, 7], [-10, 0], [-7, -7], [0, -10], [7, -7]]
    numActionSpace = len(actionSpace)
    numStateSpace = 4

    numActorFC1Unit = 50
    numCriticFC1Unit = 100
    learningRateActor = 1e-4
    learningRateCritic = 3e-4

    tau = 0.2
    gamma = 0.2

    buildActorModel = BuildActorModel(numStateSpace, numActionSpace, numActorFC1Unit, learningRateActor, tau)
    buildCriticModel = BuildCriticModel(numStateSpace, numActionSpace, numCriticFC1Unit, learningRateCritic, tau, gamma)

    actorWriter, actorModel = buildActorModel(trainable=True)
    actorTargetWriter, actorTargetModel = buildActorModel(trainable=False)

    criticWriter, criticModel = buildCriticModel(trainable=True)
    criticTargetWriter, criticTargetModel = buildCriticModel(trainable=False)

    trainActor = TrainActor(actorWriter, actByDeterministicPolicy, getActionGradients, updateActorParameter)
    trainCritic = TrainCritic(criticWriter, actByDeterministicPolicy, getCriticQValue, updateCriticParameter)

    #
    # addActionNoise = AddActionNoise(actionNoise, noiseDecay, actionLow, actionHigh)
    # ddpg = DDPG(buildActorModel, buildCriticModel, initializeMemory, sampleFromMemory, reset, addActionNoise)


# writer = tf.summary.FileWriter("logs/", criticModel.graph)


if __name__ == '__main__':
    main()
















