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
    def __init__(self, actByDeterministicPolicy, getCriticQValue, updateCriticParameter):
        self.actByDeterministicPolicy = actByDeterministicPolicy
        self.getCriticQValue = getCriticQValue
        self.updateCriticParameter = updateCriticParameter

    def __call__(self, actorTargetModel, criticModel, criticTargetModel, nextStateBatch, stateBatch, actionBatch, rewardBatch):
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

        criticLoss, criticParams, trainOpt = criticModel.run([loss_, criticParams_, trainOpt_],
                                                       feed_dict={state_: stateBatch,
                                                                  action_: actionBatch,
                                                                  reward_: rewardBatch,
                                                                  valueTarget_: targetQValue})

        criticTargetModel = self.updateCriticParameter(criticTargetModel, criticParams)

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
    def __init__(self, actByDeterministicPolicy, getActionGradients, updateActorParameter):
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
        actorParams, trainOpt = actorModel.run([actorParams_, trainOpt_],
                                               feed_dict={state_: stateBatch,
                                                          actionGradients_: actionGradients})
        actorTargetModel = self.updateActorParameter(actorTargetModel, actorParams)

        return trainOpt, actorTargetModel


class BuildActorModel:
    def __init__(self, numStateSpace, numActionSpace):
        self.numStateSpace = numStateSpace
        self.numActionSpace = numActionSpace

    def __call__(self, trainable):
        actorGraph = tf.Graph()
        with actorGraph.as_default():
            with tf.name_scope("inputs"):
                state_ = tf.placeholder(tf.float32, [None, self.numStateSpace], name="state_")
                actionGradients_ = tf.placeholder(tf.float32, [None, self.numActionSpace], name="actionGradients_")
                actorTrainParams_ = tf.placeholder(tf.float32, [None, self.numStateSpace], name="nextState_")

                tf.add_to_collection("state_", state_)
                tf.add_to_collection("actionGradients_", actionGradients_)
                tf.add_to_collection("actorTrainParams_", actorTrainParams_)

            with tf.name_scope("trainingParams"):
                initWeight_ = tf.random_uniform_initializer(-0.03, 0.03)
                initBias_ = tf.constant_initializer(0.01)
                learningRate_ = tf.constant(0, dtype=tf.float32)
                numActorFC1Unit_ = tf.constant(0, dtype=tf.float32)
                numActionSpace_ = tf.constant(0, dtype=tf.float32)
                # tau_ = tf.constant(self.tau, dtype=tf.float32)
                tau = tf.constant(0, dtype=tf.float32)

                tf.add_to_collection("initWeight", initWeight_)
                tf.add_to_collection("initBias", initBias_)
                tf.add_to_collection("learningRate", learningRate_)
                tf.add_to_collection("numActorFC1Unit", numActorFC1Unit_)
                tf.add_to_collection("numActionSpace", numActionSpace_)
                tf.add_to_collection("tau_", tau)

            with tf.name_scope("hidden"):
                fullyConnected1_ = tf.layers.dense(inputs=state_, units=numActorFC1Unit_, activation=tf.nn.relu,
                                                   kernel_initializer=initWeight_, bias_initializer=initBias_,
                                                   name='fullyConnected1_', trainable=trainable)
                actionsOutput_ = tf.layers.dense(inputs=fullyConnected1_, units=numActionSpace_, activation=tf.nn.tanh,
                                                 kernel_initializer=initWeight_, bias_initializer=initBias_,
                                                 name='actions', trainable=trainable)
                tf.add_to_collection("actionsOutput_", actionsOutput_)

            with tf.name_scope("parameters"):
                actorCurrentModelParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='hidden')
                replaceParam_ = [tf.assign(targetParam, (1 - tau) * targetParam + tau * trainParam)
                                     for targetParam, trainParam in zip(actorCurrentModelParams_, actorTrainParams_)]

                tf.add_to_collection("actorCurrentModelParams_", actorCurrentModelParams_)
                tf.add_to_collection("replaceParam_", replaceParam_)

            with tf.name_scope("policyGradient"):
                actorParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='hidden')
                policyGradient_ = tf.gradients(ys=actionsOutput_, xs=actorParams_, grad_ys=actionGradients_)

                tf.add_to_collection("actorParams_", actorParams_)

            with tf.name_scope("train"):
                optimizer = tf.train.AdamOptimizer(learningRate_)
                trainOpt_ = optimizer.apply_gradients(zip(policyGradient_, actorParams_))

                tf.add_to_collection("trainOpt_", trainOpt_)

            actorSaver = tf.train.Saver(tf.global_variables())
            actorInit = tf.global_variables_initializer()

        actorWriter = tf.summary.FileWriter('tensorBoard/actorOnlineDDPG', graph=actorGraph)
        actorModel = tf.Session(graph=actorGraph)
        actorModel.run(actorInit)
        actorWriter.flush()
        return actorModel


class BuildCriticModel:
    def __init__(self, numStateSpace, numActionSpace):
        self.numStateSpace = numStateSpace
        self.numActionSpace = numActionSpace

    def __call__(self, trainable):
        criticGraph = tf.Graph()
        with criticGraph.as_default():
            with tf.name_scope("inputs"):
                state_ = tf.placeholder(tf.float32, [None, self.numStateSpace], name="state_")
                action_ = tf.placeholder(tf.float32, [None, self.numActionSpace])
                reward_ = tf.placeholder(tf.float32, [None, self.numStateSpace], name="state_")
                valueTarget_ = tf.placeholder(tf.float32, [None, 1], name="valueTarget_")
                criticTrainParams_ = tf.placeholder(tf.float32, [None, self.numActionSpace])

                tf.add_to_collection("state_", state_)
                tf.add_to_collection("action_", action_)
                tf.add_to_collection("reward_", reward_)
                tf.add_to_collection("valueTarget_", valueTarget_)
                tf.add_to_collection("criticTrainParams_", criticTrainParams_)

            with tf.name_scope("trainingParams"):
                initWeight_ = tf.random_uniform_initializer(-0.03, 0.03)
                initBias_ = tf.constant_initializer(0.01)
                learningRate_ = tf.constant(0, dtype=tf.float32)
                numActorFC1Unit_ = tf.constant(0, dtype=tf.float32)
                numActionSpace_ = tf.constant(0, dtype=tf.float32)
                gamma = tf.constant(0, dtype=tf.float32)
                tau = tf.constant(0, dtype=tf.float32)

                tf.add_to_collection("initWeight", initWeight_)
                tf.add_to_collection("initBias", initBias_)
                tf.add_to_collection("learningRate", learningRate_)
                tf.add_to_collection("numActorFC1Unit", numActorFC1Unit_)
                tf.add_to_collection("numActionSpace", numActionSpace_)

            with tf.variable_scope('hidden'):
                w1_s = tf.get_variable('w1_s', [self.numStateSpace, numActorFC1Unit_], initializer=initWeight_,
                                       trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.numActionSpace, numActorFC1Unit_], initializer=initWeight_,
                                       trainable=trainable)
                b1 = tf.get_variable('b1', [1, numActorFC1Unit_], initializer=initBias_, trainable=trainable)
                layer1 = tf.nn.relu(tf.matmul(state_, w1_s) + tf.matmul(action_, w1_a) + b1)
                qValue_ = tf.layers.dense(inputs=layer1, units=1, activation=None, name='value',
                                          kernel_initializer=initWeight_, bias_initializer=initBias_,
                                          trainable=trainable)  # Q(s,a)
                tf.add_to_collection("qValue_", qValue_)

            with tf.name_scope("parameters"):
                criticCurrentModelParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='hidden')
                replaceParam_ = [tf.assign(targetParam, (1 - tau) * targetParam + tau * trainParam)
                                 for targetParam, trainParam in zip(criticCurrentModelParams_, criticTrainParams_)]

                tf.add_to_collection("criticCurrentModelParams_", criticCurrentModelParams_)
                tf.add_to_collection("replaceParam_", replaceParam_)

            with tf.name_scope("actionGradients"):
                actionGradients_ = tf.gradients(qValue_, action_)[0]  # = a_grads in morvan
                tf.add_to_collection("actionGradients_", actionGradients_)

            with tf.name_scope("outputs"):
                yi_ = reward_ + gamma * valueTarget_
                diff_ = tf.subtract(yi_, qValue_, name='diff_')
                loss_ = tf.reduce_mean(tf.square(diff_), name='loss_')
                tf.add_to_collection("loss_", loss_)

            with tf.name_scope("train"):
                trainOpt_ = tf.train.AdamOptimizer(learningRate_, name='adamOpt_').minimize(loss_)
                tf.add_to_collection("trainOpt_", trainOpt_)

            criticInit = tf.global_variables_initializer()
            criticModel = tf.Session(graph=criticGraph)
            criticModel.run(criticInit)
            return criticModel


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


class DDPG:
    def __init__(self, buildActorModel, buildCriticModel, initializeMemory, sampleFromMemory, reset, addActionNoise):
        self.buildActorModel = buildActorModel
        self.buildCriticModel = buildCriticModel
        self.initializeMemory = initializeMemory
        self.sampleFromMemory = sampleFromMemory
        self.reset = reset
        self.addActionNoise = addActionNoise

    def __call__(self, trainActor, trainCritic):
        actorModel = self.buildActorModel(trainable=True)
        actorTargetModel = self.buildActorModel(trainable=False)

        criticModel = self.buildCriticModel(trainable=True)
        criticTargetModel = self.buildCriticModel(trainable=False)

        bufferSize = 1000
        replayBuffer = self.initializeMemory(bufferSize)

        maxEpisode = 2000
        timeStepRange = 1000

        numStateSpace = 8
        minibatchSize = 200

        for episode in range(maxEpisode):
            state = self.reset()

            for timeStep in range(timeStepRange):

                stateBatch = np.reshape(state, (1, numStateSpace))
                actionPerfect = actByDeterministicPolicy(actorModel, stateBatch)
                action = self.addActionNoise(actionPerfect, timeStep)
                nextState, reward, terminal, info = env.step(action)
                replayBuffer = addToMemory(replayBuffer, state, action, reward, nextState, terminal)

                stateBatch, actionBatch, rewardBatch, nextStateBatch = self.sampleFromMemory(replayBuffer, minibatchSize)

                criticLoss, criticModel, criticTargetModel = trainCritic(actorTargetModel, criticTargetModel, criticModel,
                                             nextStateBatch, rewardBatch, stateBatch, actionBatch)

                actorModel, actorTargetModel = trainActor(actorModel, criticModel, stateBatch, actorTargetModel)

        return actorModel, criticModel




def main():
    # tf.set_random_seed(123)
    # np.random.seed(123)

    actionSpace = [[10, 0], [7, 7], [0, 10], [-7, 7], [-10, 0], [-7, -7], [0, -10], [7, -7]]
    numActionSpace = len(actionSpace)
    numStateSpace = 4

    numActorFC1Unit = 50
    numActorFC2Unit = 50
    numActorFC3Unit = 50
    numActorFC4Unit = 50
    numCriticFC1Unit = 100
    numCriticFC2Unit = 100
    numCriticFC3Unit = 100
    numCriticFC4Unit = 100
    learningRateActor = 1e-4
    learningRateCritic = 3e-4

    buildActorModel = BuildActorModel(numStateSpace, numActionSpace)
    buildCriticModel = BuildCriticModel(numStateSpace, numActionSpace)

    actorModel = buildActorModel(trainable=True)

    trainActor = TrainActor(actByDeterministicPolicy, getActionGradients, updateActorParameter)
    trainCritic = TrainCritic(actByDeterministicPolicy, getCriticQValue, updateCriticParameter)


    #
    # addActionNoise = AddActionNoise(actionNoise, noiseDecay, actionLow, actionHigh)
    # ddpg = DDPG(buildActorModel, buildCriticModel, initializeMemory, sampleFromMemory, reset, addActionNoise)

# writer = tf.summary.FileWriter("logs/", criticModel.graph)


if __name__ == '__main__':
    main()












#-------------------------------------------------------------------------------------


#     actorGraph = tf.Graph()
#     with actorGraph.as_default():
#         with tf.name_scope("inputs"):
#             trainable = True
#             state_ = tf.placeholder(tf.float32, [None, numStateSpace], name="state_")
#             actionGradients_ = tf.placeholder(tf.float32, [None, numActionSpace], name="actionGradients_")
#             # chosenActions_ = tf.placeholder(tf.float32, [None, numActionSpace], name="actionGradients_")
#
#             tf.add_to_collection("state_", state_)
#             tf.add_to_collection("actionGradients_", actionGradients_)
#             # tf.add_to_collection("chosenActions_", chosenActions_)
#
#
#         with tf.name_scope("trainingParams"):
#             initWeight_ = tf.random_uniform_initializer(-0.03, 0.03)
#             initBias_ = tf.constant_initializer(0.01)
#             learningRate_ = tf.constant(0, dtype=tf.float32)
#             numActorFC1Unit_ = tf.constant(0, dtype=tf.float32)
#             numActionSpace_ = tf.constant(0, dtype=tf.float32)
#
#             tf.add_to_collection("initWeight", initWeight_)
#             tf.add_to_collection("initBias", initBias_)
#             tf.add_to_collection("learningRate", learningRate_)
#             tf.add_to_collection("numActorFC1Unit", numActorFC1Unit_)
#             tf.add_to_collection("numActionSpace", numActionSpace_)
#
#         with tf.name_scope("hidden"):
#             fullyConnected1_ = tf.layers.dense(inputs=state_, units=numActorFC1Unit_, activation=tf.nn.relu,
#                                                kernel_initializer=initWeight_, bias_initializer=initBias_,
#                                                name='fullyConnected1_', trainable=trainable)
#             actionsOutput_ = tf.layers.dense(inputs=fullyConnected1_, units=numActionSpace_, activation=tf.nn.tanh,
#                                              kernel_initializer=initWeight_, bias_initializer=initBias_,
#                                              name='actions', trainable=trainable)
#             tf.add_to_collection("actionsOutput_", actionsOutput_)
#
#         with tf.name_scope("policyGradient"):
#             # evaluationParam_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net') [trainable network]
#             # policyGradient_ = tf.gradients(ys=actionsOutput_, xs=evaluationParam_, grad_ys=a_grads)
#             # opt = tf.train.AdamOptimizer(-self.lr)  # (- learning rate) for ascent policy
#             # self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))
#
#             # self.actor_gradients = tf.gradients(self.scaled_out, self.network_params, -self.action_gradient)
#
# # actionsOutput_ as placeholder
# #             actionGradients_ = tf.gradients(q_value, actionsOutput_)[0] # = a_grads in morvan
#             actorParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='hidden') # [trainable network]
#             policyGradient_ = tf.gradients(ys=actionsOutput_, xs=actorParams_, grad_ys=actionGradients_)
#
#         with tf.name_scope("train"):
#             # evaluationParam_ = tf.trainable_variables()
#             optimizer = tf.train.AdamOptimizer(learningRate_)  # (-self.lr)  # (- learning rate) for ascent policy
#             trainOpt_ = optimizer.apply_gradients(zip(policyGradient_, actorParams_))
#
#             tf.add_to_collection("actorParams_", actorParams_)
#             tf.add_to_collection("trainOpt_", trainOpt_)
#
#
#         actorInit = tf.global_variables_initializer()
#     actorInit = tf.global_variables_initializer()
#     actorModel = tf.Session(graph=actorGraph)
#     actorModel.run(actorInit)
#     actorTargetGraph = tf.Graph()
#
#     with actorTargetGraph.as_default():
#         with tf.name_scope("inputs"):
#             trainable = False
#             state_ = tf.placeholder(tf.float32, [None, numStateSpace], name="nextState_")
#             actorParams_ =  tf.placeholder(tf.float32, [None, numStateSpace], name="nextState_")
#
#             tf.add_to_collection("state_", state_)
#             tf.add_to_collection("actorParams_", actorParams_)
#
#         with tf.name_scope("trainingParams"):
#             initWeight_ = tf.random_uniform_initializer(-0.03, 0.03)
#             initBias_ = tf.constant_initializer(0.01)
#             learningRate_ = tf.constant(0, dtype=tf.float32)
#             numActorFC1Unit_ = tf.constant(0, dtype=tf.float32)
#             numActionSpace_ = tf.constant(0, dtype=tf.float32)
#             tau_ = tf.constant(tau, dtype=tf.float32)
#
#             tf.add_to_collection("initWeight", initWeight_)
#             tf.add_to_collection("initBias", initBias_)
#             tf.add_to_collection("learningRate", learningRate_)
#             tf.add_to_collection("numActorFC1Unit", numActorFC1Unit_)
#             tf.add_to_collection("numActionSpace", numActionSpace_)
#
#         with tf.name_scope("hidden"):
#             fullyConnected1_ = tf.layers.dense(inputs=state_, units=numActorFC1Unit_, activation=tf.nn.relu,
#                                                kernel_initializer=initWeight_, bias_initializer=initBias_,
#                                                name='fullyConnected1_', trainable=trainable)
#             actionsOutput_ = tf.layers.dense(inputs=fullyConnected1_, units=numActionSpace_, activation=tf.nn.tanh,
#                                        kernel_initializer=initWeight_, bias_initializer=initBias_,
#                                        name='actions', trainable=trainable)
#             actorTargetParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='hidden') # [trainable network]
#
#             tf.add_to_collection("actionsOutput_", actionsOutput_)
#
#             replaceParam_ = [tf.assign(targetParam, (1 - tau_) * targetParam + tau_ * trainParam)
#                                  for targetParam, trainParam in zip(actorTargetParams_, actorParams_)]
#
#             tf.add_to_collection("replaceParam_", replaceParam_)
#
#
#         actorInit = tf.global_variables_initializer()
#
#     criticGraph = tf.Graph()
#     with criticGraph.as_default():
#         with tf.name_scope("inputs"):
#             state_ = tf.placeholder(tf.float32, [None, numStateSpace], name="state_")
#             action_ = tf.placeholder(tf.float32, [None, numActionSpace])
#             reward_ = tf.placeholder(tf.float32, [None, numStateSpace], name="state_")
#             valueTarget_ = tf.placeholder(tf.float32, [None, 1], name="valueTarget_")
#             trainable = True
#
#             tf.add_to_collection("state_", state_)
#             tf.add_to_collection("action_", action_)
#             tf.add_to_collection("reward_", reward_)
#             tf.add_to_collection("valueTarget_", valueTarget_)
#
#         with tf.name_scope("trainingParams"):
#             initWeight_ = tf.random_uniform_initializer(-0.03, 0.03)
#             initBias_ = tf.constant_initializer(0.01)
#             learningRate_ = tf.constant(0, dtype=tf.float32)
#             numActorFC1Unit_ = tf.constant(0, dtype=tf.float32)
#             numActionSpace_ = tf.constant(0, dtype=tf.float32)
#             gamma = tf.constant(0, dtype=tf.float32)
#
#             tf.add_to_collection("initWeight", initWeight_)
#             tf.add_to_collection("initBias", initBias_)
#             tf.add_to_collection("learningRate", learningRate_)
#             tf.add_to_collection("numActorFC1Unit", numActorFC1Unit_)
#             tf.add_to_collection("numActionSpace", numActionSpace_)
#
#         with tf.variable_scope('hidden'):
#             w1_s = tf.get_variable('w1_s', [numStateSpace, numActorFC1Unit_], initializer=initWeight_, trainable=trainable)
#             w1_a = tf.get_variable('w1_a', [numActionSpace, numActorFC1Unit_], initializer=initWeight_, trainable=trainable)
#             b1 = tf.get_variable('b1', [1, numActorFC1Unit_], initializer=initBias_, trainable=trainable)
#             layer1 = tf.nn.relu(tf.matmul(state_, w1_s) + tf.matmul(action_, w1_a) + b1)
#             qValue_ = tf.layers.dense(inputs=layer1, units=1, activation=None, name='value',
#                                      kernel_initializer=initWeight_, bias_initializer=initBias_,
#                                      trainable=trainable)  # Q(s,a)
#
#         with tf.name_scope("outputs"):
#             criticParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='hidden')
#
#             actionGradients_ = tf.gradients(qValue_, action_)[0] # = a_grads in morvan
#
#             yi_ = reward_ + gamma * valueTarget_
#             diff_ = tf.subtract(yi_, qValue_, name='diff_')
#             loss_ = tf.reduce_mean(tf.square(diff_), name='loss_')
#
#             tf.add_to_collection("criticParams_", criticParams_)
#             tf.add_to_collection("actionGradients_", actionGradients_)
#             tf.add_to_collection("loss_", loss_)
#
#         with tf.name_scope("train"):
#             trainOpt_ = tf.train.AdamOptimizer(learningRate_, name='adamOpt_').minimize(loss_)
#             tf.add_to_collection("trainOpt_", trainOpt_)
#
#         criticInit = tf.global_variables_initializer()
#
#     criticModel = tf.Session(graph=criticGraph)
#     criticModel.run(criticInit)
#
#     criticTargetGraph = tf.Graph()
#     with criticTargetGraph.as_default():
#         with tf.name_scope("inputs"):
#             state_ = tf.placeholder(tf.float32, [None, numStateSpace], name="state_")
#             action_ = tf.placeholder(tf.float32, [None, numActionSpace])
#             criticParams_ = tf.placeholder(tf.float32, [None, numActionSpace])
#             trainable = False
#
#             tf.add_to_collection("state_", state_)
#             tf.add_to_collection("action_", action_)
#             tf.add_to_collection("criticParams_", criticParams_)
#
#         with tf.name_scope("trainingParams"):
#             initWeight_ = tf.random_uniform_initializer(-0.03, 0.03)
#             initBias_ = tf.constant_initializer(0.01)
#             learningRate_ = tf.constant(0, dtype=tf.float32)
#             numActorFC1Unit_ = tf.constant(0, dtype=tf.float32)
#             numActionSpace_ = tf.constant(0, dtype=tf.float32)
#             gamma = tf.constant(0, dtype=tf.float32)
#
#             tf.add_to_collection("initWeight", initWeight_)
#             tf.add_to_collection("initBias", initBias_)
#             tf.add_to_collection("learningRate", learningRate_)
#             tf.add_to_collection("numActorFC1Unit", numActorFC1Unit_)
#             tf.add_to_collection("numActionSpace", numActionSpace_)
#
#         with tf.variable_scope('hidden'):
#             w1_s = tf.get_variable('w1_s', [numStateSpace, numActorFC1Unit_], initializer=initWeight_, trainable=trainable)
#             w1_a = tf.get_variable('w1_a', [numActionSpace, numActorFC1Unit_], initializer=initWeight_, trainable=trainable)
#             b1 = tf.get_variable('b1', [1, numActorFC1Unit_], initializer=initBias_, trainable=trainable)
#             layer1 = tf.nn.relu(tf.matmul(state_, w1_s) + tf.matmul(action_, w1_a) + b1)
#             qValue_ = tf.layers.dense(inputs=layer1, units=1, activation=None, name='value',
#                                      kernel_initializer=initWeight_, bias_initializer=initBias_,
#                                      trainable=trainable)  # Q(s,a)
#
#             tf.add_to_collection("qValue_", qValue_)
#
#             criticTargetParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='hidden') # [trainable network]
#             tf.add_to_collection("criticTargetParams_", criticTargetParams_)
#
#             replaceParam_ = [tf.assign(targetParam, (1 - tau_) * targetParam + tau_ * trainParam)
#                                  for targetParam, trainParam in zip(criticTargetParams_, criticParams_)]
#
#             tf.add_to_collection("replaceParam_", replaceParam_)
#
#
#         criticInit = tf.global_variables_initializer()














