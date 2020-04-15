
import tensorflow as tf
import numpy as np
from collections import deque
import random

def actByPolicyTrain(actorModel, stateBatch):
    actorGraph = actorModel.graph
    states_ = actorGraph.get_collection_ref("states_")[0]
    trainAction_ = actorGraph.get_collection_ref("trainAction_")[0]
    trainAction = actorModel.run(trainAction_, feed_dict={states_: stateBatch})

    return trainAction

def actByPolicyTarget(actorModel, stateBatch):
    actorGraph = actorModel.graph
    states_ = actorGraph.get_collection_ref("states_")[0]
    targetAction_ = actorGraph.get_collection_ref("targetAction_")[0]
    targetAction = actorModel.run(targetAction_, feed_dict={states_: stateBatch})
    return targetAction

def evaluateCriticTarget(criticModel, stateBatch, actionsBatch):
    criticGraph = criticModel.graph
    states_ = criticGraph.get_collection_ref("states_")[0]
    action_ = criticGraph.get_collection_ref("action_")[0]
    targetValues_ = criticGraph.get_collection_ref("targetValues_")[0]
    targetValues = criticModel.run(targetValues_, feed_dict={states_: stateBatch, action_: actionsBatch})
    return targetValues

def getActionGradients(criticModel, stateBatch, actionsBatch):
    criticGraph = criticModel.graph
    states_ = criticGraph.get_collection_ref("states_")[0]
    action_ = criticGraph.get_collection_ref("action_")[0]
    actionGradients_ = criticGraph.get_collection_ref("actionGradients_")[0]
    actionGradients = criticModel.run(actionGradients_, feed_dict={states_: stateBatch,
                                                                   action_: actionsBatch})
    return actionGradients


class BuildActorModel:
    def __init__(self, numStateSpace, actionDim, actionRange, seed=128):
        self.numStateSpace = numStateSpace
        self.actionDim = actionDim
        self.actionRange = actionRange
        self.seed = seed

    def __call__(self, trainingLayerWidths, targetLayerWidths, summaryPath="./tbdata"):
        print("Generating Actor NN Model with training layers: {}, target layers: {}".format(trainingLayerWidths, targetLayerWidths))
        graph = tf.Graph()
        with graph.as_default():
            if self.seed is not None:
                tf.set_random_seed(self.seed)

            with tf.name_scope("inputs"):
                # states_ = tf.layers.batch_normalization(tf.placeholder(tf.float32, [None, self.numStateSpace]))
                # states_ = tf.placeholder(tf.float32, [None, self.numStateSpace])
                states_ = tf.placeholder(tf.float32, [None, self.numStateSpace])

                actionGradients_ = tf.placeholder(tf.float32, [None, self.actionDim])
                tf.add_to_collection("states_", states_)
                tf.add_to_collection("actionGradients_", actionGradients_)

            with tf.name_scope("trainingParams"):
                states_ = tf.layers.batch_normalization(states_)

                learningRate_ = tf.constant(0, dtype=tf.float32)
                tau_ = tf.constant(0, dtype=tf.float32)
                tf.add_to_collection("learningRate_", learningRate_)
                tf.add_to_collection("tau_", tau_)

            initWeight = tf.random_uniform_initializer(-0.03, 0.03)
            initBias = tf.constant_initializer(0.001)
            with tf.variable_scope("trainHidden"):
                activation_ = states_
                for i in range(len(trainingLayerWidths)):
                    fcLayer = tf.layers.Dense(units=trainingLayerWidths[i], activation=tf.nn.relu, kernel_initializer=initWeight,
                                              bias_initializer=initBias, name="fc{}".format(i+1))
                    activation_ = tf.layers.batch_normalization(fcLayer(activation_))
                    # activation_ = fcLayer(activation_)

                    tf.add_to_collections(["weights", f"weight/{fcLayer.kernel.name}"], fcLayer.kernel)
                    tf.add_to_collections(["biases", f"bias/{fcLayer.bias.name}"], fcLayer.bias)
                    tf.add_to_collections(["activations", f"activation/{activation_.name}"], activation_)
                trainActivation_ = tf.identity(activation_, name="output")
                outputFCLayer = tf.layers.Dense(units=self.actionDim, activation= tf.nn.tanh, kernel_initializer=initWeight,
                                                bias_initializer=initBias,name="fc{}".format(len(trainingLayerWidths) + 1))
                # trainActivationOutput_ = outputFCLayer(trainActivation_)
                trainActivationOutput_ = tf.layers.batch_normalization(outputFCLayer(trainActivation_))

                tf.add_to_collections(["weights", f"weight/{outputFCLayer.kernel.name}"], outputFCLayer.kernel)
                tf.add_to_collections(["biases", f"bias/{outputFCLayer.bias.name}"], outputFCLayer.bias)
                tf.add_to_collections(["activations", f"activation/{trainActivationOutput_.name}"], trainActivationOutput_)

            with tf.variable_scope("targetHidden"):
                activation_ = states_
                for i in range(len(targetLayerWidths)):
                    fcLayer = tf.layers.Dense(units=targetLayerWidths[i], activation=tf.nn.relu, kernel_initializer=initWeight,
                                              bias_initializer=initBias, name="fc{}".format(i+1))
                    activation_ = tf.layers.batch_normalization(fcLayer(activation_))
                    # activation_ = fcLayer(activation_)

                    tf.add_to_collections(["weights", f"weight/{fcLayer.kernel.name}"], fcLayer.kernel)
                    tf.add_to_collections(["biases", f"bias/{fcLayer.bias.name}"], fcLayer.bias)
                    tf.add_to_collections(["activations", f"activation/{activation_.name}"], activation_)
                targetActivation_ = tf.identity(activation_, name="output")
                outputFCLayer = tf.layers.Dense(units=self.actionDim, activation=tf.nn.tanh, kernel_initializer=initWeight,
                                                bias_initializer=initBias,name="fc{}".format(len(trainingLayerWidths) + 1))
                # targetActivationOutput_ = outputFCLayer(targetActivation_)
                targetActivationOutput_ = tf.layers.batch_normalization(outputFCLayer(targetActivation_))


                tf.add_to_collections(["weights", f"weight/{outputFCLayer.kernel.name}"], outputFCLayer.kernel)
                tf.add_to_collections(["biases", f"bias/{outputFCLayer.bias.name}"], outputFCLayer.bias)
                tf.add_to_collections(["activations", f"activation/{targetActivationOutput_.name}"], targetActivationOutput_)

            # with tf.name_scope("updateParameters"):
            #     trainParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='trainHidden')
            #     targetParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='targetHidden')
            #     numParams_ = tf.constant(len(trainParams_), name='numParams_')
            #     updateTargetParameter_ = [tf.assign(tarParam_, (1 - tau_) * tarParam_ + tau_ * evaParam_)
            #                               for tarParam_, evaParam_ in zip(targetParams_, trainParams_)]
            #
            #     tf.add_to_collection("trainParams_", trainParams_)
            #     tf.add_to_collection("targetParams_", targetParams_)
            #     for i in range(numParams_):
            #         tf.add_to_collection("assign" + str(i), updateTargetParameter_[i])

            with tf.name_scope("updateParameters"):
                trainParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='trainHidden')
                targetParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='targetHidden')
                updateParameters_ = [targetParams_[i].assign((1 - tau_) * targetParams_[i] + tau_ * trainParams_[i]) for i in range(len(targetParams_))]

                tf.add_to_collection("trainParams_", trainParams_)
                tf.add_to_collection("targetParams_", targetParams_)
                tf.add_to_collection("updateParameters_", updateParameters_)


            with tf.name_scope("output"):
                trainAction_ = tf.multiply(trainActivationOutput_, self.actionRange, name='trainAction_')
                targetAction_ = tf.multiply(targetActivationOutput_, self.actionRange, name='targetAction_')
                policyGradient_ = tf.gradients(ys=trainAction_, xs=trainParams_, grad_ys= actionGradients_)

                tf.add_to_collection("trainAction_", trainAction_)
                tf.add_to_collection("targetAction_", targetAction_)
                tf.add_to_collection("policyGradient_", policyGradient_)

            with tf.name_scope("train"):
                optimizer = tf.train.AdamOptimizer(-learningRate_, name='adamOptimizer')
                trainOpt_ = optimizer.apply_gradients(list(zip(policyGradient_, trainParams_)))
                tf.add_to_collection("trainOpt_", trainOpt_)


            fullSummary = tf.summary.merge_all()
            tf.add_to_collection("summaryOps", fullSummary)


            # actorSaver = tf.train.Saver(tf.global_variables())
            actorSaver = tf.train.Saver(max_to_keep=None)
            model = tf.Session(graph=graph)
            model.run(tf.global_variables_initializer())

        actorWriter = tf.summary.FileWriter('tensorBoard/actorOnlineDDPG', graph= graph)
        return actorSaver, actorWriter, model


class BuildCriticModel:
    def __init__(self, numStateSpace, actionDim, seed=128):
        self.numStateSpace = numStateSpace
        self.actionDim = actionDim
        self.seed = seed

    def __call__(self, trainingLayerWidths, targetLayerWidths, summaryPath="./tbdata"):
        print("Generating Critic NN Model with training layers: {}, target layers: {}".format(trainingLayerWidths, targetLayerWidths))
        graph = tf.Graph()
        with graph.as_default():
            if self.seed is not None:
                tf.set_random_seed(self.seed)

            with tf.name_scope("inputs"):
                states_ = tf.placeholder(tf.float32, [None, self.numStateSpace])
                action_ = tf.stop_gradient(tf.placeholder(tf.float32, [None, self.actionDim]))

                reward_ = tf.placeholder(tf.float32, [None, 1])
                valueTarget_ = tf.placeholder(tf.float32, [None, 1])

                tf.add_to_collection("states_", states_)
                tf.add_to_collection("action_", action_)
                tf.add_to_collection("reward_", reward_)
                tf.add_to_collection("valueTarget_", valueTarget_)

            with tf.name_scope("trainingParams"):
                states_ = tf.layers.batch_normalization(states_)

                learningRate_ = tf.constant(0, dtype=tf.float32)
                tau_ = tf.constant(0, dtype=tf.float32)
                gamma_ = tf.constant(0, dtype=tf.float32)

                tf.add_to_collection("learningRate_", learningRate_)
                tf.add_to_collection("tau_", tau_)
                tf.add_to_collection("gamma_", gamma_)

            initWeight = tf.random_uniform_initializer(-0.03, 0.03)
            initBias = tf.constant_initializer(0.001)
            with tf.variable_scope("trainHidden"):
                activation_ = states_
                for i in range(len(trainingLayerWidths)-1):
                    fcLayer = tf.layers.Dense(units=trainingLayerWidths[i], activation=tf.nn.relu, kernel_initializer=initWeight,
                                              bias_initializer=initBias, name="fc{}".format(i+1))
                    activation_ = tf.layers.batch_normalization(fcLayer(activation_))
                    # activation_ = fcLayer(activation_)

                    tf.add_to_collections(["weights", f"weight/{fcLayer.kernel.name}"], fcLayer.kernel)
                    tf.add_to_collections(["biases", f"bias/{fcLayer.bias.name}"], fcLayer.bias)
                    tf.add_to_collections(["activations", f"activation/{activation_.name}"], activation_)
                trainActivation_ = tf.identity(activation_, name="output")

                secondLastFCUnit = trainingLayerWidths[-2]
                lastFCUnit = trainingLayerWidths[-1]
                trainStateFCToLastFCWeights_ = tf.get_variable(name='trainStateFCToLastFCWeights_', shape=[secondLastFCUnit,lastFCUnit], initializer=initWeight)
                trainActionFCToLastFCWeights_ = tf.get_variable(name='evaActionToFullyConnected2Weights', shape=[self.actionDim, lastFCUnit], initializer=initWeight)
                trainActionLastFCBias_ = tf.get_variable(name='trainActionLastFCBias_', shape=[lastFCUnit], initializer=initBias)

                trainLastFCActivation_ = tf.nn.relu(tf.matmul(trainActivation_, trainStateFCToLastFCWeights_) +
                                          tf.matmul(action_, trainActionFCToLastFCWeights_) +
                                          trainActionLastFCBias_)

                tf.add_to_collections(["weights", "weight/trainStateFCToLastFCWeights_"], trainStateFCToLastFCWeights_)
                tf.add_to_collections(["weights", "weight/trainActionFCToLastFCWeights_"], trainActionFCToLastFCWeights_)
                tf.add_to_collections(["biases", "bias/trainActionLastFCBias_"], trainActionLastFCBias_)
                tf.add_to_collections(["activations", "activation/trainLastFCActivation_"], trainLastFCActivation_)

                trainOutputLayer = tf.layers.Dense(units=1, kernel_initializer=initWeight, bias_initializer=initBias, activation=None)
                trainValues_ = trainOutputLayer(trainLastFCActivation_)
                tf.add_to_collections(["weights", f"weight/{trainOutputLayer.kernel.name}"], trainOutputLayer.kernel)
                tf.add_to_collections(["biases", f"bias/{trainOutputLayer.bias.name}"], trainOutputLayer.bias)
                tf.add_to_collections(["activations", f"activation/{trainValues_.name}"], trainValues_)
                tf.add_to_collection("trainValues_", trainValues_)

            with tf.variable_scope("targetHidden"):
                activation_ = states_
                for i in range(len(targetLayerWidths)-1):
                    fcLayer = tf.layers.Dense(units=targetLayerWidths[i], activation=tf.nn.relu, kernel_initializer=initWeight,
                                              bias_initializer=initBias, name="fc{}".format(i+1))
                    activation_ = tf.layers.batch_normalization(fcLayer(activation_))
                    # activation_ = fcLayer(activation_)

                    tf.add_to_collections(["weights", f"weight/{fcLayer.kernel.name}"], fcLayer.kernel)
                    tf.add_to_collections(["biases", f"bias/{fcLayer.bias.name}"], fcLayer.bias)
                    tf.add_to_collections(["activations", f"activation/{activation_.name}"], activation_)
                targetActivation_ = tf.identity(activation_, name="output")

                secondLastFCUnit = targetLayerWidths[-2] # 3
                lastFCUnit = targetLayerWidths[-1] # 5
                targetStateFCToLastFCWeights_ = tf.get_variable(name='targetStateFCToLastFCWeights_', shape=[secondLastFCUnit, lastFCUnit], initializer=initWeight)
                targetActionFCToLastFCWeights_ = tf.get_variable(name='evaActionToFullyConnected2Weights', shape=[self.actionDim, lastFCUnit], initializer=initWeight)
                targetActionLastFCBias_ = tf.get_variable(name='targetActionLastFCBias_', shape=[lastFCUnit], initializer=initBias)
                targetLastFCActivation_ = tf.nn.relu(tf.matmul(targetActivation_, targetStateFCToLastFCWeights_) +
                                           tf.matmul(action_,targetActionFCToLastFCWeights_) +
                                           targetActionLastFCBias_)

                tf.add_to_collections(["weights", "weight/targetStateFCToLastFCWeights_"], targetStateFCToLastFCWeights_)
                tf.add_to_collections(["weights", "weight/targetActionFCToLastFCWeights_"], targetActionFCToLastFCWeights_)
                tf.add_to_collections(["biases", "bias/targetActionLastFCBias_"], targetActionLastFCBias_)
                tf.add_to_collections(["activations", "activation/targetLastFCActivation_"], targetLastFCActivation_)

                targetOutputLayer = tf.layers.Dense(units=1, kernel_initializer=initWeight, bias_initializer=initBias, activation=None)
                targetValues_ = targetOutputLayer(targetLastFCActivation_)
                tf.add_to_collections(["weights", f"weight/{targetOutputLayer.kernel.name}"], targetOutputLayer.kernel)
                tf.add_to_collections(["biases", f"bias/{targetOutputLayer.bias.name}"], targetOutputLayer.bias)
                tf.add_to_collections(["activations", f"activation/{targetValues_.name}"], targetValues_)
                tf.add_to_collection("targetValues_", targetValues_)

            # with tf.name_scope("parameters"):
            #     trainParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='trainHidden')
            #     targetParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='targetHidden')
            #     numParams_ = tf.constant(len(trainParams_), name='numParams_')
            #     updateTargetParameter_ = [tf.assign(tarParam_, (1 - tau_) * tarParam_ + tau_ * evaParam_)
            #                               for tarParam_, evaParam_ in zip(targetParams_, trainParams_)]
            #
            #     tf.add_to_collection("trainParams_", trainParams_)
            #     tf.add_to_collection("targetParams_", targetParams_)
            #     for i in range(numParams_):
            #         tf.add_to_collection("assign"+ str(i), updateTargetParameter_[i])

            with tf.name_scope("parameters"):
                trainParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='trainHidden')
                targetParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='targetHidden')
                updateParam_ = [targetParams_[i].assign((1 - tau_) * targetParams_[i] + tau_ * trainParams_[i]) for i in range(len(targetParams_))]

                tf.add_to_collection("trainParams_", trainParams_)
                tf.add_to_collection("targetParams_", targetParams_)
                tf.add_to_collection("updateParam_", updateParam_)

            with tf.name_scope("actionGradients"):
                actionGradients_ = tf.gradients(trainValues_, action_)[0]  # = a_grads in morvan
                tf.add_to_collection("actionGradients_", actionGradients_)

            with tf.name_scope("output"):
                trainQ_ = tf.multiply(trainValues_, 1, name='trainQ_')
                targetQ_ = tf.multiply(targetValues_, 1, name='targetQ_')
                tf.add_to_collection("trainQ_", trainQ_)
                tf.add_to_collection("targetQ_", targetQ_)


            with tf.name_scope("evaluate"):
                yi_ = reward_ + gamma_ * valueTarget_
                valueLoss_ = tf.reduce_mean(tf.squared_difference(yi_, trainValues_))

                tf.add_to_collection("valueLoss_", valueLoss_)

            with tf.name_scope("train"):
                trainOpt_ = tf.train.AdamOptimizer(learningRate_, name='adamOptimizer').minimize(valueLoss_)
                tf.add_to_collection("trainOpt_", trainOpt_)

            fullSummary = tf.summary.merge_all()
            tf.add_to_collection("summaryOps", fullSummary)

            criticSaver = tf.train.Saver(max_to_keep=None)
            model = tf.Session(graph=graph)
            model.run(tf.global_variables_initializer())

        criticWriter = tf.summary.FileWriter('tensorBoard/criticOnlineDDPG', graph=graph)
        return criticSaver, criticWriter, model


class TrainCritic:
    def __init__(self, criticLearningRate, gamma, tau, criticWriter, actByPolicyTarget, rewardFunction, evaluateCriticTarget):
        self.criticLearningRate = criticLearningRate
        self.gamma = gamma
        self.tau = tau
        self.criticWriter = criticWriter
        self.actByPolicyTarget = actByPolicyTarget
        self.rewardFunction = rewardFunction
        self.evaluateCriticTarget = evaluateCriticTarget

    def __call__(self, actorModel, criticModel, miniBatch, eposideID):
        states, actions, nextStates = list(zip(*miniBatch))
        rewards = np.array([self.rewardFunction(state, action) for state, action in zip(states, actions)])
        stateBatch = np.asarray(states).reshape(len(miniBatch), -1)
        actionBatch = np.asarray(actions).reshape(len(miniBatch), -1)
        nextStateBatch = np.asarray(nextStates).reshape(len(miniBatch), -1)
        rewardBatch = rewards.reshape(len(miniBatch), -1)

        targetNextActionBatch = self.actByPolicyTarget(actorModel, nextStateBatch)
        targetQValue = self.evaluateCriticTarget(criticModel, nextStateBatch, targetNextActionBatch)

        criticGraph = criticModel.graph
        states_ = criticGraph.get_collection_ref("states_")[0]
        action_ = criticGraph.get_collection_ref("action_")[0]
        reward_ = criticGraph.get_collection_ref("reward_")[0]
        valueTarget_ = criticGraph.get_collection_ref("valueTarget_")[0]
        learningRate_ = criticGraph.get_collection_ref("learningRate_")[0]
        tau_ = criticGraph.get_collection_ref("tau_")[0]
        gamma_ = criticGraph.get_collection_ref("gamma_")[0]

        # valueLoss_ = criticGraph.get_collection_ref("valueLoss_")[0]
        # trainOpt_ = criticGraph.get_collection_ref("trainOpt_")[0]
        # numParams_ = criticGraph.get_collection_ref("numParams_")[0]
        # criticLoss, trainOpt, numParams = criticModel.run([valueLoss_, trainOpt_, numParams_],
        #                                        feed_dict={states_: stateBatch, action_: actionBatch, reward_: rewardBatch, valueTarget_: targetQValue,
        #                                                   learningRate_: self.criticLearningRate, tau_: self.tau, gamma_: self.gamma})
        #
        # updateTargetParameter_ = [criticGraph.get_collection_ref("assign"+ str(i))[0] for i in range(numParams)]
        # criticModel.run(updateTargetParameter_)
        # self.criticWriter.flush()
        # return criticLoss, criticModel

        valueLoss_ = criticGraph.get_collection_ref("valueLoss_")[0]
        trainOpt_ = criticGraph.get_collection_ref("trainOpt_")[0]
        criticLoss, trainOpt = criticModel.run([valueLoss_, trainOpt_],
                                               feed_dict={states_: stateBatch, action_: actionBatch, reward_: rewardBatch, valueTarget_: targetQValue,
                                                          learningRate_: self.criticLearningRate, tau_: self.tau, gamma_: self.gamma})
        updateParam_ = criticGraph.get_collection_ref("updateParam_")[0]
        criticModel.run(updateParam_)

        ######
        summary = tf.Summary()
        summary.value.add(tag='reward', simple_value=float(np.mean(rewardBatch)))
        summary.value.add(tag='loss', simple_value=float(criticLoss))
        ###
        self.criticWriter.add_summary(summary, eposideID)
        self.criticWriter.flush()

        return criticLoss, criticModel



class TrainActor:
    def __init__(self, actorLearningRate, tau, actorWriter, actByPolicyTrain, getActionGradients):
        self.actorLearningRate = actorLearningRate
        self.tau = tau
        self.actorWriter = actorWriter
        self.actByPolicyTrain = actByPolicyTrain
        self.getActionGradients = getActionGradients

    def __call__(self, actorModel, criticModel, miniBatch):
        states, actions, nextStates = list(zip(*miniBatch))
        stateBatch = np.asarray(states).reshape(len(miniBatch), -1)

        actionsBatch = self.actByPolicyTrain(actorModel, stateBatch)
        actionGradients = self.getActionGradients(criticModel, stateBatch, actionsBatch)

        actorGraph = actorModel.graph
        states_ = actorGraph.get_collection_ref("states_")[0]
        actionGradients_ = actorGraph.get_collection_ref("actionGradients_")[0]
        tau_ = actorGraph.get_collection_ref("tau_")[0]
        learningRate_ = actorGraph.get_collection_ref("learningRate_")[0]

        trainOpt_ = actorGraph.get_collection_ref("trainOpt_")[0]
        trainOpt = actorModel.run([trainOpt_], feed_dict={states_: stateBatch, actionGradients_: actionGradients,
                                                        tau_: self.tau, learningRate_: self.actorLearningRate})

        updateParameters_ = actorGraph.get_collection_ref("updateParam_")
        actorModel.run(updateParameters_)

######
        self.actorWriter.flush()
        return trainOpt, actorModel


class AddActionNoise():
    def __init__(self, actionNoise, noiseDecay, actionLow, actionHigh):
        self.actionNoise = actionNoise
        self.noiseDecay = noiseDecay
        self.actionLow, self.actionHigh = actionLow, actionHigh

    def __call__(self, actionPerfect, episodeIndex):
        noisyAction = np.random.normal(actionPerfect, self.actionNoise * (self.noiseDecay ** episodeIndex))
        action = np.clip(noisyAction, self.actionLow, self.actionHigh)
        return action


def addToMemory(buffer, state, action, nextState):
    experience = (state, action, nextState)
    buffer.append(experience)
    return buffer

def initializeMemory(bufferSize):
    return deque(maxlen=bufferSize)


class ActAngleWithNoise:
    def __init__(self, actByModel, addActionNoise):
        self.actByModel = actByModel
        self.addActionNoise = addActionNoise

    def __call__(self, actorModel, stateBatch, timeStep):
        actionPerfect = self.actByModel(actorModel, stateBatch)
        actionAngle = self.addActionNoise(actionPerfect, timeStep)
        return actionAngle

class ActByAngle:
    def __init__(self,velocity):
        self.velocity = velocity

    def __call__(self, actionAngle):
        positiveX = np.array([1, 0])
        positiveY = np.array([0, 1])
        action = np.cos(actionAngle) * positiveX + np.sin(actionAngle) * positiveY
        return action


class RunDDPGTimeStep:
    def __init__(self, actAngleWithNoise, actByAngle, transitionFunction, addToMemory, trainCritic, trainActor, minibatchSize, numStateSpace):
        self.actAngleWithNoise = actAngleWithNoise
        self.actByAngle = actByAngle
        self.transitionFunction = transitionFunction
        self.addToMemory = addToMemory
        self.trainCritic = trainCritic
        self.trainActor = trainActor
        self.minibatchSize = minibatchSize
        self.numStateSpace = numStateSpace

    def __call__(self, timeStep, state, actorModel, criticModel, replayBuffer, episodeID):
        stateBatch = np.asarray(state).reshape(1, -1)
        actionOutput = self.actAngleWithNoise(actorModel, stateBatch, timeStep)
        action = self.actByAngle(actionOutput)[0]
        nextState = self.transitionFunction(state, action)
        replayBuffer = self.addToMemory(replayBuffer, state, actionOutput, nextState)
        if len(replayBuffer) >= self.minibatchSize:
            miniBatch = random.sample(replayBuffer, self.minibatchSize)
            criticLoss, criticModel = self.trainCritic(actorModel, criticModel, miniBatch, episodeID)
            actorTrainOpt, actorModel = self.trainActor(actorModel, criticModel, miniBatch)

        return nextState, actorModel, criticModel, replayBuffer


class RunEpisode:
    def __init__(self, reset, isTerminal, runTimeStep, maxTimeStep):
        self.reset = reset
        self.isTerminal = isTerminal
        self.runTimeStep = runTimeStep
        self.maxTimeStep = maxTimeStep

    def __call__(self, actorModel, criticModel, replayBuffer):
        state = self.reset()
        for timeStep in range(self.maxTimeStep):
            state, actorModel, criticModel, replayBuffer = self.runTimeStep(timeStep, state, actorModel, criticModel, replayBuffer, timeStep)
            if self.isTerminal(state):
                break
        return actorModel, criticModel, replayBuffer


class DDPG:
    def __init__(self, initializeMemory, runEpisode, bufferSize, maxEpisode):
        self.bufferSize = bufferSize
        self.initializeMemory = initializeMemory
        self.runEpisode = runEpisode
        self.maxEpisode = maxEpisode

    def __call__(self, actorModel, criticModel):
        replayBuffer = self.initializeMemory(self.bufferSize)
        for episode in range(self.maxEpisode):
            if episode % 50 == 0:
                print(episode)
            actorModel, criticModel, replayBuffer = self.runEpisode(actorModel, criticModel, replayBuffer)

        return actorModel, criticModel
