import tensorflow as tf
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


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
    actionTarget_ = criticGraph.get_collection_ref("actionTarget_")[0]
    targetValues_ = criticGraph.get_collection_ref("targetValues_")[0]
    targetValues = criticModel.run(targetValues_, feed_dict={states_: stateBatch, actionTarget_: actionsBatch})
    return targetValues


def evaluateCriticTrain(criticModel, stateBatch, actionsBatch):
    criticGraph = criticModel.graph
    states_ = criticGraph.get_collection_ref("states_")[0]
    action_ = criticGraph.get_collection_ref("action_")[0]
    trainValues_ = criticGraph.get_collection_ref("trainValues_")[0]
    trainValues = criticModel.run(trainValues_, feed_dict={states_: stateBatch, action_: actionsBatch})
    return trainValues


def getActionGradients(criticModel, stateBatch, actionsBatch):
    criticGraph = criticModel.graph
    states_ = criticGraph.get_collection_ref("states_")[0]
    action_ = criticGraph.get_collection_ref("action_")[0]
    actionGradients_ = criticGraph.get_collection_ref("actionGradients_")[0]
    actionGradients = criticModel.run(actionGradients_, feed_dict={states_: stateBatch,
                                                                   action_: actionsBatch})
    return actionGradients


class BuildActorModel:
    def __init__(self, numStateSpace, actionDim, actionRange):
        self.numStateSpace = numStateSpace
        self.actionDim = actionDim
        self.actionRange = actionRange

    def __call__(self, layersWidths, summaryPath="./tbdata"):
        print("Generating Actor NN Model with layers: {}".format(layersWidths))
        graph = tf.Graph()
        with graph.as_default():
            with tf.name_scope("inputs"):
                states_ = tf.placeholder(tf.float32, [None, self.numStateSpace])
                actionGradients_ = tf.placeholder(tf.float32, [None, self.actionDim])
                tf.add_to_collection("states_", states_)
                tf.add_to_collection("actionGradients_", actionGradients_)

            with tf.name_scope("trainingParams"):
                learningRate_ = tf.constant(0, dtype=tf.float32)
                tau_ = tf.constant(0, dtype=tf.float32)
                tf.add_to_collection("learningRate_", learningRate_)
                tf.add_to_collection("tau_", tau_)

            initWeight = tf.random_uniform_initializer(0, 0.3)
            initBias = tf.constant_initializer(0.1)
            with tf.variable_scope("trainHidden"):
                activation_ = states_
                for i in range(len(layersWidths)):
                    fcLayer = tf.layers.Dense(units=layersWidths[i], activation=tf.nn.relu,
                                              kernel_initializer=initWeight,
                                              bias_initializer=initBias, name="fc{}".format(i + 1), trainable=True)
                    activation_ = fcLayer(activation_)

                    tf.add_to_collections(["weights", f"weight/{fcLayer.kernel.name}"], fcLayer.kernel)
                    tf.add_to_collections(["biases", f"bias/{fcLayer.bias.name}"], fcLayer.bias)
                    tf.add_to_collections(["activations", f"activation/{activation_.name}"], activation_)
                trainActivation_ = tf.identity(activation_)
                tf.add_to_collection("trainActivation_", trainActivation_)

                outputFCLayer = tf.layers.Dense(units=self.actionDim, activation=tf.nn.tanh,
                                                kernel_initializer=initWeight,
                                                bias_initializer=initBias, name="fc{}".format(len(layersWidths) + 1),
                                                trainable=True)
                trainActivationOutput_ = outputFCLayer(trainActivation_)

                tf.add_to_collections(["weights", f"weight/{outputFCLayer.kernel.name}"], outputFCLayer.kernel)
                tf.add_to_collections(["biases", f"bias/{outputFCLayer.bias.name}"], outputFCLayer.bias)
                tf.add_to_collections(["activations", f"activation/{trainActivationOutput_.name}"],
                                      trainActivationOutput_)

            with tf.variable_scope("targetHidden"):
                activation_ = states_
                for i in range(len(layersWidths)):
                    fcLayer = tf.layers.Dense(units=layersWidths[i], activation=tf.nn.relu,
                                              kernel_initializer=initWeight,
                                              bias_initializer=initBias, name="fc{}".format(i + 1), trainable=False)
                    activation_ = fcLayer(activation_)

                    tf.add_to_collections(["weights", f"weight/{fcLayer.kernel.name}"], fcLayer.kernel)
                    tf.add_to_collections(["biases", f"bias/{fcLayer.bias.name}"], fcLayer.bias)
                    tf.add_to_collections(["activations", f"activation/{activation_.name}"], activation_)
                targetActivation_ = tf.identity(activation_, name="output")
                outputFCLayer = tf.layers.Dense(units=self.actionDim, activation=tf.nn.tanh,
                                                kernel_initializer=initWeight,
                                                bias_initializer=initBias, name="fc{}".format(len(layersWidths) + 1),
                                                trainable=False)
                targetActivationOutput_ = outputFCLayer(targetActivation_)

                tf.add_to_collections(["weights", f"weight/{outputFCLayer.kernel.name}"], outputFCLayer.kernel)
                tf.add_to_collections(["biases", f"bias/{outputFCLayer.bias.name}"], outputFCLayer.bias)
                tf.add_to_collections(["activations", f"activation/{targetActivationOutput_.name}"],
                                      targetActivationOutput_)

            with tf.name_scope("updateParameters"):
                trainParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='trainHidden')
                targetParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='targetHidden')
                updateParam_ = [targetParams_[i].assign((1 - tau_) * targetParams_[i] + tau_ * trainParams_[i]) for i in
                                range(len(targetParams_))]

                tf.add_to_collection("trainParams_", trainParams_)
                tf.add_to_collection("targetParams_", targetParams_)
                tf.add_to_collection("updateParam_", updateParam_)

                hardReplaceTargetParam_ = [tf.assign(trainParam, targetParam) for trainParam, targetParam in
                                           zip(trainParams_, targetParams_)]
                tf.add_to_collection("hardReplaceTargetParam_", hardReplaceTargetParam_)

            with tf.name_scope("output"):
                trainAction_ = tf.multiply(trainActivationOutput_, self.actionRange, name='trainAction_')
                targetAction_ = tf.multiply(targetActivationOutput_, self.actionRange, name='targetAction_')
                policyGradient_ = tf.gradients(ys=trainAction_, xs=trainParams_, grad_ys=actionGradients_)
                # ys = policy, xs = policy's parameters; a_grads = the gradients of the policy to get more Q
                # tf.gradients will calculate dys/dxs with a initial gradients for ys, so this is dq/da * da/dparams

                tf.add_to_collection("trainAction_", trainAction_)
                tf.add_to_collection("targetAction_", targetAction_)
                tf.add_to_collection("policyGradient_", policyGradient_)

            with tf.name_scope("train"):
                optimizer = tf.train.AdamOptimizer(-learningRate_, name='adamOptimizer')
                trainOpt_ = optimizer.apply_gradients(zip(policyGradient_, trainParams_))
                tf.add_to_collection("trainOpt_", trainOpt_)

            fullSummary = tf.summary.merge_all()
            tf.add_to_collection("summaryOps", fullSummary)

            actorSaver = tf.train.Saver(max_to_keep=None)
            tf.add_to_collection("saver", actorSaver)

            model = tf.Session(graph=graph)
            model.run(tf.global_variables_initializer())

            actorWriter = tf.summary.FileWriter('tensorBoard/actorOnlineDDPG', graph=graph)
            tf.add_to_collection("actorWriter", actorWriter)

        return actorWriter, model


class BuildCriticModel:
    def __init__(self, numStateSpace, actionDim):
        self.numStateSpace = numStateSpace
        self.actionDim = actionDim

    def __call__(self, layersWidths, summaryPath="./tbdata"):
        print("Generating Actor NN Model with layers: {}".format(layersWidths))
        graph = tf.Graph()
        with graph.as_default():

            with tf.name_scope("inputs"):
                states_ = tf.placeholder(tf.float32, [None, self.numStateSpace])
                action_ = tf.stop_gradient(tf.placeholder(tf.float32, [None, self.actionDim]))
                actionTarget_ = tf.placeholder(tf.float32, [None, self.actionDim])
                reward_ = tf.placeholder(tf.float32, [None, 1])
                valueTarget_ = tf.placeholder(tf.float32, [None, 1])

                tf.add_to_collection("states_", states_)
                tf.add_to_collection("action_", action_)
                tf.add_to_collection("actionTarget_", actionTarget_)
                tf.add_to_collection("reward_", reward_)
                tf.add_to_collection("valueTarget_", valueTarget_)

            with tf.name_scope("trainingParams"):
                learningRate_ = tf.constant(0, dtype=tf.float32)
                tau_ = tf.constant(0, dtype=tf.float32)
                gamma_ = tf.constant(0, dtype=tf.float32)

                tf.add_to_collection("learningRate_", learningRate_)
                tf.add_to_collection("tau_", tau_)
                tf.add_to_collection("gamma_", gamma_)

            initWeight = tf.random_uniform_initializer(0, 0.1)
            initBias = tf.constant_initializer(0.1)
            with tf.variable_scope("trainHidden"):
                activation_ = states_
                for i in range(len(layersWidths) - 1):
                    fcLayer = tf.layers.Dense(units=layersWidths[i], activation=tf.nn.relu,
                                              kernel_initializer=initWeight,
                                              bias_initializer=initBias, name="fc{}".format(i + 1), trainable=True)
                    activation_ = fcLayer(activation_)

                    tf.add_to_collections(["weights", f"weight/{fcLayer.kernel.name}"], fcLayer.kernel)
                    tf.add_to_collections(["biases", f"bias/{fcLayer.bias.name}"], fcLayer.bias)
                    tf.add_to_collections(["activations", f"activation/{activation_.name}"], activation_)
                trainActivation_ = tf.identity(activation_, name="output")
                secondLastFCUnit = layersWidths[-2] if len(layersWidths) >= 2 else self.numStateSpace

                lastFCUnit = layersWidths[-1]
                trainStateFCToLastFCWeights_ = tf.get_variable(name='trainStateFCToLastFCWeights_',
                                                               shape=[secondLastFCUnit, lastFCUnit],
                                                               initializer=initWeight)
                trainActionFCToLastFCWeights_ = tf.get_variable(name='trainActionFCToLastFCWeights_',
                                                                shape=[self.actionDim, lastFCUnit],
                                                                initializer=initWeight)
                trainActionLastFCBias_ = tf.get_variable(name='trainActionLastFCBias_', shape=[lastFCUnit],
                                                         initializer=initBias)

                trainLastFCZ_ = tf.matmul(trainActivation_, trainStateFCToLastFCWeights_) + \
                                tf.matmul(action_, trainActionFCToLastFCWeights_) + trainActionLastFCBias_
                tf.add_to_collections("trainLastFCZ_", trainLastFCZ_)

                trainLastFCActivation_ = tf.nn.relu(tf.matmul(trainActivation_, trainStateFCToLastFCWeights_) +
                                                    tf.matmul(action_, trainActionFCToLastFCWeights_) +
                                                    trainActionLastFCBias_)

                tf.add_to_collections(["weights", "weight/trainStateFCToLastFCWeights_"], trainStateFCToLastFCWeights_)
                tf.add_to_collections(["weights", "weight/trainActionFCToLastFCWeights_"],
                                      trainActionFCToLastFCWeights_)
                tf.add_to_collections(["biases", "bias/trainActionLastFCBias_"], trainActionLastFCBias_)
                tf.add_to_collections(["activations", "activation/trainLastFCActivation_"], trainLastFCActivation_)

                trainOutputLayer = tf.layers.Dense(units=1, kernel_initializer=initWeight, bias_initializer=initBias,
                                                   activation=None, trainable=True)
                trainValues_ = trainOutputLayer(trainLastFCActivation_)
                tf.add_to_collections(["weights", f"weight/{trainOutputLayer.kernel.name}"], trainOutputLayer.kernel)
                tf.add_to_collections(["biases", f"bias/{trainOutputLayer.bias.name}"], trainOutputLayer.bias)
                tf.add_to_collections(["activations", f"activation/{trainValues_.name}"], trainValues_)
                tf.add_to_collection("trainValues_", trainValues_)

            with tf.variable_scope("targetHidden"):
                activation_ = states_
                for i in range(len(layersWidths) - 1):
                    fcLayer = tf.layers.Dense(units=layersWidths[i], activation=tf.nn.relu,
                                              kernel_initializer=initWeight,
                                              bias_initializer=initBias, name="fc{}".format(i + 1), trainable=False)
                    activation_ = fcLayer(activation_)

                    tf.add_to_collections(["weights", f"weight/{fcLayer.kernel.name}"], fcLayer.kernel)
                    tf.add_to_collections(["biases", f"bias/{fcLayer.bias.name}"], fcLayer.bias)
                    tf.add_to_collections(["activations", f"activation/{activation_.name}"], activation_)
                targetActivation_ = tf.identity(activation_, name="output")

                secondLastFCUnit = layersWidths[-2] if len(layersWidths) >= 2 else self.numStateSpace
                lastFCUnit = layersWidths[-1]
                targetStateFCToLastFCWeights_ = tf.get_variable(name='targetStateFCToLastFCWeights_',
                                                                shape=[secondLastFCUnit, lastFCUnit],
                                                                initializer=initWeight)
                targetActionFCToLastFCWeights_ = tf.get_variable(name='targetActionFCToLastFCWeights_',
                                                                 shape=[self.actionDim, lastFCUnit],
                                                                 initializer=initWeight)
                targetActionLastFCBias_ = tf.get_variable(name='targetActionLastFCBias_', shape=[lastFCUnit],
                                                          initializer=initBias)
                targetLastFCActivation_ = tf.nn.relu(tf.matmul(targetActivation_, targetStateFCToLastFCWeights_) +
                                                     tf.matmul(actionTarget_, targetActionFCToLastFCWeights_) +
                                                     targetActionLastFCBias_)

                tf.add_to_collections(["weights", "weight/targetStateFCToLastFCWeights_"],
                                      targetStateFCToLastFCWeights_)
                tf.add_to_collections(["weights", "weight/targetActionFCToLastFCWeights_"],
                                      targetActionFCToLastFCWeights_)
                tf.add_to_collections(["biases", "bias/targetActionLastFCBias_"], targetActionLastFCBias_)
                tf.add_to_collections(["activations", "activation/targetLastFCActivation_"], targetLastFCActivation_)

                targetOutputLayer = tf.layers.Dense(units=1, kernel_initializer=initWeight, bias_initializer=initBias,
                                                    activation=None, trainable=False)
                targetValues_ = targetOutputLayer(targetLastFCActivation_)
                tf.add_to_collections(["weights", f"weight/{targetOutputLayer.kernel.name}"], targetOutputLayer.kernel)
                tf.add_to_collections(["biases", f"bias/{targetOutputLayer.bias.name}"], targetOutputLayer.bias)
                tf.add_to_collections(["activations", f"activation/{targetValues_.name}"], targetValues_)
                tf.add_to_collection("targetValues_", targetValues_)

            with tf.name_scope("parameters"):
                trainParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='trainHidden')
                targetParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='targetHidden')
                updateParam_ = [targetParams_[i].assign((1 - tau_) * targetParams_[i] + tau_ * trainParams_[i]) for i in
                                range(len(targetParams_))]

                tf.add_to_collection("trainParams_", trainParams_)
                tf.add_to_collection("targetParams_", targetParams_)
                tf.add_to_collection("updateParam_", updateParam_)

                hardReplaceTargetParam_ = [tf.assign(trainParam, targetParam) for trainParam, targetParam in
                                           zip(trainParams_, targetParams_)]
                tf.add_to_collection("hardReplaceTargetParam_", hardReplaceTargetParam_)

            with tf.name_scope("actionGradients"):
                actionGradients_ = tf.gradients(trainValues_, action_)[0]
                tf.add_to_collection("actionGradients_", actionGradients_)

            with tf.name_scope("output"):
                trainQ_ = tf.multiply(trainValues_, 1, name='trainQ_')
                targetQ_ = tf.multiply(targetValues_, 1, name='targetQ_')
                tf.add_to_collection("trainQ_", trainQ_)
                tf.add_to_collection("targetQ_", targetQ_)

            with tf.name_scope("evaluate"):
                yi_ = reward_ + gamma_ * valueTarget_
                valueLoss_ = tf.losses.mean_squared_error(labels=yi_, predictions=trainQ_)

                tf.add_to_collection("yi_", yi_)
                tf.add_to_collection("valueLoss_", valueLoss_)

            with tf.name_scope("train"):
                trainOpt_ = tf.train.AdamOptimizer(learningRate_, name='adamOptimizer').minimize(valueLoss_,
                                                                                                 var_list=trainParams_)
                tf.add_to_collection("trainOpt_", trainOpt_)

            fullSummary = tf.summary.merge_all()
            tf.add_to_collection("summaryOps", fullSummary)

            criticSaver = tf.train.Saver(max_to_keep=None)
            tf.add_to_collection("saver", criticSaver)

            model = tf.Session(graph=graph)
            model.run(tf.global_variables_initializer())

            criticWriter = tf.summary.FileWriter('tensorBoard/criticOnlineDDPG', graph=graph)
            tf.add_to_collection("criticWriter", criticWriter)

        return criticWriter, model


class TrainCriticBySASRQ:
    def __init__(self, criticLearningRate, gamma, criticWriter):
        self.criticLearningRate = criticLearningRate
        self.gamma = gamma
        self.criticWriter = criticWriter

    def __call__(self, criticModel, stateBatch, actionBatch, rewardBatch, targetQValue):
        criticGraph = criticModel.graph
        states_ = criticGraph.get_collection_ref("states_")[0]
        action_ = criticGraph.get_collection_ref("action_")[0]
        reward_ = criticGraph.get_collection_ref("reward_")[0]
        valueTarget_ = criticGraph.get_collection_ref("valueTarget_")[0]
        learningRate_ = criticGraph.get_collection_ref("learningRate_")[0]
        gamma_ = criticGraph.get_collection_ref("gamma_")[0]

        valueLoss_ = criticGraph.get_collection_ref("valueLoss_")[0]
        trainOpt_ = criticGraph.get_collection_ref("trainOpt_")[0]
        criticLoss, trainOpt = criticModel.run([valueLoss_, trainOpt_],
                                               feed_dict={states_: stateBatch, action_: actionBatch,
                                                          reward_: rewardBatch, valueTarget_: targetQValue,
                                                          learningRate_: self.criticLearningRate, gamma_: self.gamma})

        summary = tf.Summary()
        summary.value.add(tag='reward', simple_value=float(np.mean(rewardBatch)))
        summary.value.add(tag='loss', simple_value=float(criticLoss))
        self.criticWriter.flush()
        return criticLoss, criticModel


class TrainCritic:
    def __init__(self, actByPolicyTarget, evaluateCriticTarget, trainCriticBySASRQ):
        self.actByPolicyTarget = actByPolicyTarget
        self.evaluateCriticTarget = evaluateCriticTarget
        self.trainCriticBySASRQ = trainCriticBySASRQ

    def __call__(self, actorModel, criticModel, miniBatch):
        states, actions, rewards, nextStates = list(zip(*miniBatch))
        stateBatch = np.asarray(states).reshape(len(miniBatch), -1)
        actionBatch = np.asarray(actions).reshape(len(miniBatch), -1)
        nextStateBatch = np.asarray(nextStates).reshape(len(miniBatch), -1)
        rewardBatch = np.asarray(rewards).reshape(len(miniBatch), -1)

        targetNextActionBatch = self.actByPolicyTarget(actorModel, nextStateBatch)
        targetQValue = self.evaluateCriticTarget(criticModel, nextStateBatch, targetNextActionBatch)

        criticLoss, criticModel = self.trainCriticBySASRQ(criticModel, stateBatch, actionBatch, rewardBatch,
                                                          targetQValue)
        return criticLoss, criticModel


class TrainActorFromGradients:
    def __init__(self, actorLearningRate, actorWriter):
        self.actorLearningRate = actorLearningRate
        self.actorWriter = actorWriter

    def __call__(self, actorModel, stateBatch, actionGradients):
        actorGraph = actorModel.graph
        states_ = actorGraph.get_collection_ref("states_")[0]
        actionGradients_ = actorGraph.get_collection_ref("actionGradients_")[0]
        learningRate_ = actorGraph.get_collection_ref("learningRate_")[0]

        trainOpt_ = actorGraph.get_collection_ref("trainOpt_")[0]
        trainActivation_ = actorGraph.get_collection_ref("trainActivation_")[0]
        trainActivation, trainOpt = actorModel.run([trainActivation_, trainOpt_],
                                                   feed_dict={states_: stateBatch, actionGradients_: actionGradients,
                                                              learningRate_: self.actorLearningRate})
        self.actorWriter.flush()
        return actorModel


class TrainActorOneStep:
    def __init__(self, actByPolicyTrain, trainActorFromGradients, getActionGradients):
        self.actByPolicyTrain = actByPolicyTrain
        self.trainActorFromGradients = trainActorFromGradients
        self.getActionGradients = getActionGradients

    def __call__(self, actorModel, criticModel, stateBatch):
        actionsBatch = self.actByPolicyTrain(actorModel, stateBatch)
        actionGradients = self.getActionGradients(criticModel, stateBatch, actionsBatch)
        actorModel = self.trainActorFromGradients(actorModel, stateBatch, actionGradients)
        return actorModel


class TrainActor:
    def __init__(self, trainActorOneStep):
        self.trainActorOneStep = trainActorOneStep

    def __call__(self, actorModel, criticModel, miniBatch):
        states, actions, rewards, nextStates = list(zip(*miniBatch))
        stateBatch = np.asarray(states).reshape(len(miniBatch), -1)
        actorModel = self.trainActorOneStep(actorModel, criticModel, stateBatch)

        return actorModel


class TrainDDPGModels:
    def __init__(self, updateParameters, trainActor, trainCritic, actorModel, criticModel):
        self.updateParameters = updateParameters
        self.trainActor = trainActor
        self.trainCritic = trainCritic
        self.actorModel = actorModel
        self.criticModel = criticModel

    def __call__(self, miniBatch):
        criticLoss, self.criticModel = self.trainCritic(self.actorModel, self.criticModel, miniBatch)
        self.actorModel = self.trainActor(self.actorModel, self.criticModel, miniBatch)

        self.actorModel = self.updateParameters(self.actorModel)
        self.criticModel = self.updateParameters(self.criticModel)

    def getTrainedModels(self):
        return [self.actorModel, self.criticModel]