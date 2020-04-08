import tensorflow as tf
import numpy as np
import random
tf.enable_eager_execution()

class GenerateModel:
    def __init__(self, numStateSpace, numActionSpace, regularizationFactor=0, valueRelativeErrBound=0.01, seed=128):
        self.numStateSpace = numStateSpace
        self.numActionSpace = numActionSpace
        self.regularizationFactor = regularizationFactor
        self.valueRelativeErrBound = valueRelativeErrBound
        self.seed = seed

    def __call__(self, sharedWidths, actionLayerWidths, valueLayerWidths, summaryPath="./tbdata"):
        print("Generating NN with shared layers: {}, action layers: {}, value layers: {}"
              .format(sharedWidths, actionLayerWidths, valueLayerWidths))
        graph = tf.Graph()
        with graph.as_default():
            if self.seed is not None:
                tf.set_random_seed(self.seed)

            with tf.name_scope("inputs"):
                states_ = tf.placeholder(tf.float32, [None, self.numStateSpace], name="states")
                tf.add_to_collection("inputs", states_)

            with tf.name_scope("groundTruths"):
                groundTruthAction_ = tf.placeholder(tf.int32, [None, self.numActionSpace], name="action")
                groundTruthValue_ = tf.placeholder(tf.float32, [None, 1], name="value")
                tf.add_to_collection("groundTruths", groundTruthAction_)
                tf.add_to_collection("groundTruths", groundTruthValue_)

            with tf.name_scope("trainingParams"):
                learningRate_ = tf.constant(0, dtype=tf.float32)
                actionLossCoef_ = tf.constant(1, dtype=tf.float32)
                valueLossCoef_ = tf.constant(1, dtype=tf.float32)
                tf.add_to_collection("learningRate", learningRate_)
                tf.add_to_collection("lossCoefs", actionLossCoef_)
                tf.add_to_collection("lossCoefs", valueLossCoef_)

            initWeight = tf.random_uniform_initializer(-0.03, 0.03)
            initBias = tf.constant_initializer(0.001)

            with tf.variable_scope("shared"):
                activation_ = states_
                for i in range(len(sharedWidths)):
                    fcLayer = tf.layers.Dense(units=sharedWidths[i], activation=tf.nn.relu, kernel_initializer=initWeight,
                                              bias_initializer=initBias, name="fc{}".format(i+1))
                    activation_ = fcLayer(activation_)
                    tf.add_to_collections(["weights", f"weight/{fcLayer.kernel.name}"], fcLayer.kernel)
                    tf.add_to_collections(["biases", f"bias/{fcLayer.bias.name}"], fcLayer.bias)
                    tf.add_to_collections(["activations", f"activation/{activation_.name}"], activation_)
                sharedOutput_ = tf.identity(activation_, name="output")

            with tf.variable_scope("action"):
                activation_ = sharedOutput_
                for i in range(len(actionLayerWidths)):
                    fcLayer = tf.layers.Dense(units=actionLayerWidths[i], activation=tf.nn.relu, kernel_initializer=initWeight,
                                              bias_initializer=initBias, name="fc{}".format(i+1))
                    activation_ = fcLayer(activation_)
                    tf.add_to_collections(["weights", f"weight/{fcLayer.kernel.name}"], fcLayer.kernel)
                    tf.add_to_collections(["biases", f"bias/{fcLayer.bias.name}"], fcLayer.bias)
                    tf.add_to_collections(["activations", f"activation/{activation_.name}"], activation_)
                outputFCLayer = tf.layers.Dense(units=self.numActionSpace, activation=None, kernel_initializer=initWeight,
                                                bias_initializer=initBias, name="fc{}".format(len(actionLayerWidths) + 1))
                actionOutputLayerActivation_ = outputFCLayer(activation_)
                tf.add_to_collections(["weights", f"weight/{outputFCLayer.kernel.name}"], outputFCLayer.kernel)
                tf.add_to_collections(["biases", f"bias/{outputFCLayer.bias.name}"], outputFCLayer.bias)
                tf.add_to_collections(["activations", f"activation/{actionOutputLayerActivation_.name}"],
                                      actionOutputLayerActivation_)

            with tf.name_scope("actionOutputs"):
                actionDistributions_ = tf.nn.softmax(actionOutputLayerActivation_, name="distributions")
                actionIndices_ = tf.argmax(actionDistributions_, axis=1, name="indices")
                tf.add_to_collection("actionDistributions", actionDistributions_)
                tf.add_to_collection("actionIndices", actionIndices_)

            with tf.variable_scope("value"):
                activation_ = sharedOutput_
                for i in range(len(valueLayerWidths)):
                    fcLayer = tf.layers.Dense(units=valueLayerWidths[i], activation=tf.nn.relu, kernel_initializer=initWeight,
                                              bias_initializer=initBias, name="fc{}".format(i+1))
                    activation_ = fcLayer(activation_)
                    tf.add_to_collections(["weights", f"weight/{fcLayer.kernel.name}"], fcLayer.kernel)
                    tf.add_to_collections(["biases", f"bias/{fcLayer.bias.name}"], fcLayer.bias)
                    tf.add_to_collections(["activations", f"activation/{activation_.name}"], activation_)

                outputFCLayer = tf.layers.Dense(units=1, activation=None, kernel_initializer=initWeight,
                                                bias_initializer=initBias, name="fc{}".format(len(valueLayerWidths) + 1))
                valueOutputLayerActivation_ = outputFCLayer(activation_)
                tf.add_to_collections(["weights", f"weight/{outputFCLayer.kernel.name}"], outputFCLayer.kernel)
                tf.add_to_collections(["biases", f"bias/{outputFCLayer.bias.name}"], outputFCLayer.bias)
                tf.add_to_collections(["activations", f"activation/{valueOutputLayerActivation_.name}"],
                                      valueOutputLayerActivation_)

            with tf.name_scope("valueOutputs"):
                values_ = tf.identity(valueOutputLayerActivation_, name="values")
                tf.add_to_collection("values", values_)

            with tf.name_scope("evaluate"):
                with tf.name_scope("action"):
                    crossEntropy_ = tf.nn.softmax_cross_entropy_with_logits_v2(logits=actionOutputLayerActivation_,
                                                                               labels=groundTruthAction_)
                    actionLoss_ = tf.reduce_mean(crossEntropy_, name='loss')
                    tf.add_to_collection("actionLoss", actionLoss_)
                    actionLossSummary = tf.summary.scalar("actionLoss", actionLoss_)

                    groundTruthActionIndices_ = tf.argmax(groundTruthAction_, axis=1)
                    actionAccuracy_ = tf.reduce_mean(tf.cast(tf.equal(actionIndices_, groundTruthActionIndices_), tf.float32), name="accuracy")
                    tf.add_to_collection("actionAccuracy", actionAccuracy_)
                    actionAccuracySummary = tf.summary.scalar("actionAccuracy", actionAccuracy_)

                with tf.name_scope("value"):
                    valueLoss_ = tf.losses.mean_squared_error(groundTruthValue_, values_)
                    tf.add_to_collection("valueLoss", valueLoss_)
                    valueLossSummary = tf.summary.scalar("valueLoss", valueLoss_)

                    relativeErrorBound_ = tf.constant(self.valueRelativeErrBound)
                    relativeValueError_ = tf.cast((tf.abs((values_ - groundTruthValue_) / groundTruthValue_)), relativeErrorBound_.dtype)
                    valueAccuracy_ = tf.reduce_mean(tf.cast(tf.less(relativeValueError_, relativeErrorBound_), tf.float64), name="accuracy")
                    tf.add_to_collection("valueAccuracy", valueAccuracy_)
                    valueAccuracySummary = tf.summary.scalar("valueAccuracy", valueAccuracy_)

                with tf.name_scope("regularization"):
                    l2RegularizationLoss_ = tf.multiply(tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]),
                                                        self.regularizationFactor, name="l2RegLoss")
                    tf.summary.scalar("l2RegLoss", l2RegularizationLoss_)

                loss_ = tf.add_n([actionLossCoef_*actionLoss_, valueLossCoef_*valueLoss_, l2RegularizationLoss_], name="loss")
                tf.add_to_collection("loss", loss_)
                lossSummary = tf.summary.scalar("loss", loss_)

            with tf.name_scope("train"):
                optimizer = tf.train.AdamOptimizer(learningRate_, name='adamOptimizer')
                gradVarPairs_ = optimizer.compute_gradients(loss_)
                trainOp = optimizer.apply_gradients(gradVarPairs_)
                tf.add_to_collection("trainOp", trainOp)

                with tf.name_scope("inspectGrad"):
                    for grad_, var_ in gradVarPairs_:
                        keyPrefix = "weightGradient" if "kernel" in var_.name else "biasGradient"
                        tf.add_to_collection(f"{keyPrefix}/{var_.name}", grad_)
                    gradients_ = [tf.reshape(grad_, [1, -1]) for (grad_, _) in gradVarPairs_]
                    allGradTensor_ = tf.concat(gradients_, 1)
                    allGradNorm_ = tf.norm(allGradTensor_)
                    tf.add_to_collection("allGradNorm", allGradNorm_)
                    tf.summary.histogram("allGradients", allGradTensor_)
                    tf.summary.scalar("allGradNorm", allGradNorm_)
            fullSummary = tf.summary.merge_all()
            evalSummary = tf.summary.merge([lossSummary, actionLossSummary, valueLossSummary,
                                            actionAccuracySummary, valueAccuracySummary])
            tf.add_to_collection("summaryOps", fullSummary)
            tf.add_to_collection("summaryOps", evalSummary)

            if summaryPath is not None:
                trainWriter = tf.summary.FileWriter(summaryPath + "/train", graph=tf.get_default_graph())
                testWriter = tf.summary.FileWriter(summaryPath + "/test", graph=tf.get_default_graph())
                tf.add_to_collection("writers", trainWriter)
                tf.add_to_collection("writers", testWriter)

            saver = tf.train.Saver(max_to_keep=None)
            tf.add_to_collection("saver", saver)

            model = tf.Session(graph=graph)
            model.run(tf.global_variables_initializer())

        return model


class Train:
    def __init__(self, maxStepNum, batchSize, sampleData, learningRateModifier, terimnalController, coefficientController, trainReporter):
        self.maxStepNum = maxStepNum
        self.batchSize = batchSize
        self.sampleData = sampleData
        self.learningRateModifier = learningRateModifier
        self.terminalController = terimnalController
        self.coefficientController = coefficientController
        self.reporter = trainReporter

    def __call__(self, model, trainingData):
        print("ENTER TRAIN")
        graph = model.graph
        state_ = graph.get_collection_ref("inputs")[0]
        groundTruthAction_, groundTruthValue_ = graph.get_collection_ref("groundTruths")
        learningRate_ = graph.get_collection_ref("learningRate")[0]
        actionLossCoef_, valueLossCoef_ = graph.get_collection_ref("lossCoefs")
        loss_ = graph.get_collection_ref("loss")[0]
        actionLoss_ = graph.get_collection_ref("actionLoss")[0]
        valueLoss_ = graph.get_collection_ref("valueLoss")[0]
        actionAccuracy_ = graph.get_collection_ref("actionAccuracy")[0]
        valueAccuracy_ = graph.get_collection_ref("valueAccuracy")[0]
        evaluations_ = {"loss": loss_, "actionLoss": actionLoss_, "actionAcc": actionAccuracy_, "valueLoss": valueLoss_,
                        "valueAcc": valueAccuracy_}
        trainOp = graph.get_collection_ref("trainOp")[0]
        fullSummaryOp = graph.get_collection_ref('summaryOps')[0]
        trainWriter = graph.get_collection_ref('writers')[0]
        fetches = [evaluations_, trainOp, fullSummaryOp]

        evalDict = None
        trainingDataList = list(zip(*trainingData))

        for stepNum in range(self.maxStepNum):
            if self.batchSize == 0:
                stateBatch, actionBatch, valueBatch = trainingData
            else:
                stateBatch, actionBatch, valueBatch = self.sampleData(trainingDataList, self.batchSize)
            learningRate = self.learningRateModifier(stepNum)
            actionLossCoef, valueLossCoef = self.coefficientController(evalDict)
            feedDict = {state_: stateBatch, groundTruthAction_: actionBatch, groundTruthValue_: valueBatch,
                        learningRate_: learningRate, actionLossCoef_: actionLossCoef, valueLossCoef_: valueLossCoef}
            evalDict, _, summary = model.run(fetches, feed_dict=feedDict)

            self.reporter(evalDict, stepNum, trainWriter, summary)

            if self.terminalController(evalDict, stepNum):
                break

        return model


class TrainWithCustomizedFetches:
    def __init__(self, maxStepNum, batchSize, sampleData, learningRateModifier, terimnalController,
                 coefficientController, trainReporter, prepareFetches):
        self.maxStepNum = maxStepNum
        self.batchSize = batchSize
        self.sampleData = sampleData
        self.learningRateModifier = learningRateModifier
        self.terminalController = terimnalController
        self.coefficientController = coefficientController
        self.reporter = trainReporter
        self.prepareFetches = prepareFetches

    def __call__(self, model, trainingData):
        graph = model.graph
        state_ = graph.get_collection_ref("inputs")[0]
        groundTruthAction_, groundTruthValue_ = graph.get_collection_ref("groundTruths")
        learningRate_ = graph.get_collection_ref("learningRate")[0]
        actionLossCoef_, valueLossCoef_ = graph.get_collection_ref("lossCoefs")
        trainOp = graph.get_collection_ref("trainOp")[0]
        fetches = [self.prepareFetches(graph), trainOp]

        evalDict = None
        trainingDataList = list(zip(*trainingData))

        for stepNum in range(1, self.maxStepNum + 1):
            if self.batchSize == 0:
                stateBatch, actionBatch, valueBatch = trainingData
            else:
                stateBatch, actionBatch, valueBatch = self.sampleData(trainingDataList, self.batchSize)
            learningRate = self.learningRateModifier(stepNum)
            actionLossCoef, valueLossCoef = self.coefficientController(evalDict)
            feedDict = {state_: stateBatch, groundTruthAction_: actionBatch, groundTruthValue_: valueBatch,
                        learningRate_: learningRate, actionLossCoef_: actionLossCoef, valueLossCoef_: valueLossCoef}
            reportDict, _ = model.run(fetches, feed_dict=feedDict)
            evalDict = reportDict['evaluations']
            self.reporter(stepNum, model, reportDict)

            if self.terminalController(evalDict, stepNum):
                break

        return model


def evaluate(model, testData, summaryOn=False, stepNum=None):
    graph = model.graph
    state_ = graph.get_collection_ref("inputs")[0]
    groundTruthAction_, groundTruthValue_ = graph.get_collection_ref("groundTruths")
    loss_ = graph.get_collection_ref("loss")[0]
    actionLoss_ = graph.get_collection_ref("actionLoss")[0]
    valueLoss_ = graph.get_collection_ref("valueLoss")[0]
    actionAccuracy_ = graph.get_collection_ref("actionAccuracy")[0]
    valueAccuracy_ = graph.get_collection_ref("valueAccuracy")[0]
    evalSummaryOp = graph.get_collection_ref('summaryOps')[1]
    testWriter = graph.get_collection_ref('writers')[1]
    fetches = [{"actionLoss": actionLoss_, "actionAcc": actionAccuracy_, "valueLoss": valueLoss_, "valueAcc": valueAccuracy_},
               evalSummaryOp]

    stateBatch, actionBatch, valueBatch = testData
    evalDict, summary = model.run(fetches, feed_dict={state_: stateBatch, groundTruthAction_: actionBatch, groundTruthValue_: valueBatch})
    if summaryOn:
        testWriter.add_summary(summary, stepNum)
    return evalDict


def sampleData(data, batchSize):
    batch = [list(varBatch) for varBatch in zip(*random.sample(data, batchSize))]
    return batch


def saveVariables(model, path):
    graph = model.graph
    saver = graph.get_collection_ref("saver")[0]
    saver.save(model, path)
    print("Model saved in {}".format(path))


def restoreVariables(model, path):
    graph = model.graph
    saver = graph.get_collection_ref("saver")[0]
    saver.restore(model, path)
    print("Model restored from {}".format(path))
    return model


class ApproximatePolicy:
    def __init__ (self, policyValueNet, actionSpace):
        self.policyValueNet = policyValueNet
        self.actionSpace = actionSpace

    def __call__(self, stateBatch):
        if np.array(stateBatch).ndim == 3:
            stateBatch = [np.concatenate(state) for state in stateBatch]
        if np.array(stateBatch).ndim == 2:
            stateBatch = np.concatenate(stateBatch)
        if np.array(stateBatch).ndim == 1:
            stateBatch = np.array([stateBatch])
        graph = self.policyValueNet.graph
        state_ = graph.get_collection_ref("inputs")[0]
        actionDist_ = graph.get_collection_ref("actionDistributions")[0]
        actionProbs = self.policyValueNet.run(actionDist_, feed_dict={state_: stateBatch})[0]
        actionDist = {action: prob for action, prob in zip(self.actionSpace, actionProbs)}
        return actionDist


class ApproximateValue:
    def __init__(self, policyValueNet):
        self.policyValueNet = policyValueNet

    def __call__(self, stateBatch):
        scalarOutput = False
        if np.array(stateBatch).ndim == 3:
            stateBatch = [np.concatenate(state) for state in stateBatch]
        if np.array(stateBatch).ndim == 2:
            stateBatch = np.concatenate(stateBatch)
        if np.array(stateBatch).ndim == 1:
            stateBatch = np.array([stateBatch])
            scalarOutput = True
        graph = self.policyValueNet.graph
        state_ = graph.get_collection_ref("inputs")[0]
        valuePrediction_ = graph.get_collection_ref("values")[0]
        valuePrediction = self.policyValueNet.run(valuePrediction_, feed_dict={state_: stateBatch})
        if scalarOutput:
            valuePrediction = valuePrediction[0][0]
        return valuePrediction






# structure?
# critic network parameter dimension? (the placeholder in input section)


