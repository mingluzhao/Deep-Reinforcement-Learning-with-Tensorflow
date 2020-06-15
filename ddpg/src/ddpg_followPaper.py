import tensorflow as tf
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tensorflow.contrib.layers as layers
import maddpg.maddpgAlgor.common.tf_util as U


class ActOneStepMADDPGWithNoise:
    def __init__(self, actorModel, actByPolicyTrain):
        self.actorModel = actorModel
        self.actByPolicyTrain = actByPolicyTrain

    def __call__(self, agentObs, runTimeStep = None):
        Action = self.actByPolicyTrain(self.actorModel, agentObs)
        action = Action[0]

        return action


def actByPolicyTrain(actorModel, stateBatch):
    # stateBatch = np.array([[ 0., 0., -0.91739134,  0.96584283,  0.30922375, -0.95104116, 0. , 0.]])

    actorGraph = actorModel.graph
    states_ = actorGraph.get_collection_ref("states_")[0]
    noisyTrainAction_ = actorGraph.get_collection_ref("noisyTrainAction_")[0]
    noisyTrainAction = actorModel.run(noisyTrainAction_, feed_dict={states_: stateBatch})

    # trainAction_ = actorGraph.get_collection_ref('trainAction_')[0]
    # trainAction = actorModel.run(trainAction_, feed_dict={states_: stateBatch})
    #
    # sampleNoiseTrain_ = actorGraph.get_collection_ref('sampleNoiseTrain_')[0]
    # sampleNoiseTrain = actorModel.run(sampleNoiseTrain_, feed_dict={states_: stateBatch})

    return noisyTrainAction

def actByPolicyTarget(actorModel, stateBatch):
    actorGraph = actorModel.graph
    states_ = actorGraph.get_collection_ref("states_")[0]
    noisyTargetAction_ = actorGraph.get_collection_ref("noisyTargetAction_")[0]
    noisyTargetAction = actorModel.run(noisyTargetAction_, feed_dict={states_: stateBatch})
    return noisyTargetAction

def evaluateCriticTarget(criticModel, stateBatch, actionsBatch):
    criticGraph = criticModel.graph
    states_ = criticGraph.get_collection_ref("states_")[0]
    actionTarget_ = criticGraph.get_collection_ref("actionTarget_")[0]
    targetQ_ = criticGraph.get_collection_ref("targetQ_")[0]
    targetQ = criticModel.run(targetQ_, feed_dict={states_: stateBatch, actionTarget_: actionsBatch})
    return targetQ

def evaluateCriticTrain(criticModel, stateBatch, actionsBatch):
    criticGraph = criticModel.graph
    states_ = criticGraph.get_collection_ref("states_")[0]
    action_ = criticGraph.get_collection_ref("action_")[0]
    trainQ_ = criticGraph.get_collection_ref("trainQ_")[0]
    trainQ = criticModel.run(trainQ_, feed_dict={states_: stateBatch, action_: actionsBatch})
    return trainQ

class BuildActorModel:
    def __init__(self, numStateSpace, actionDim, actionRange = 1):
        self.numStateSpace = numStateSpace
        self.actionDim = actionDim
        self.actionRange = actionRange

    def __call__(self, layersWidths, agentID = None):
        agentStr = 'Agent'+ str(agentID) if agentID is not None else ''
        print("Generating Actor NN Model with layers: {}".format(layersWidths))
        graph = tf.Graph()
        with graph.as_default():
            with tf.name_scope("inputs"+ agentStr):
                states_ = tf.placeholder(tf.float32, [None, self.numStateSpace], name='states_')
                qVal_ = tf.placeholder(tf.float32, [None, 1], name='qVal_')

                tf.add_to_collection("states_", states_)
                tf.add_to_collection("qVal_", qVal_)

            with tf.name_scope("trainingParams"+ agentStr):
                learningRate_ = tf.constant(0, dtype=tf.float32)
                tau_ = tf.constant(0, dtype=tf.float32)
                tf.add_to_collection("learningRate_", learningRate_)
                tf.add_to_collection("tau_", tau_)

            with tf.variable_scope("trainHidden"+ agentStr):
                activation_ = states_
                for i in range(len(layersWidths)):
                    # activation_ = layers.fully_connected(activation_, num_outputs= layersWidths[i], activation_fn=tf.nn.relu,
                    #                                      scope="fc{}".format(i+1), weights_initializer=tf.initializers.glorot_uniform(seed=0))
                    activation_ = layers.fully_connected(activation_, num_outputs= layersWidths[i], activation_fn=tf.nn.relu,
                                                         scope="fc{}".format(i+1))

                trainActivationOutput_ = layers.fully_connected(activation_, num_outputs= self.actionDim, activation_fn= None,
                                                                scope="fc{}".format(len(layersWidths)+1))

            with tf.variable_scope("targetHidden"+ agentStr):
                activation_ = states_
                for i in range(len(layersWidths)):
                    activation_ = layers.fully_connected(activation_, num_outputs= layersWidths[i], activation_fn=tf.nn.relu,
                                                         scope="fc{}".format(i+1))

                targetActivationOutput_ = layers.fully_connected(activation_, num_outputs= self.actionDim, activation_fn=None,
                                                                 scope="fc{}".format(len(layersWidths)+1))

            with tf.name_scope("updateParameters"+ agentStr):
                trainParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='trainHidden')
                targetParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='targetHidden')
                updateParam_ = [targetParams_[i].assign((1 - tau_) * targetParams_[i] + tau_ * trainParams_[i]) for i in range(len(targetParams_))]

                tf.add_to_collection("trainParams_", trainParams_)
                tf.add_to_collection("targetParams_", targetParams_)
                tf.add_to_collection("updateParam_", updateParam_)

                hardReplaceTargetParam_ = [tf.assign(trainParam, targetParam) for trainParam, targetParam in zip(trainParams_, targetParams_)]
                tf.add_to_collection("hardReplaceTargetParam_", hardReplaceTargetParam_)

            with tf.name_scope("output"+ agentStr):
                trainAction_ = tf.multiply(trainActivationOutput_, self.actionRange, name='trainAction_')
                targetAction_ = tf.multiply(targetActivationOutput_, self.actionRange, name='targetAction_')

                sampleNoiseTrain_ = tf.random_uniform(tf.shape(trainActivationOutput_))
                noisyTrainAction_ = U.softmax(trainActivationOutput_ - tf.log(-tf.log(sampleNoiseTrain_)), axis=-1) # give this to q input

                tf.add_to_collection("sampleNoiseTrain_", sampleNoiseTrain_)

                sampleNoiseTarget_ = tf.random_uniform(tf.shape(targetActivationOutput_))
                noisyTargetAction_ = U.softmax(targetActivationOutput_ - tf.log(-tf.log(sampleNoiseTarget_)), axis=-1)

                tf.add_to_collection("trainAction_", trainAction_)
                tf.add_to_collection("targetAction_", targetAction_)

                tf.add_to_collection("noisyTrainAction_", noisyTrainAction_)
                tf.add_to_collection("noisyTargetAction_", noisyTargetAction_)

            with tf.name_scope("train"+ agentStr):
                p_reg = tf.reduce_mean(tf.square(trainActivationOutput_))
                pg_loss = -tf.reduce_mean(qVal_)
                actorLoss_ = pg_loss + p_reg * 1e-3

                tf.summary.scalar("pg_loss", pg_loss)
                tf.add_to_collection("actorLoss_", actorLoss_)

                # optimizer = tf.train.AdamOptimizer(learningRate_, name='adamOptimizer')
                # grad_norm_clipping = 0.5
                # trainOpt_ = U.minimize_and_clip(optimizer, actorLoss_, trainParams_, grad_norm_clipping)


                optimizer = tf.train.AdamOptimizer(learningRate_, name='adamOptimizer')
                grad_norm_clipping = 0.5

                gradients = optimizer.compute_gradients(actorLoss_, var_list=trainParams_)
                for i, (grad, var) in enumerate(gradients):
                    if grad is not None:
                        gradients[i] = (tf.clip_by_norm(grad, grad_norm_clipping), var)

                with tf.name_scope("inspectGrad"):
                    for i, (grad_, var_) in enumerate(gradients):
                        keyPrefix = "weightGradient" if "weights" in var_.name else "biasGradient"
                        tf.add_to_collection(f"{keyPrefix}/{var_.name}", grad_)
                    gradients_ = [tf.reshape(grad_, [1, -1]) for i, (grad_, var_) in enumerate(gradients)]
                    allGradTensor_ = tf.concat(gradients_, 1)
                    allGradNorm_ = tf.norm(allGradTensor_)
                    tf.add_to_collection("allGradNorm", allGradNorm_)
                    tf.summary.histogram("allGradients", allGradTensor_)
                    tf.summary.scalar("allGradNorm", allGradNorm_)


                trainOpt_ =  optimizer.apply_gradients(gradients)

                tf.add_to_collection("trainOpt_", trainOpt_)

            with tf.name_scope("summary"+ agentStr):
                actorLossSummary_ = tf.identity(actorLoss_)
                tf.add_to_collection("actorLossSummary_", actorLossSummary_)
                tf.summary.scalar("actorLossSummary", actorLossSummary_)


            fullSummary = tf.summary.merge_all()
            tf.add_to_collection("summaryOps", fullSummary)

            actorSaver = tf.train.Saver(max_to_keep=None)
            tf.add_to_collection("saver", actorSaver)

            model = tf.Session(graph=graph)
            model.run(tf.global_variables_initializer())

            actorWriter = tf.summary.FileWriter('tensorBoard/actorOnlineDDPG'+ agentStr, graph= graph)
            tf.add_to_collection("actorWriter", actorWriter)

        return actorWriter, model


class BuildCriticModel:
    def __init__(self, numStateSpace, actionDim):
        self.numStateSpace = numStateSpace
        self.actionDim = actionDim

    def __call__(self, layersWidths, agentID = None):
        agentStr = 'Agent'+ str(agentID) if agentID is not None else ''
        print("Generating Critic NN Model with layers: {}".format(layersWidths))
        graph = tf.Graph()
        with graph.as_default():
            with tf.name_scope("inputs" + agentStr):
                states_ = tf.placeholder(tf.float32, [None, self.numStateSpace], name='states_')
                action_ = tf.stop_gradient(tf.placeholder(tf.float32, [None, self.actionDim]), name='action_')

                actionTarget_ = tf.placeholder(tf.float32, [None, self.actionDim], name='actionTarget_')
                reward_ = tf.placeholder(tf.float32, [None, 1], name='reward_')
                valueTarget_ = tf.placeholder(tf.float32, [None, 1], name='valueTarget_')

                tf.add_to_collection("states_", states_)
                tf.add_to_collection("action_", action_)
                tf.add_to_collection("actionTarget_", actionTarget_)
                tf.add_to_collection("reward_", reward_)
                tf.add_to_collection("valueTarget_", valueTarget_)

            with tf.name_scope("trainingParams" + agentStr):
                learningRate_ = tf.constant(0, dtype=tf.float32)
                tau_ = tf.constant(0, dtype=tf.float32)
                gamma_ = tf.constant(0, dtype=tf.float32)

                tf.add_to_collection("learningRate_", learningRate_)
                tf.add_to_collection("tau_", tau_)
                tf.add_to_collection("gamma_", gamma_)

            with tf.variable_scope("trainHidden"+ agentStr):
                activation_ = tf.concat([states_, action_], axis=1)
                for i in range(len(layersWidths)):
                    activation_ = layers.fully_connected(activation_, num_outputs= layersWidths[i], activation_fn=tf.nn.relu, scope="fc{}".format(i+1) )

                trainValues_ = layers.fully_connected(activation_, num_outputs= 1, activation_fn= tf.nn.tanh, scope="fc{}".format(len(layersWidths)+1) )

            with tf.variable_scope("targetHidden"+ agentStr):
                activation_ = tf.concat([states_, actionTarget_], axis=1)
                for i in range(len(layersWidths)):
                    activation_ = layers.fully_connected(activation_, num_outputs= layersWidths[i], activation_fn=tf.nn.relu, scope="fc{}".format(i+1) )

                targetValues_ = layers.fully_connected(activation_, num_outputs= 1, activation_fn= tf.nn.tanh, scope="fc{}".format(len(layersWidths)+1) )

            with tf.name_scope("parameters"+ agentStr):
                trainParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='trainHidden')
                targetParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='targetHidden')
                updateParam_ = [targetParams_[i].assign((1 - tau_) * targetParams_[i] + tau_ * trainParams_[i]) for i in range(len(targetParams_))]

                tf.add_to_collection("trainParams_", trainParams_)
                tf.add_to_collection("targetParams_", targetParams_)
                tf.add_to_collection("updateParam_", updateParam_)

                hardReplaceTargetParam_ = [tf.assign(trainParam, targetParam) for trainParam, targetParam in zip(trainParams_, targetParams_)]
                tf.add_to_collection("hardReplaceTargetParam_", hardReplaceTargetParam_)

            with tf.name_scope("output"+ agentStr):
                trainQ_ = tf.multiply(trainValues_, 1, name='trainQ_')
                targetQ_ = tf.multiply(targetValues_, 1, name='targetQ_')
                tf.add_to_collection("trainQ_", trainQ_)
                tf.add_to_collection("targetQ_", targetQ_)

            with tf.name_scope("evaluate"+ agentStr):
                yi_ = reward_ + gamma_ * valueTarget_
                # criticLoss_ = tf.losses.mean_squared_error(labels=yi_, predictions=trainQ_)

                criticLoss_ = tf.reduce_mean(tf.squared_difference(tf.squeeze(yi_), tf.squeeze(trainQ_)))

                # loss = tf.reduce_mean(tf.square(q - yi_))

                tf.add_to_collection("yi_", yi_)
                tf.add_to_collection("valueLoss_", criticLoss_)

            with tf.name_scope("train"+ agentStr):
                # trainOpt_ = tf.train.AdamOptimizer(learningRate_, name='adamOptimizer').minimize(criticLoss_, var_list=trainParams_)
                optimizer = tf.train.AdamOptimizer(learningRate_, name='adamOptimizer')
                grad_norm_clipping = 0.5
                trainOpt_ = U.minimize_and_clip(optimizer, criticLoss_, trainParams_, grad_norm_clipping)

                tf.add_to_collection("trainOpt_", trainOpt_)

            with tf.name_scope("summary"+ agentStr):
                criticLossSummary = tf.identity(criticLoss_)
                tf.add_to_collection("criticLossSummary", criticLossSummary)
                tf.summary.scalar("criticLossSummary", criticLossSummary)


            fullSummary = tf.summary.merge_all()
            tf.add_to_collection("summaryOps", fullSummary)

            criticSaver = tf.train.Saver(max_to_keep=None)
            tf.add_to_collection("saver", criticSaver)

            model = tf.Session(graph=graph)
            model.run(tf.global_variables_initializer())

            criticWriter = tf.summary.FileWriter('tensorBoard/criticOnlineDDPG'+ agentStr, graph=graph)
            tf.add_to_collection("criticWriter", criticWriter)

        return criticWriter, model


class TrainCriticBySASRQ:
    def __init__(self, criticLearningRate, gamma, criticWriter):
        self.criticLearningRate = criticLearningRate
        self.gamma = gamma
        self.criticWriter = criticWriter
        self.runCount = 0

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

        criticSummary_ = criticGraph.get_collection_ref("summaryOps")[0]

        criticSummary, criticLoss, trainOpt = criticModel.run([criticSummary_, valueLoss_, trainOpt_],
                                               feed_dict={states_: stateBatch, action_: actionBatch, reward_: rewardBatch, valueTarget_: targetQValue,
                                                          learningRate_: self.criticLearningRate, gamma_: self.gamma})

        self.criticWriter.add_summary(criticSummary, self.runCount)
        self.runCount += 1
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

        criticLoss, criticModel = self.trainCriticBySASRQ(criticModel, stateBatch, actionBatch, rewardBatch, targetQValue)
        return criticLoss, criticModel


class TrainActorFromQVal:
    def __init__(self, actorLearningRate, actorWriter):
        self.actorLearningRate = actorLearningRate
        self.actorWriter = actorWriter
        self.runCount = 0

    def __call__(self, actorModel, stateBatch, qVal):
        # print(np.mean(qVal))
        actorGraph = actorModel.graph
        states_ = actorGraph.get_collection_ref("states_")[0]
        qVal_ = actorGraph.get_collection_ref("qVal_")[0]
        learningRate_ = actorGraph.get_collection_ref("learningRate_")[0]
        actorSummary_ = actorGraph.get_collection_ref("summaryOps")[0]
        actorLoss_ = actorGraph.get_collection_ref("actorLoss_")[0]

        trainOpt_ = actorGraph.get_collection_ref("trainOpt_")[0]
        actorSummary, actorLoss, trainOpt = actorModel.run([actorSummary_, actorLoss_, trainOpt_],
                                                feed_dict={states_: stateBatch, qVal_: qVal, learningRate_: self.actorLearningRate})
        # self.actorWriter.flush()
        self.actorWriter.add_summary(actorSummary, self.runCount)
        self.runCount += 1

        return actorLoss, actorModel


class TrainActorOneStep:
    def __init__(self, actByPolicyTrain, trainActorFromGradients, evaluateCriticTrain):
        self.actByPolicyTrain = actByPolicyTrain
        self.trainActorFromGradients = trainActorFromGradients
        self.evaluateCriticTrain = evaluateCriticTrain

    def __call__(self, actorModel, criticModel, stateBatch):
        actionsBatch = self.actByPolicyTrain(actorModel, stateBatch)
        qVal = self.evaluateCriticTrain(criticModel, stateBatch, actionsBatch)
        actorLoss, actorModel = self.trainActorFromGradients(actorModel, stateBatch, qVal)
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