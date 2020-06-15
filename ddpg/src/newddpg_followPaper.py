import tensorflow as tf
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tensorflow.contrib.layers as layers
import maddpg.maddpgAlgor.common.tf_util as U


class BuildDDPGModels:
    def __init__(self, numStateSpace, actionDim, actionRange = 1):
        self.numStateSpace = numStateSpace
        self.actionDim = actionDim
        self.actionRange = actionRange

    def __call__(self, layersWidths, agentID = None):
        agentStr = 'Agent'+ str(agentID) if agentID is not None else ''
        graph = tf.Graph()
        with graph.as_default():
            with tf.variable_scope("inputs/"+ agentStr):
                states_ = tf.placeholder(tf.float32, [None, self.numStateSpace], name='states_')
                nextStates_ = tf.placeholder(tf.float32, [None, self.numStateSpace], name='nextStates_')
                action_ = tf.stop_gradient(tf.placeholder(tf.float32, [None, self.actionDim]), name='action_')
                reward_ = tf.placeholder(tf.float32, [None, 1], name='reward_')

                tf.add_to_collection("states_", states_)
                tf.add_to_collection("nextStates_", nextStates_)
                tf.add_to_collection("action_", action_)
                tf.add_to_collection("reward_", reward_)

            with tf.variable_scope("trainingParams" + agentStr):
                learningRate_ = tf.constant(0, dtype=tf.float32)
                tau_ = tf.constant(0, dtype=tf.float32)
                gamma_ = tf.constant(0, dtype=tf.float32)

                tf.add_to_collection("learningRate_", learningRate_)
                tf.add_to_collection("tau_", tau_)
                tf.add_to_collection("gamma_", gamma_)

            with tf.variable_scope("actor/trainHidden/"+ agentStr):
                actorTrainActivation_ = states_
                for i in range(len(layersWidths)):
                    actorTrainActivation_ = layers.fully_connected(actorTrainActivation_, num_outputs= layersWidths[i],
                                                                   activation_fn=tf.nn.relu)

                actorTrainActivation_ = layers.fully_connected(actorTrainActivation_, num_outputs= self.actionDim,
                                                               activation_fn= None)

            with tf.variable_scope("actor/targetHidden/"+ agentStr):
                actorTargetActivation_ = nextStates_
                for i in range(len(layersWidths)):
                    actorTargetActivation_ = layers.fully_connected(actorTargetActivation_, num_outputs= layersWidths[i],
                                                                    activation_fn=tf.nn.relu)

                actorTargetActivation_ = layers.fully_connected(actorTargetActivation_, num_outputs= self.actionDim,
                                                                activation_fn=None)

            with tf.variable_scope("actorNetOutput/"+ agentStr):
                trainAction_ = tf.multiply(actorTrainActivation_, self.actionRange, name='trainAction_')
                targetAction_ = tf.multiply(actorTargetActivation_, self.actionRange, name='targetAction_')

                sampleNoiseTrain_ = tf.random_uniform(tf.shape(trainAction_))
                noisyTrainAction_ = U.softmax(trainAction_ - tf.log(-tf.log(sampleNoiseTrain_)), axis=-1) # give this to q input

                sampleNoiseTarget_ = tf.random_uniform(tf.shape(targetAction_))
                noisyTargetAction_ = U.softmax(targetAction_ - tf.log(-tf.log(sampleNoiseTarget_)), axis=-1)

                tf.add_to_collection("trainAction_", trainAction_)
                tf.add_to_collection("targetAction_", targetAction_)

                tf.add_to_collection("noisyTrainAction_", noisyTrainAction_)
                tf.add_to_collection("noisyTargetAction_", noisyTargetAction_)


            with tf.variable_scope("critic/trainHidden/"+ agentStr):
                criticTrainActivationOfGivenAction_ = tf.concat([states_, action_], axis=1)
                for i in range(len(layersWidths)):
                    criticTrainActivationOfGivenAction_ = layers.fully_connected(criticTrainActivationOfGivenAction_, num_outputs= layersWidths[i],
                                                                   activation_fn=tf.nn.relu)

                criticTrainActivationOfGivenAction_ = layers.fully_connected(criticTrainActivationOfGivenAction_, num_outputs= 1,
                                                               activation_fn= None)

            with tf.variable_scope("critic/trainHidden/" + agentStr, reuse= True):
                criticTrainActivation_ = tf.concat([states_, noisyTrainAction_], axis=1)
                for i in range(len(layersWidths)):
                    criticTrainActivation_ = layers.fully_connected(criticTrainActivation_,
                                                                    num_outputs=layersWidths[i],
                                                                    activation_fn=tf.nn.relu)

                criticTrainActivation_ = layers.fully_connected(criticTrainActivation_, num_outputs=1,
                                                                activation_fn=None)

            with tf.variable_scope("critic/targetHidden/"+ agentStr):
                criticTargetActivation_ = tf.concat([nextStates_, noisyTargetAction_], axis=1)
                for i in range(len(layersWidths)):
                    criticTargetActivation_ = layers.fully_connected(criticTargetActivation_, num_outputs= layersWidths[i],activation_fn=tf.nn.relu)

                criticTargetActivation_ = layers.fully_connected(criticTargetActivation_, num_outputs= 1,activation_fn=None)

            with tf.variable_scope("updateParameters/"+ agentStr):
                actorTrainParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor/trainHidden/'+ agentStr)
                actorTargetParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor/targetHidden/'+ agentStr)
                actorUpdateParam_ = [actorTargetParams_[i].assign((1 - tau_) * actorTargetParams_[i] + tau_ * actorTrainParams_[i]) for i in range(len(actorTargetParams_))]

                tf.add_to_collection("actorTrainParams_", actorTrainParams_)
                tf.add_to_collection("actorTargetParams_", actorTargetParams_)
                tf.add_to_collection("actorUpdateParam_", actorUpdateParam_)

                hardReplaceActorTargetParam_ = [tf.assign(trainParam, targetParam) for trainParam, targetParam in zip(actorTrainParams_, actorTargetParams_)]
                tf.add_to_collection("hardReplaceActorTargetParam_", hardReplaceActorTargetParam_)

                criticTrainParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic/trainHidden/'+ agentStr)
                criticTargetParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic/targetHidden/'+ agentStr)

                criticUpdateParam_ = [criticTargetParams_[i].assign((1 - tau_) * criticTargetParams_[i] + tau_ * criticTrainParams_[i]) for i in range(len(criticTargetParams_))]

                tf.add_to_collection("criticTrainParams_", criticTrainParams_)
                tf.add_to_collection("criticTargetParams_", criticTargetParams_)
                tf.add_to_collection("criticUpdateParam_", criticUpdateParam_)

                hardReplaceCriticTargetParam_ = [tf.assign(trainParam, targetParam) for trainParam, targetParam in zip(criticTrainParams_, criticTargetParams_)]
                tf.add_to_collection("hardReplaceCriticTargetParam_", hardReplaceCriticTargetParam_)

                updateParam_ = actorUpdateParam_ + criticUpdateParam_
                hardReplaceTargetParam_ = hardReplaceActorTargetParam_ + hardReplaceCriticTargetParam_
                tf.add_to_collection("updateParam_", updateParam_)
                tf.add_to_collection("hardReplaceTargetParam_", hardReplaceTargetParam_)


            with tf.variable_scope("trainActorNet/"+ agentStr):
                trainQ = criticTrainActivation_[:, 0]
                pg_loss = -tf.reduce_mean(trainQ)
                p_reg = tf.reduce_mean(tf.square(actorTrainActivation_))
                actorLoss_ = pg_loss + p_reg * 1e-3

                actorOptimizer = tf.train.AdamOptimizer(learningRate_, name='actorOptimizer')
                grad_norm_clipping = 0.5
                actorTrainOpt_ = U.minimize_and_clip(actorOptimizer, actorLoss_, actorTrainParams_, grad_norm_clipping)

                tf.add_to_collection("actorLoss_", actorLoss_)
                tf.add_to_collection("actorTrainOpt_", actorTrainOpt_)

            with tf.variable_scope("trainCriticNet/"+ agentStr):
                yi_ = reward_ + gamma_ * criticTargetActivation_
                criticLoss_ = tf.reduce_mean(tf.squared_difference(tf.squeeze(yi_), tf.squeeze(criticTrainActivationOfGivenAction_)))

                tf.add_to_collection("yi_", yi_)
                tf.add_to_collection("valueLoss_", criticLoss_)

                criticOptimizer = tf.train.AdamOptimizer(learningRate_, name='criticOptimizer')
                grad_norm_clipping = 0.5
                crticTrainOpt_ = U.minimize_and_clip(criticOptimizer, criticLoss_, criticTrainParams_, grad_norm_clipping)

                tf.add_to_collection("crticTrainOpt_", crticTrainOpt_)

            with tf.variable_scope("summary"+ agentStr):
                criticLossSummary = tf.identity(criticLoss_)
                tf.add_to_collection("criticLossSummary", criticLossSummary)
                tf.summary.scalar("criticLossSummary", criticLossSummary)

            fullSummary = tf.summary.merge_all()
            tf.add_to_collection("summaryOps", fullSummary)

            saver = tf.train.Saver(max_to_keep=None)
            tf.add_to_collection("saver", saver)

            model = tf.Session(graph=graph)
            model.run(tf.global_variables_initializer())

            writer = tf.summary.FileWriter('tensorBoard/onlineDDPG/'+ agentStr, graph= graph)
            tf.add_to_collection("writer", writer)

        return writer, model


class ActOneStepMADDPGWithNoise:
    def __init__(self, model, actByPolicyTrainNoisy):
        self.model = model
        self.actByPolicyTrainNoisy = actByPolicyTrainNoisy

    def __call__(self, agentObs, runTimeStep = None):
        Action = self.actByPolicyTrainNoisy(self.model, agentObs)
        action = Action[0]

        return action

def actByPolicyTrainNoisy(model, stateBatch):
    graph = model.graph
    states_ = graph.get_collection_ref("states_")[0]
    noisyTrainAction_ = graph.get_collection_ref("noisyTrainAction_")[0]
    noisyTrainAction = model.run(noisyTrainAction_, feed_dict={states_: stateBatch})

    return noisyTrainAction


class TrainCriticBySASR:
    def __init__(self, criticLearningRate, gamma, writer):
        self.criticLearningRate = criticLearningRate
        self.gamma = gamma
        self.writer = writer
        self.runCount = 0

    def __call__(self, model, stateBatch, actionBatch, nextStateBatch, rewardBatch):
        graph = model.graph
        states_ = graph.get_collection_ref("states_")[0]
        nextStates_ = graph.get_collection_ref("nextStates_")[0]
        action_ = graph.get_collection_ref("action_")[0]
        reward_ = graph.get_collection_ref("reward_")[0]
        learningRate_ = graph.get_collection_ref("learningRate_")[0]
        gamma_ = graph.get_collection_ref("gamma_")[0]

        valueLoss_ = graph.get_collection_ref("valueLoss_")[0]
        crticTrainOpt_ = graph.get_collection_ref("crticTrainOpt_")[0]
        criticSummary_ = graph.get_collection_ref("summaryOps")[0]

        criticSummary, criticLoss, crticTrainOpt = model.run([criticSummary_, valueLoss_, crticTrainOpt_],
                                               feed_dict={states_: stateBatch, action_: actionBatch, reward_: rewardBatch,
                                                          nextStates_: nextStateBatch,
                                                          learningRate_: self.criticLearningRate, gamma_: self.gamma})

        self.writer.add_summary(criticSummary, self.runCount)
        self.runCount += 1

        return criticLoss, model


def reshapeBatchToGetSASR(miniBatch):
    states, actions, rewards, nextStates = list(zip(*miniBatch))
    stateBatch = np.asarray(states).reshape(len(miniBatch), -1)
    actionBatch = np.asarray(actions).reshape(len(miniBatch), -1)
    nextStateBatch = np.asarray(nextStates).reshape(len(miniBatch), -1)
    rewardBatch = np.asarray(rewards).reshape(len(miniBatch), -1)

    return stateBatch, actionBatch, nextStateBatch, rewardBatch


class TrainCritic:
    def __init__(self, reshapeBatchToGetSASR, trainCriticBySASR):
        self.reshapeBatchToGetSASR = reshapeBatchToGetSASR
        self.trainCriticBySASR = trainCriticBySASR

    def __call__(self, model, miniBatch):
        stateBatch, actionBatch, nextStateBatch, rewardBatch = self.reshapeBatchToGetSASR(miniBatch)
        criticLoss, model = self.trainCriticBySASR(model, stateBatch, actionBatch, nextStateBatch, rewardBatch)

        return criticLoss, model


class TrainActorFromState:
    def __init__(self, actorLearningRatte, writer):
        self.actorLearningRate = actorLearningRatte
        self.writer = writer

    def __call__(self, model, stateBatch):
        graph = model.graph
        states_ = graph.get_collection_ref("states_")[0]
        learningRate_ = graph.get_collection_ref("learningRate_")[0]
        actorTrainOpt_ = graph.get_collection_ref("actorTrainOpt_")[0]
        actorTrainOpt = model.run(actorTrainOpt_, feed_dict={states_: stateBatch, learningRate_: self.actorLearningRate})
        self.writer.flush()
        return model


class TrainActor:
    def __init__(self, reshapeBatchToGetSASR, trainActorFromState):
        self.trainActorFromState = trainActorFromState
        self.reshapeBatchToGetSASR = reshapeBatchToGetSASR

    def __call__(self, model, miniBatch):
        stateBatch, actionBatch, nextStateBatch, rewardBatch = self.reshapeBatchToGetSASR(miniBatch)
        model = self.trainActorFromState(model, stateBatch)

        return model


class TrainDDPGModels:
    def __init__(self, updateParameters, trainActor, trainCritic, model):
        self.updateParameters = updateParameters
        self.trainActor = trainActor
        self.trainCritic = trainCritic
        self.model = model

    def __call__(self, miniBatch):
        criticLoss, self.model = self.trainCritic(self.model, miniBatch)
        self.model = self.trainActor(self.model, miniBatch)
        self.model = self.updateParameters(self.model)

    def getTrainedModels(self):
        return self.model















