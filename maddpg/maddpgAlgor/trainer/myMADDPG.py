import tensorflow as tf
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tensorflow.contrib.layers as layers
import maddpg.maddpgAlgor.common.tf_util as U


class BuildMADDPGModels:
    def __init__(self, actionDim, numAgents, obsShapeList, actionRange = 1):
        self.actionDim = actionDim
        self.numAgents = numAgents
        self.obsShapeList = obsShapeList
        self.actionRange = actionRange
        self.gradNormClipping = 0.5

    def __call__(self, layersWidths, agentID):
        agentStr = 'Agent'+ str(agentID)
        graph = tf.Graph()
        with graph.as_default():
            with tf.variable_scope("inputs/"+ agentStr):
                allAgentsStates_ = [tf.placeholder(dtype=tf.float32, shape=[None, agentObsDim], name="state"+str(i)) for i, agentObsDim in enumerate(self.obsShapeList)]
                allAgentsNextStates_ =  [tf.placeholder(dtype=tf.float32, shape=[None, agentObsDim], name="nextState"+str(i)) for i, agentObsDim in enumerate(self.obsShapeList)]

                allAgentsActions_ = [tf.placeholder(dtype=tf.float32, shape=[None, self.actionDim], name="action"+str(i)) for i in range(self.numAgents)]
                allAgentsNextActionsByTargetNet_ = [tf.placeholder(dtype=tf.float32, shape=[None, self.actionDim], name= "actionTarget"+str(i)) for i in range(self.numAgents)]

                agentReward_ = tf.placeholder(tf.float32, [None, 1], name='reward_')

                tf.add_to_collection("allAgentsStates_", allAgentsStates_)
                tf.add_to_collection("allAgentsNextStates_", allAgentsNextStates_)
                tf.add_to_collection("allAgentsActions_", allAgentsActions_)
                tf.add_to_collection("allAgentsNextActionsByTargetNet_", allAgentsNextActionsByTargetNet_)
                tf.add_to_collection("agentReward_", agentReward_)

            with tf.variable_scope("trainingParams" + agentStr):
                learningRate_ = tf.constant(0, dtype=tf.float32)
                tau_ = tf.constant(0, dtype=tf.float32)
                gamma_ = tf.constant(0, dtype=tf.float32)

                tf.add_to_collection("learningRate_", learningRate_)
                tf.add_to_collection("tau_", tau_)
                tf.add_to_collection("gamma_", gamma_)

            with tf.variable_scope("actor/trainHidden/"+ agentStr): # act by personal observation
                currentAgentState_ = allAgentsStates_[agentID]
                actorTrainActivation_ = currentAgentState_

                for i in range(len(layersWidths)):
                    actorTrainActivation_ = layers.fully_connected(actorTrainActivation_, num_outputs= layersWidths[i],
                                                                   activation_fn=tf.nn.relu)

                actorTrainActivation_ = layers.fully_connected(actorTrainActivation_, num_outputs= self.actionDim,
                                                               activation_fn= None)

            with tf.variable_scope("actor/targetHidden/"+ agentStr):
                currentAgentNextState_ = allAgentsNextStates_[agentID]
                actorTargetActivation_ = currentAgentNextState_

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
                criticTrainActivationOfGivenAction_ = tf.concat(allAgentsStates_ + allAgentsActions_, axis=1)

                for i in range(len(layersWidths)):
                    criticTrainActivationOfGivenAction_ = layers.fully_connected(criticTrainActivationOfGivenAction_, num_outputs= layersWidths[i], activation_fn=tf.nn.relu)

                criticTrainActivationOfGivenAction_ = layers.fully_connected(criticTrainActivationOfGivenAction_, num_outputs= 1, activation_fn= None)

            with tf.variable_scope("critic/trainHidden/" + agentStr, reuse= True):
                criticInputActionList = allAgentsActions_ + []
                criticInputActionList[agentID] = noisyTrainAction_
                criticTrainActivation_ = tf.concat(allAgentsStates_ + criticInputActionList, axis=1)

                for i in range(len(layersWidths)):
                    criticTrainActivation_ = layers.fully_connected(criticTrainActivation_, num_outputs=layersWidths[i], activation_fn=tf.nn.relu)

                criticTrainActivation_ = layers.fully_connected(criticTrainActivation_, num_outputs=1, activation_fn=None)

            with tf.variable_scope("critic/targetHidden/"+ agentStr):
                criticTargetActivation_ = tf.concat(allAgentsNextStates_ + allAgentsNextActionsByTargetNet_, axis=1)
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
                actorTrainOpt_ = U.minimize_and_clip(actorOptimizer, actorLoss_, actorTrainParams_, self.gradNormClipping)

                tf.add_to_collection("actorLoss_", actorLoss_)
                tf.add_to_collection("actorTrainOpt_", actorTrainOpt_)

            with tf.variable_scope("trainCriticNet/"+ agentStr):
                yi_ = agentReward_ + gamma_ * criticTargetActivation_
                criticLoss_ = tf.reduce_mean(tf.squared_difference(tf.squeeze(yi_), tf.squeeze(criticTrainActivationOfGivenAction_)))

                tf.add_to_collection("yi_", yi_)
                tf.add_to_collection("valueLoss_", criticLoss_)

                criticOptimizer = tf.train.AdamOptimizer(learningRate_, name='criticOptimizer')
                crticTrainOpt_ = U.minimize_and_clip(criticOptimizer, criticLoss_, criticTrainParams_, self.gradNormClipping)

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

        return model

class ActOneStep:
    def __init__(self, actByTrainNoisy):
        self.actByTrain = actByTrainNoisy

    def __call__(self, model, allAgentsStatesBatch):
        allAgentsStates = np.array(allAgentsStatesBatch)[None]
        actions = self.actByTrain(model, allAgentsStates)[0]
        return actions



def actByPolicyTrainNoisy(model, allAgentsStatesBatch):
    graph = model.graph
    allAgentsStates_ = graph.get_collection_ref("allAgentsStates_")[0]
    noisyTrainAction_ = graph.get_collection_ref("noisyTrainAction_")[0]
    stateDict = {agentState_: [states[i] for states in allAgentsStatesBatch] for i, agentState_ in enumerate(allAgentsStates_)}

    noisyTrainAction = model.run(noisyTrainAction_, feed_dict= stateDict)

    return noisyTrainAction


def actByPolicyTargetNoisyForNextState(model, allAgentsNextStatesBatch):
    graph = model.graph
    allAgentsNextStates_ = graph.get_collection_ref("allAgentsNextStates_")[0]
    noisyTargetAction_ = graph.get_collection_ref("noisyTargetAction_")[0]

    nextStateDict = {agentNextState_: [states[i] for states in allAgentsNextStatesBatch] for i, agentNextState_ in enumerate(allAgentsNextStates_)}
    noisyTargetAction = model.run(noisyTargetAction_, feed_dict= nextStateDict)

    return noisyTargetAction



class TrainCriticBySASR:
    def __init__(self, actByPolicyTargetNoisyForNextState, criticLearningRate, gamma):
        self.actByPolicyTargetNoisyForNextState = actByPolicyTargetNoisyForNextState
        self.criticLearningRate = criticLearningRate
        self.gamma = gamma
        self.runCount = 0

    def __call__(self, agentID, allAgentsModels, allAgentsStateBatch, allAgentsActionsBatch, allAgentsNextStatesBatch, allAgentsRewardBatch):
        agentModel = allAgentsModels[agentID]
        agentReward = [[reward[agentID]] for reward in allAgentsRewardBatch]
        graph = agentModel.graph

        allAgentsStates_ = graph.get_collection_ref("allAgentsStates_")[0]#
        allAgentsNextStates_ = graph.get_collection_ref("allAgentsNextStates_")[0]
        allAgentsNextActionsByTargetNet_ = graph.get_collection_ref("allAgentsNextActionsByTargetNet_")[0]
        agentReward_ = graph.get_collection_ref("agentReward_")[0]
        allAgentsActions_ = graph.get_collection_ref("allAgentsActions_")[0]

        learningRate_ = graph.get_collection_ref("learningRate_")[0]
        gamma_ = graph.get_collection_ref("gamma_")[0]

        valueLoss_ = graph.get_collection_ref("valueLoss_")[0]
        crticTrainOpt_ = graph.get_collection_ref("crticTrainOpt_")[0]
        criticSummary_ = graph.get_collection_ref("summaryOps")[0]

        # stateDict = {agentState: stateBatch for agentState, stateBatch in zip(allAgentsStates_, allAgentsStateBatch)}
        # nextStateDict = {agentState: stateBatch for agentState, stateBatch in zip(allAgentsNextStates_, allAgentsNextStatesBatch)}
        # nextActionDict = {action_: actionBatch for action_, actionBatch in zip(allAgentsNextActionsByTargetNet_, allAgentsNextTargetActions)}
        # actionDict = {action_: actionBatch for action_, actionBatch in zip(allAgentsActions_, allAgentsActionsBatch)}
        valueDict = {agentReward_: agentReward, learningRate_: self.criticLearningRate, gamma_: self.gamma}

        stateDict = {agentState_: [states[i] for states in allAgentsStateBatch] for i, agentState_ in enumerate(allAgentsStates_)}
        actionDict = {agentAction_: [actions[i] for actions in allAgentsActionsBatch] for i, agentAction_ in enumerate(allAgentsActions_)}
        nextStateDict = {agentNextState_: [states[i] for states in allAgentsNextStatesBatch] for i, agentNextState_ in enumerate(allAgentsNextStates_)}

        getAgentNextAction = lambda agentID: self.actByPolicyTargetNoisyForNextState(allAgentsModels[agentID], allAgentsNextStatesBatch)
        nextActionDict = {nextAction_: getAgentNextAction(i) for i, nextAction_ in enumerate(allAgentsNextActionsByTargetNet_)}


        criticSummary, criticLoss, crticTrainOpt = agentModel.run([criticSummary_, valueLoss_, crticTrainOpt_],
                                               feed_dict={**stateDict, **nextStateDict, **nextActionDict, **actionDict, **valueDict} )

        # self.writer.add_summary(criticSummary, self.runCount)
        self.runCount += 1

        return criticLoss, agentModel



class TrainCritic:
    def __init__(self, trainCriticBySASR):
        self.trainCriticBySASR = trainCriticBySASR

    def __call__(self, agentID, allAgentsModels, miniBatch):
        allAgentsStateBatch, allAgentsActionsBatch, allAgentsRewardBatch, allAgentsNextStatesBatch = list(zip(*miniBatch))
        criticLoss, agentModel = self.trainCriticBySASR(agentID, allAgentsModels, allAgentsStateBatch, allAgentsActionsBatch, allAgentsNextStatesBatch, allAgentsRewardBatch)

        return agentModel



class TrainActorFromSA:
    def __init__(self, actorLearningRatte):
        self.actorLearningRate = actorLearningRatte
        # self.writer = writer

    def __call__(self, agentID, agentModel, allAgentsStateBatch, allAgentsActionsBatch):
        graph = agentModel.graph
        allAgentsStates_ = graph.get_collection_ref("allAgentsStates_")[0]#
        allAgentsActions_ = graph.get_collection_ref("allAgentsActions_")[0]

        learningRate_ = graph.get_collection_ref("learningRate_")[0]
        actorTrainOpt_ = graph.get_collection_ref("actorTrainOpt_")[0]

        stateDict = {agentState_: [states[i] for states in allAgentsStateBatch] for i, agentState_ in enumerate(allAgentsStates_)}
        actionDict = {agentAction_: [actions[i] for actions in allAgentsActionsBatch] for i, agentAction_ in enumerate(allAgentsActions_)}
        valueDict = {learningRate_: self.actorLearningRate}

        actorTrainOpt = agentModel.run(actorTrainOpt_, feed_dict={**stateDict, **actionDict, **valueDict} )

        return agentModel


class TrainActor:
    def __init__(self, trainActorFromSA):
        self.trainActorFromSA = trainActorFromSA

    def __call__(self, agentID, allAgentsModels, miniBatch):
        allAgentsStateBatch, allAgentsActionsBatch, allAgentsRewardBatch, allAgentsNextStatesBatch = list(zip(*miniBatch))
        agentModel = self.trainActorFromSA(agentID, allAgentsModels, allAgentsStateBatch, allAgentsActionsBatch)

        return agentModel


# class TrainMADDPGModels:
#     def __init__(self, updateParameters, trainActor, trainCritic, allModels):
#         self.updateParameters = updateParameters
#         self.trainActor = trainActor
#         self.trainCritic = trainCritic
#         self.allModels = allModels
#
#     def __call__(self, miniBatch):
#         numAgents = len(self.allModels)
#         for agentID in range(numAgents):
#             agentModel = self.trainCritic(agentID, self.allModels, miniBatch)
#             agentModel = self.trainActor(agentID, agentModel, miniBatch)
#             agentModel = self.updateParameters(agentModel)
#             self.allModels[agentID] = agentModel
#
#     def getTrainedModels(self):
#         return self.allModels


class TrainMADDPGModelsWithBuffer:
    def __init__(self, updateParameters, trainActor, trainCritic, sampleFromBuffer, startLearn, allModels):
        self.updateParameters = updateParameters
        self.trainActor = trainActor
        self.trainCritic = trainCritic
        self.sampleFromBuffer = sampleFromBuffer
        self.startLearn = startLearn
        self.allModels = allModels

    def __call__(self, buffer, runTime):
        if not self.startLearn(runTime):
            return

        numAgents = len(self.allModels)
        for agentID in range(numAgents):
            miniBatch = self.sampleFromBuffer(buffer)
            agentModel = self.trainCritic(agentID, self.allModels, miniBatch)
            agentModel = self.trainActor(agentID, agentModel, miniBatch)
            agentModel = self.updateParameters(agentModel)
            self.allModels[agentID] = agentModel

    def getTrainedModels(self):
        return self.allModels