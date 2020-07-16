import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import tensorflow as tf
import numpy as np
import random
from collections import  deque
import gym
from functionTools.loadSaveModel import saveVariables
import matplotlib.pyplot as plt

class GetActorNetwork:
    def __init__(self, hyperparamDict, batchNorm = False):
        self.actorWeightInit = hyperparamDict['actorWeightInit']
        # self.actorBiasInit = hyperparamDict['actorBiasInit']
        self.actorActivFunctionList = hyperparamDict['actorActivFunction']
        self.actorLayersWidths = hyperparamDict['actorLayersWidths']
        self.batchNorm = batchNorm

    def __call__(self, stateDim, actionDim, actionBound, scope):
        with tf.variable_scope(scope):
            inputs = tf.placeholder(tf.float32,shape=(None, stateDim))
            layerNum = len(self.actorLayersWidths)
            net = inputs
            if self.batchNorm:
                for i in range(layerNum):
                    layerWidth = self.actorLayersWidths[i]
                    activFunction = self.actorActivFunctionList[i]
                    net = tf.layers.dense(net, layerWidth)
                    net = tf.layers.batch_normalization(net)
                    net = activFunction(net)
            else:
                for i in range(layerNum):
                    layerWidth = self.actorLayersWidths[i]
                    activFunction = self.actorActivFunctionList[i]
                    net = tf.layers.dense(net, layerWidth, activation = activFunction, kernel_initializer = self.actorWeightInit)

            out = tf.layers.dense(net, actionDim, activation = self.actorActivFunctionList[-1], kernel_initializer = self.actorWeightInit )
            scaled_out = tf.multiply(out, actionBound)

        return inputs, out, scaled_out


class Actor(object):
    def __init__(self, getActorNetwork, numStateSpace, actionDim, session, hyperparamDict, agentID = None, actionRange = 1):
        self.getActorNetwork = getActorNetwork
        self.numStateSpace = numStateSpace
        self.actionDim = actionDim
        self.actionRange = actionRange

        self.actorLR = hyperparamDict['actorLR']
        self.tau = hyperparamDict['tau']
        self.gamma = hyperparamDict['gamma']
        self.gradNormClipValue = hyperparamDict['gradNormClipValue']

        self.session = session
        self.scope = 'Agent'+ str(agentID) if agentID is not None else ''

        with tf.variable_scope(self.scope):
            actorTrainScope = 'actorTrain'
            actorTargetScope = 'actorTarget'
            self.states_, self.trainNetOut_, self.trainAction_ = self.getActorNetwork(numStateSpace, actionDim, actionRange, actorTrainScope)
            self.nextStates_, self.targetNetOut_, self.targetAction_ = self.getActorNetwork(numStateSpace, actionDim, actionRange, actorTargetScope)

            with tf.variable_scope("updateParameters"):
                actorTrainParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope= self.scope  + actorTrainScope)
                actorTargetParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope= self.scope  + actorTargetScope)
                self.actorUpdateParam_ = [actorTargetParams_[i].assign((1 - self.tau) * actorTargetParams_[i] + self.tau * actorTrainParams_[i]) for i in range(len(actorTargetParams_))]
                self.hardReplaceActorTargetParam_ = [tf.assign(trainParam, targetParam) for trainParam, targetParam in zip(actorTrainParams_, actorTargetParams_)]

            with tf.variable_scope("trainActorNet"):
                self.actionGradients_ = tf.placeholder(tf.float32, [None, actionDim])
                self.policyGradientRaw_ = tf.gradients(ys=self.trainAction_, xs=actorTrainParams_, grad_ys= self.actionGradients_)
                self.policyGradient_ = self.policyGradientRaw_ if self.gradNormClipValue is None else [tf.clip_by_norm(grad, self.gradNormClipValue) for grad in self.policyGradientRaw_]
                self.actorOptimizer = tf.train.AdamOptimizer(-self.actorLR, name='actorOptimizer')
                self.actorTrainOpt_ = self.actorOptimizer.apply_gradients(zip(self.policyGradient_, actorTrainParams_))

                # self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.clipped_gradients))

    def train(self, stateBatch, actionGradients):
        self.session.run(self.actorTrainOpt_, feed_dict = {self.states_: stateBatch, self.actionGradients_: actionGradients})
        
    def actByTrain(self, stateBatch):
        action = self.session.run(self.trainAction_, feed_dict = {self.states_: stateBatch})
        return action

    def actByTarget(self, nextStateBatch):
        action = self.session.run(self.targetAction_, feed_dict = {self.nextStates_: nextStateBatch})
        return action

    def updateParam(self):
        self.session.run(self.actorUpdateParam_)

class GetCriticNetwork:
    def __init__(self, hyperparamDict, addActionToLastLayer = False, batchNorm = False):
        self.criticWeightInit = hyperparamDict['criticWeightInit']
        # self.criticBiasInit = hyperparamDict['criticBiasInit']
        self.criticActivFunctionList = hyperparamDict['criticActivFunction']
        self.criticLayersWidths = hyperparamDict['criticLayersWidths']
        self.batchNorm = batchNorm
        self.addActionToLastLayer  = addActionToLastLayer

    def __call__(self, stateDim, actionDim, scope):
        with tf.variable_scope(scope):
            statesInput_ = tf.placeholder(tf.float32, shape=(None, stateDim))
            actionsInput_ = tf.placeholder(tf.float32, shape = (None, actionDim))
            layerNum = len(self.criticActivFunctionList)
            if self.addActionToLastLayer:
                net = statesInput_
                if self.batchNorm:
                    for i in range(layerNum - 1):
                        layerWidth = self.criticLayersWidths[i]
                        activFunction = self.criticActivFunctionList[i]
                        net = tf.layers.dense(net, layerWidth)
                        net = tf.layers.batch_normalization(net)
                        net = activFunction(net)

                    net = tf.layers.dense(net, self.criticLayersWidths[-1])
                    actorLastLayerActiv = tf.layers.dense(actionsInput_, self.criticLayersWidths[-1])
                    lastLayerActiv = tf.add(net, actorLastLayerActiv)
                    lastLayerActiv = tf.layers.batch_normalization(lastLayerActiv)
                    net = self.criticActivFunctionList[-1](lastLayerActiv)
                    out = tf.layers.dense(net,1, kernel_initializer=self.criticWeightInit)

                else:
                    for i in range(layerNum - 1):
                        layerWidth = self.criticLayersWidths[i]
                        activFunction = self.criticActivFunctionList[i]
                        net = tf.layers.dense(net, layerWidth, activation=activFunction, kernel_initializer=self.criticWeightInit)
                    lastLayerWidth = self.criticLayersWidths[-1]
                    lastLayerActivFunc = self.criticActivFunctionList[-1]
                    net = tf.layers.dense(net, lastLayerWidth)
                    actorLastLayerActiv = tf.layers.dense(actionsInput_, lastLayerWidth)
                    lastLayerActiv = tf.add(net, actorLastLayerActiv)
                    lastLayerActiv = tf.layers.dense(lastLayerActiv, lastLayerWidth, activation=lastLayerActivFunc, kernel_initializer=self.criticWeightInit)
                    net = self.criticActivFunctionList[-1](lastLayerActiv)
                    out = tf.layers.dense(net,1, kernel_initializer=self.criticWeightInit)
            else:
                net = tf.concat([statesInput_, actionsInput_], axis=1)
                for i in range(layerNum):
                    layerWidth = self.criticLayersWidths[i]
                    activFunction = self.criticActivFunctionList[i]
                    net = tf.layers.dense(net, layerWidth, activation=activFunction, kernel_initializer=self.criticWeightInit)
                out = tf.layers.dense(net, 1, kernel_initializer=self.criticWeightInit)

        return statesInput_, actionsInput_, out


class Critic(object):
    def __init__(self, getCriticNetwork, numStateSpace, actionDim, session, hyperparamDict, agentID=None):
        self.getCriticNetwork = getCriticNetwork
        self.numStateSpace = numStateSpace
        self.actionDim = actionDim

        self.criticLR = hyperparamDict['criticLR']
        self.tau = hyperparamDict['tau']
        self.gamma = hyperparamDict['gamma']

        self.session = session
        self.scope = 'Agent' + str(agentID) if agentID is not None else ''

        with tf.variable_scope(self.scope):
            criticTrainScope = 'criticTrain'
            criticTargetScope = 'criticTarget'
            self.states_, self.actions_, self.trainValue_ = self.getCriticNetwork(numStateSpace, actionDim, criticTrainScope)
            self.nextStates_, self.targetActions_, self.targetValue_ = self.getCriticNetwork(numStateSpace, actionDim, criticTargetScope)

            with tf.variable_scope("updateParameters"):
                criticTrainParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope  + criticTrainScope)
                criticTargetParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope  + criticTargetScope)
                self.criticUpdateParam_ = [criticTargetParams_[i].assign(
                    (1 - self.tau) * criticTargetParams_[i] + self.tau * criticTrainParams_[i]) for i in range(len(criticTargetParams_))]
                self.hardReplaceCriticTargetParam_ = [tf.assign(trainParam, targetParam) for trainParam, targetParam in
                                                     zip(criticTrainParams_, criticTargetParams_)]

            with tf.variable_scope("trainCriticNet"):
                self.reward_ = tf.placeholder(tf.float32, [None, 1])
                self.qtarget_ = tf.placeholder(tf.float32, [None, 1])
                self.qUpdateTarget = self.reward_ + self.gamma * self.qtarget_
                self.criticLoss_ = tf.losses.mean_squared_error(self.qUpdateTarget, self.trainValue_)
                self.criticOptimizer = tf.train.AdamOptimizer(self.criticLR, name='criticOptimizer')
                self.criticTrainOpt_ = self.criticOptimizer.minimize(self.criticLoss_, var_list=criticTrainParams_)
                
            with tf.variable_scope("gradients"):
                self.actionGradients_ = tf.gradients(self.trainValue_, self.actions_)[0]


    def train(self, stateBatch, actionBatch, rewardBatch, qTargetBatch):
        self.session.run(self.criticTrainOpt_,
                         feed_dict={self.states_: stateBatch, self.actions_: actionBatch, self.reward_: rewardBatch, self.qtarget_: qTargetBatch})

    def getTrainNetValue(self, stateBatch, actionBatch):
        value = self.session.run(self.trainValue_, feed_dict={self.states_: stateBatch, self.actions_: actionBatch})
        return value
    
    def getTargetNetValue(self, nextStateBatch, actionBatch):
        value = self.session.run(self.targetValue_, feed_dict={self.nextStates_: nextStateBatch, self.targetActions_: actionBatch})
        return value

    def getActionGradients(self, stateBatch, actionBatch):
        actionGradients = self.session.run(self.actionGradients_, feed_dict={self.states_: stateBatch, self.actions_: actionBatch})
        return actionGradients

    def updateParam(self):
        self.session.run(self.criticUpdateParam_)




def reshapeBatchToGetSASR(miniBatch):
    states, actions, rewards, nextStates = list(zip(*miniBatch))
    stateBatch = np.asarray(states).reshape(len(miniBatch), -1)
    actionBatch = np.asarray(actions).reshape(len(miniBatch), -1)
    nextStateBatch = np.asarray(nextStates).reshape(len(miniBatch), -1)
    rewardBatch = np.asarray(rewards).reshape(len(miniBatch), -1)

    return stateBatch, actionBatch, nextStateBatch, rewardBatch



class TrainDDPGModelsOneStep:
    def __init__(self, reshapeBatchToGetSASR, actor, critic):
        self.reshapeBatchToGetSASR = reshapeBatchToGetSASR
        self.actor = actor
        self.critic = critic

    def __call__(self, miniBatch):
        stateBatch, actionBatch, nextStateBatch, rewardBatch = self.reshapeBatchToGetSASR(miniBatch)

        targetActionBatch = self.actor.actByTarget(nextStateBatch)
        targetQValue = self.critic.getTargetNetValue(nextStateBatch, targetActionBatch)
        self.critic.train(stateBatch, actionBatch, rewardBatch, targetQValue)

        trainActionBatch = self.actor.actByTrain(stateBatch)
        actionGradients = self.critic.getActionGradients(stateBatch, trainActionBatch)
        self.actor.train(stateBatch, actionGradients)

        self.critic.updateParam()
        self.actor.updateParam()


class LearnFromBuffer:
    def __init__(self, learningStartBufferSize, trainModels, learnInterval = 1):
        self.learningStartBufferSize = learningStartBufferSize
        self.trainModels = trainModels
        self.learnInterval = learnInterval

    def __call__(self, miniBatch, runTime):
        if runTime >= self.learningStartBufferSize and runTime % self.learnInterval == 0:
            self.trainModels(miniBatch)


class MemoryBuffer(object):
    def __init__(self, size, minibatchSize):
        self.size = size
        self.buffer = self.reset()
        self.minibatchSize = minibatchSize

    def reset(self):
        return deque(maxlen=int(self.size))

    def add(self, observation, action, reward, nextObservation):
        self.buffer.append((observation, action, reward, nextObservation))

    def sample(self):
        if len(self.buffer) < self.minibatchSize:
            return []
        sampleIndex = [random.randint(0, len(self.buffer) - 1) for _ in range(self.minibatchSize)]
        sample = [self.buffer[index] for index in sampleIndex]

        return sample

class OrnsteinUhlenbeckActionNoise(object):
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def getNoise(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

class ActOneStep:
    def __init__(self, actor, noise):
        self.actor = actor
        self.noise = noise

    def __call__(self, state, episodeID):
        action = self.actor.actByTrain(state)
        noise = self.noise.getNoise()
        action += noise
        return action

class ActOneStepForInvPendulum:
    def __init__(self, actor, maxEpisode, action_space):
        self.actor = actor
        self.maxEpisode = maxEpisode
        self.action_space = action_space

    def __call__(self, state, episodeID, noise):
        if episodeID < self.maxEpisode or ((episodeID % 9 < 5) and episodeID < 15* self.maxEpisode):
            action = self.action_space.sample()
        else:
            action = self.actor.actByTrain(state)
            noise = noise.getNoise()
            action += noise
        return action

class TrainDDPGWithGym:
    def __init__(self, maxEpisode, maxTimeStep, memoryBuffer, noise, actOneStep, learnFromBuffer, env, saveModel):
        self.maxEpisode = maxEpisode
        self.maxTimeStep = maxTimeStep
        self.memoryBuffer = memoryBuffer
        self.noise = noise
        self.actOneStep = actOneStep
        self.env = env
        self.learnFromBuffer = learnFromBuffer
        self.saveModel = saveModel

    def __call__(self):
        episodeRewardList = []
        for episodeID in range(self.maxEpisode):
            state = self.env.reset()
            epsReward = 0
            self.noise.reset()

            for timeStep in range(self.maxTimeStep):
                action = self.actOneStep(state, episodeID, self.noise)
                nextState, reward, terminal, info = self.env.step(action)
                self.memoryBuffer.add(state, action, reward, nextState)
                epsReward += reward

                miniBatch= self.memoryBuffer.sample()
                runTime = len(self.memoryBuffer.buffer)
                self.learnFromBuffer(miniBatch, runTime)
                state = nextState

                if terminal:
                    break
            self.saveModel()
            episodeRewardList.append(epsReward)
            last100EpsMeanReward = np.mean(episodeRewardList[-1000: ])
            if episodeID % 5000 == 0:
                print('episode: {}, last 1000eps mean reward: {}, last eps reward: {} with {} steps'.format(episodeID, last100EpsMeanReward, epsReward, timeStep))

        return episodeRewardList

class SaveModel:
    def __init__(self, modelSaveRate, saveVariables, modelSavePath, sess, saveAllmodels = False):
        self.modelSaveRate = modelSaveRate
        self.saveVariables = saveVariables
        self.epsNum = 1
        self.modelSavePath = modelSavePath
        self.saveAllmodels = saveAllmodels
        self.sess = sess

    def __call__(self):
        self.epsNum += 1
        if self.epsNum % self.modelSaveRate == 0:
            modelSavePathToUse = self.modelSavePath + str(self.epsNum) + "eps" if self.saveAllmodels else self.modelSavePath
            with self.sess.as_default():
                self.saveVariables(self.sess, modelSavePathToUse)

def main():
    env_name = 'InvertedDoublePendulum-v2'
    env = gym.make(env_name)

    hyperparamDict = dict()
    hyperparamDict['actorWeightInit'] = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
    hyperparamDict['actorActivFunction'] = [tf.nn.relu, tf.nn.relu, tf.nn.tanh]
    hyperparamDict['actorLayersWidths'] = [400, 300]
    hyperparamDict['actorLR'] = 1e-4

    hyperparamDict['criticWeightInit'] = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
    hyperparamDict['criticActivFunction']= [tf.nn.relu, tf.nn.relu]
    hyperparamDict['criticLayersWidths'] = [400, 300]
    hyperparamDict['criticLR'] = 1e-3

    hyperparamDict['tau'] = 0.001
    hyperparamDict['gamma'] = 0.99
    hyperparamDict['gradNormClipValue'] = 5

    maxEpisode = 5000* 1000
    maxTimeStep = 100
    bufferSize = 1e5
    minibatchSize = 128

    session = tf.Session()

    stateDim = env.observation_space.shape[0] #11
    actionDim = env.action_space.shape[0] #1
    getActorNetwork = GetActorNetwork(hyperparamDict, batchNorm= True)
    actor = Actor(getActorNetwork, stateDim, actionDim, session, hyperparamDict, actionRange= 3)

    getCriticNetwork = GetCriticNetwork(hyperparamDict, addActionToLastLayer = True, batchNorm = True)
    critic = Critic(getCriticNetwork, stateDim, actionDim, session, hyperparamDict)

    saver = tf.train.Saver(max_to_keep=None)
    tf.add_to_collection("saver", saver)
    session.run(tf.global_variables_initializer())

    fileName = 'ddpg_mujoco_invDblPendulum'
    modelPath = os.path.join(dirName, '..', 'trainedModels', fileName)
    modelSaveRate = 500
    saveModel = SaveModel(modelSaveRate, saveVariables, modelPath, session)

    trainDDPGOneStep = TrainDDPGModelsOneStep(reshapeBatchToGetSASR, actor, critic)

    learningStartBufferSize = minibatchSize
    learnFromBuffer = LearnFromBuffer(learningStartBufferSize, trainDDPGOneStep, learnInterval = 1)

    buffer = MemoryBuffer(bufferSize, minibatchSize)

    noiseMu = np.zeros((actionDim, 1))
    noiseSigma = 0.05
    noise = OrnsteinUhlenbeckActionNoise(noiseMu, noiseSigma)

    actOneStep = ActOneStepForInvPendulum(actor, maxEpisode, env.action_space)
    ddpg = TrainDDPGWithGym(maxEpisode, maxTimeStep, buffer, noise, actOneStep, learnFromBuffer, env, saveModel)

    episodeRewardList = ddpg()
    plt.plot(range(len(episodeRewardList)), episodeRewardList)


if __name__ == '__main__':
    main()
