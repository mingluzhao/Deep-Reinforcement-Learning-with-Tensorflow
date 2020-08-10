import tensorflow as tf
import numpy as np
import random
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from collections import  deque
from functionTools.loadSaveModel import saveVariables
import matplotlib.pyplot as plt
import gym
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

class LearnFromBuffer:
    def __init__(self, learningStartBufferSize, trainModels, learnInterval = 1):
        self.learningStartBufferSize = learningStartBufferSize
        self.trainModels = trainModels
        self.learnInterval = learnInterval

    def __call__(self, miniBatch, runTime):
        if runTime >= self.learningStartBufferSize and runTime % self.learnInterval == 0:
            self.trainModels(miniBatch)

class ToOneHotAction:
    def __init__(self, actionSpace):
        self.actionSpace = actionSpace

    def __call__(self, actionID):
        actionNum = len(self.actionSpace)
        actions = np.zeros(actionNum)
        actions[actionID] = 1
        return actions


class MemoryBuffer(object):
    def __init__(self, size, minibatchSize, toOneHotAction = None):
        self.size = size
        self.buffer = self.reset()
        self.minibatchSize = minibatchSize
        self.toOneHotAction = toOneHotAction

    def reset(self):
        return deque(maxlen=int(self.size))

    def add(self, observation, action, reward, nextObservation, done):
        if self.toOneHotAction is not None:
            action = self.toOneHotAction(action)
        self.buffer.append((observation, action, reward, nextObservation, done))

    def sample(self):
        if len(self.buffer) < self.minibatchSize:
            return []
        sampleIndex = [random.randint(0, len(self.buffer) - 1) for _ in range(self.minibatchSize)]
        sample = [self.buffer[index] for index in sampleIndex]

        return sample

class BuildDQNNet:
    def __init__(self, hyperparamDict, stateDim, actionSpace):
        self.hyperparamDict = hyperparamDict
        self.stateDim = stateDim
        self.actionDim = len(actionSpace)
        self.weightInit = hyperparamDict['weightInit']
        self.biasInit = hyperparamDict['biasInit']
        self.layersWidths = hyperparamDict['layersWidths']
        self.activFunctionList = hyperparamDict['activFunctionList']


    def __call__(self, states_, scope):
        with tf.variable_scope(scope):
            layerNum = len(self.layersWidths)
            net = states_
            for i in range(layerNum):
                layerWidth = self.layersWidths[i]
                activFunction = self.activFunctionList[i]
                net = tf.layers.dense(net, layerWidth, activation=activFunction, kernel_initializer=self.weightInit, bias_initializer=self.biasInit)
            out = tf.layers.dense(net, self.actionDim, kernel_initializer=self.weightInit, bias_initializer=self.biasInit)

        return out

class DQNAgent(object):
    def __init__(self, stateDim, actionSpace, sess, hyperparamDict, buildDQNNet):
        self.stateDim = stateDim
        self.actionSpace = actionSpace
        self.actionDim = len(actionSpace)

        self.buildDQNNet = buildDQNNet
        self.sess = sess
        self.lr = hyperparamDict['lr']
        self.gamma = hyperparamDict['gamma']
        self.minibatchSize = hyperparamDict['minibatchSize']

        self.states_ = tf.placeholder(tf.float32, [None, self.stateDim], name='states_')
        self.nextStates_ = tf.placeholder(tf.float32, [None, self.stateDim], name='nextStates_')
        self.actions_ = tf.placeholder(tf.float32, [None, self.actionDim], name='actions_')

        self.qTrainOut = self.buildDQNNet(self.states_, scope = 'qTrainNet')
        self.qTargetOut = self.buildDQNNet(self.nextStates_, scope = 'qTargetNet')

        with tf.name_scope("updateParameters"):
            trainParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='qTrainNet')
            targetParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='qTargetNet')
            self.updateParam_ = [tf.assign(trainParam, targetParam) for trainParam, targetParam in zip(trainParams_, targetParams_)]

        with tf.name_scope("train"):
            self.qTargetVal_ = tf.placeholder(tf.float32, [None, self.actionDim], name='states_')
            self.qLoss_ = tf.losses.mean_squared_error(self.qTrainOut, self.qTargetVal_)
            self.trainOpt_ = tf.train.AdamOptimizer(self.lr).minimize(self.qLoss_, var_list=trainParams_)

    def updateParam(self):
        self.sess.run(self.updateParam_)

    def train(self, stateBatch, actionBatch, rewardBatch, doneBatch, currentValueTrain, nextValueTarget):
        # batch actions (one hot batch)
        actionIndices = np.dot(np.array(actionBatch, dtype = np.int8), self.actionSpace)
        batchIndices = np.arange(self.minibatchSize, dtype=np.int32)

        # put the max next q values at index of batch actions
        qNextStateTarget = currentValueTrain.copy()
        selectedValuetarget = np.max(nextValueTarget, axis =1)
        qTargetWithMaxVal = rewardBatch+ (1- doneBatch) * self.gamma * selectedValuetarget.reshape(self.minibatchSize, -1)

        qNextStateTarget[batchIndices, actionIndices] = qTargetWithMaxVal.reshape(-1) # target of update differs with train only at batch actions
        self.sess.run(self.trainOpt_, feed_dict={self.states_: stateBatch, self.actions_: actionBatch, self.qTargetVal_: qNextStateTarget})

    def actGreedy(self, stateBatch):
        qTrain = self.sess.run(self.qTrainOut, feed_dict = {self.states_: stateBatch})
        action = np.argmax(qTrain)
        return action

    def actRandom(self):
        action = np.random.randint(0, self.actionDim)
        return action

    def getQValueTarget(self, nextStateBatch):
        value = self.sess.run(self.qTargetOut, feed_dict = {self.nextStates_: nextStateBatch})
        return value

    def getQValueTrain(self, stateBatch):
        value = self.sess.run(self.qTrainOut, feed_dict = {self.states_: stateBatch})
        return value

class GetEpsilon:
    def __init__(self, initEpsilon, epsilonDecay, epsilonMin):
        self.initEpsilon = initEpsilon
        self.epsilonDecay = epsilonDecay
        self.epsilonMin = epsilonMin
        self.runTime = 0

    def __call__(self):
        epsilon = self.initEpsilon* self.epsilonDecay ** self.runTime
        epsilon = max(epsilon, self.epsilonMin)
        self.runTime += 1
        return epsilon



class TrainDQNOneStep:
    def __init__(self, dqn, reshapeBatchToGetSASR, hyperparamDict):
        self.dqn = dqn
        self.reshapeBatchToGetSASR = reshapeBatchToGetSASR
        self.updateInterval = hyperparamDict['updateInterval']
        self.learnTime = 0

    def __call__(self, miniBatch):
        stateBatch, actionBatch, nextStateBatch, rewardBatch, doneBatch = self.reshapeBatchToGetSASR(miniBatch)
        nextValueTarget = self.dqn.getQValueTarget(nextStateBatch)
        currentValueTrain = self.dqn.getQValueTrain(stateBatch)

        self.dqn.train(stateBatch, actionBatch, rewardBatch, doneBatch, currentValueTrain, nextValueTarget)

        if self.learnTime % self.updateInterval == 0:
            self.dqn.updateParam()
        self.learnTime += 1

class ActOneStep:
    def __init__(self, dqn, actionSpace, getEpsilon):
        self.dqn = dqn
        self.actionSpace = actionSpace
        self.getEpsilon = getEpsilon

    def __call__(self, state):
        epsilon = self.getEpsilon()
        stateBatch = np.asarray(state).reshape(1, -1)
        if np.random.uniform() < epsilon:  # prob of epsilon for choosing random action
            actionID = self.dqn.actRandom()
        else:
            actionID = self.dqn.actGreedy(stateBatch)

        action = self.actionSpace[actionID]
        return action

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


class TrainDQN:
    def __init__(self, maxEpisode, maxTimeStep, memoryBuffer, actOneStep, learnFromBuffer, env, saveModel):
        self.maxEpisode = maxEpisode
        self.maxTimeStep = maxTimeStep
        self.memoryBuffer = memoryBuffer
        self.actOneStep = actOneStep
        self.env = env
        self.learnFromBuffer = learnFromBuffer
        self.saveModel = saveModel

    def __call__(self):
        episodeRewardList = []
        for episodeID in range(self.maxEpisode):
            state = self.env.reset()
            state = state.reshape(1, -1)
            epsReward = 0

            for timeStep in range(self.maxTimeStep):
                action = self.actOneStep(state)
                nextState, reward, terminal, info = self.env.step(action)
                nextState = nextState.reshape(1, -1)
                self.memoryBuffer.add(state, action, reward, nextState, terminal)
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
            if episodeID % 1 == 0:
                print('episode: {}, last 1000eps mean reward: {}, last eps reward: {} with {} steps'.format(episodeID, last100EpsMeanReward, epsReward, timeStep))

        return episodeRewardList

def reshapeBatchToGetSASR(miniBatch):
    states, actions, rewards, nextStates, done = list(zip(*miniBatch))
    stateBatch = np.asarray(states).reshape(len(miniBatch), -1)
    actionBatch = np.asarray(actions).reshape(len(miniBatch), -1)
    nextStateBatch = np.asarray(nextStates).reshape(len(miniBatch), -1)
    rewardBatch = np.asarray(rewards).reshape(len(miniBatch), -1)
    doneBatch = np.asarray(done).reshape(len(miniBatch), -1)

    return stateBatch, actionBatch, nextStateBatch, rewardBatch, doneBatch


def main():
    hyperparamDict = dict()
    hyperparamDict['weightInit'] = tf.random_normal_initializer(0., 0.3)
    hyperparamDict['biasInit'] = tf.constant_initializer(0.1)

    hyperparamDict['layersWidths'] = [20]
    hyperparamDict['activFunctionList'] = [tf.nn.relu]
    hyperparamDict['lr'] = 0.005
    hyperparamDict['gamma'] = 0.9
    hyperparamDict['updateInterval'] = 200
    hyperparamDict['minibatchSize'] = 32

    bufferSize = 3000
    maxEpisode = 100
    maxTimeStep = 200

    env = gym.make("CartPole-v0")
    actionSpace = list(range(env.action_space.n))
    stateDim = env.observation_space.shape[0]

    sess = tf.Session()
    buildDQNNet = BuildDQNNet(hyperparamDict, stateDim, actionSpace)
    dqn = DQNAgent(stateDim, actionSpace, sess, hyperparamDict, buildDQNNet)
    trainDQN = TrainDQNOneStep(dqn, reshapeBatchToGetSASR, hyperparamDict)

    initEpsilon = 1
    epsilonDecay = 0.97
    epsilonMin = 0.1
    getEpsilon = GetEpsilon(initEpsilon, epsilonDecay, epsilonMin)
    actOneStep = ActOneStep(dqn, actionSpace, getEpsilon)
    learningStartBufferSize = hyperparamDict['minibatchSize']
    learnFromBuffer = LearnFromBuffer(learningStartBufferSize, trainDQN)
    toOneHotAction = ToOneHotAction(actionSpace)
    buffer = MemoryBuffer(bufferSize, hyperparamDict['minibatchSize'], toOneHotAction)

    saver = tf.train.Saver(max_to_keep=None)
    tf.add_to_collection("saver", saver)
    writer = tf.summary.FileWriter('tensorBoard/', graph=sess.graph)
    tf.add_to_collection("writer", writer)
    sess.run(tf.global_variables_initializer())

    fileName = 'dqnCartPole'
    modelPath = os.path.join(dirName, '..', 'trainedModels', fileName)
    modelSaveRate = 50
    saveModel = SaveModel(modelSaveRate, saveVariables, modelPath, sess, saveAllmodels=False)

    trainDQN = TrainDQN(maxEpisode, maxTimeStep, buffer, actOneStep, learnFromBuffer, env, saveModel)

    episodeRewardList = trainDQN()
    imageSavePath = os.path.join(dirName, '..', 'plots')
    if not os.path.exists(imageSavePath):
        os.makedirs(imageSavePath)
    plt.plot(range(len(episodeRewardList)), episodeRewardList)
    plt.savefig(os.path.join(imageSavePath, fileName + str('.png')))


if __name__ == '__main__':
    main()