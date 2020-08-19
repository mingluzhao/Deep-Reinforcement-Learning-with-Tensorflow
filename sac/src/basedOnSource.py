import tensorflow as tf
import numpy as np
import random
from collections import deque


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


class BuildValueNet:
    def __init__(self, hyperparamDict):
        self.valueNetWeightInit = hyperparamDict['valueNetWeightInit']
        self.valueNetBiasInit = hyperparamDict['valueNetBiasInit']
        self.valueNetActivFunctionList = hyperparamDict['valueNetActivFunction']
        self.valueNetLayersWidths = hyperparamDict['valueNetLayersWidths']

    def __call__(self, statesInput_, scope):
        with tf.variable_scope(scope):
            layerNum = len(self.valueNetLayersWidths)
            net = statesInput_
            for i in range(layerNum):
                layerWidth = self.valueNetLayersWidths[i]
                activFunction = self.valueNetActivFunctionList[i]
                net = tf.layers.dense(net, layerWidth, activation=activFunction, kernel_initializer=self.valueNetWeightInit, bias_initializer=self.valueNetBiasInit)
            out = tf.layers.dense(net, 1, kernel_initializer=self.valueNetWeightInit, bias_initializer=self.valueNetBiasInit)

        return out


class BuildQNet:
    def __init__(self, hyperparamDict):
        self.qNetWeightInit = hyperparamDict['qNetWeightInit']
        self.qNetBiasInit = hyperparamDict['qNetBiasInit']
        self.qNetActivFunctionList = hyperparamDict['qNetActivFunction']
        self.qNetLayersWidths = hyperparamDict['qNetLayersWidths']

    def __call__(self, statesInput_, actionsInput_, scope):
        with tf.variable_scope(scope):
            layerNum = len(self.qNetLayersWidths)
            net = tf.concat([statesInput_, actionsInput_], axis=1)
            for i in range(layerNum):
                layerWidth = self.qNetLayersWidths[i]
                activFunction = self.qNetActivFunctionList[i]
                net = tf.layers.dense(net, layerWidth, activation=activFunction, kernel_initializer=self.qNetWeightInit,
                                      bias_initializer=self.qNetBiasInit)
            out = tf.layers.dense(net, 1, kernel_initializer=self.qNetWeightInit, bias_initializer=self.qNetBiasInit)

        return out


# TODO: no initialization for fc layers (only for the last layer)


class BuildPolicyNet:
    def __init__(self, hyperparamDict):
        self.policyWeightInit = hyperparamDict['policyWeightInit']
        self.policyBiasInit = hyperparamDict['policyBiasInit']
        self.policyActivFunctionList = hyperparamDict['policyActivFunction']
        self.policyLayersWidths = hyperparamDict['policyLayersWidths']
        self.policyMuWeightInit = hyperparamDict['policyMuWeightInit']
        self.policySDWeightInit = hyperparamDict['policySDWeightInit']

        self.policySDlow = hyperparamDict['policySDlow']
        self.policySDhigh = hyperparamDict['policySDhigh']

    def __call__(self, stateInput_, stateDim, actionDim, actionBound, scope):
        with tf.variable_scope(scope):
            layerNum = len(self.policyLayersWidths)
            net = stateInput_
            for i in range(layerNum):
                layerWidth = self.policyLayersWidths[i]
                activFunction = self.policyActivFunctionList[i]
                net = tf.layers.dense(net, layerWidth, activation = activFunction, kernel_initializer = self.policyWeightInit, bias_initializer= self.policyBiasInit)

            mu_ = tf.layers.dense(net, actionDim, kernel_initializer=self.policyMuWeightInit, name='mu_')
            logSigmaRaw_ = tf.layers.dense(net, actionDim, kernel_initializer=self.policySDWeightInit, name='sigma_')
            logSigma_ = tf.clip_by_value(logSigmaRaw_, self.policySDlow, self.policySDhigh)

        return mu_, logSigma_