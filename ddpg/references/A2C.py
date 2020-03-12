import tensorflow as tf
import numpy as np
import functools as ft
import env
import reward
import tensorflow_probability as tfp
import random
import agentsEnv as ag
import itertools as it
import pygame as pg


# return action
class ApproximatePolicy():
    def __init__(self, actionSpace):
        self.actionSpace = actionSpace
        self.numActionSpace = len(self.actionSpace)

    def __call__(self, stateBatch, model):
        graph = model.graph
        state_ = graph.get_tensor_by_name('inputs/state_:0')
        actionDistribution_ = graph.get_tensor_by_name('outputs/actionDistribution_:0')
        actionDistributionBatch = model.run(actionDistribution_, feed_dict={state_: stateBatch})
        actionIndexBatch = [np.random.choice(range(self.numActionSpace), p=actionDistribution) for actionDistribution in
                            actionDistributionBatch]
        actionBatch = np.array([self.actionSpace[actionIndex] for actionIndex in actionIndexBatch])
        return actionBatch


# returnTrajectory
class SampleTrajectory():
    def __init__(self, maxTimeStep, transitionFunction, isTerminal):
        self.maxTimeStep = maxTimeStep
        self.transitionFunction = transitionFunction
        self.isTerminal = isTerminal

    def __call__(self, actor):
        oldState, action = None, None
        oldState = self.transitionFunction(oldState, action)
        trajectory = []

        for time in range(self.maxTimeStep):
            oldStateBatch = oldState.reshape(1, -1)
            actionBatch = actor(oldStateBatch)
            action = actionBatch[0]
            # actionBatch shape: batch * action Dimension; only keep action Dimention in shape
            newState = self.transitionFunction(oldState, action)
            trajectory.append((oldState, action))
            terminal = self.isTerminal(oldState)
            if terminal:
                break
            oldState = newState
        return trajectory


# input: trajectory
# return accumulatedReturn
class AccumulateReward():
    def __init__(self, decay, rewardFunction):
        self.decay = decay
        self.rewardFunction = rewardFunction

    def __call__(self, trajectory):
        rewards = [self.rewardFunction(state, action) for state, action in trajectory]
        accumulateReward = lambda accumulatedReward, reward: self.decay * accumulatedReward + reward
        accumulatedRewards = np.array(
            [ft.reduce(accumulateReward, reversed(rewards[TimeT:])) for TimeT in range(len(rewards))])
        return accumulatedRewards


class TrainCriticMonteCarloTensorflow():
    def __init__(self, accumulateReward):
        self.accumulateReward = accumulateReward

    def __call__(self, episode, criticModel):
        mergedEpisode = np.concatenate(episode)
        numBatch = len(mergedEpisode)
        stateEpisode, actionEpisode = list(zip(*mergedEpisode))
        stateBatch = np.array(stateEpisode).reshape(numBatch, -1)

        mergedAccumulatedRewardsEpisode = np.concatenate([self.accumulateReward(trajectory) for trajectory in episode])
        valueTargetBatch = np.array(mergedAccumulatedRewardsEpisode).reshape(numBatch, -1)

        graph = criticModel.graph
        state_ = graph.get_tensor_by_name('inputs/state_:0')
        valueTarget_ = graph.get_tensor_by_name('inputs/valueTarget_:0')
        loss_ = graph.get_tensor_by_name('outputs/loss_:0')
        trainOpt_ = graph.get_operation_by_name('train/adamOpt_')
        loss, trainOpt = criticModel.run([loss_, trainOpt_], feed_dict={state_: stateBatch,
                                                                        valueTarget_: valueTargetBatch
                                                                        })
        return loss, criticModel


def approximateValue(stateBatch, criticModel):
    graph = criticModel.graph
    state_ = graph.get_tensor_by_name('inputs/state_:0')
    value_ = graph.get_tensor_by_name('outputs/value_/BiasAdd:0')
    valueBatch = criticModel.run(value_, feed_dict={state_: stateBatch})
    return valueBatch


class EstimateAdvantageMonteCarlo():
    def __init__(self, accumulateReward):
        self.accumulateReward = accumulateReward

    def __call__(self, episode, critic):
        mergedEpisode = np.concatenate(episode)
        numBatch = len(mergedEpisode)
        stateEpisode, actionEpisode = list(zip(*mergedEpisode))
        stateBatch, actionBatch = np.array(stateEpisode).reshape(numBatch, -1), np.array(actionEpisode).reshape(
            numBatch, -1)

        mergedAccumulatedRewardsEpisode = np.concatenate([self.accumulateReward(trajectory) for trajectory in episode])
        accumulatedRewardsBatch = np.array(mergedAccumulatedRewardsEpisode).reshape(numBatch, -1)

        advantageBatch = accumulatedRewardsBatch - critic(stateBatch)
        advantages = np.concatenate(advantageBatch)
        return advantages


class TrainActorMonteCarloTensorflow():
    def __init__(self, actionSpace):
        self.actionSpace = actionSpace
        self.numActionSpace = len(actionSpace)

    def __call__(self, episode, advantages, actorModel):
        mergedEpisode = np.concatenate(episode)
        numBatch = len(mergedEpisode)
        stateEpisode, actionEpisode = list(zip(*mergedEpisode))
        actionIndexEpisode = np.array([list(self.actionSpace).index(list(action)) for action in actionEpisode])
        actionLabelEpisode = np.zeros([numBatch, self.numActionSpace])
        actionLabelEpisode[np.arange(numBatch), actionIndexEpisode] = 1
        stateBatch, actionLabelBatch = np.array(stateEpisode).reshape(numBatch, -1), np.array(
            actionLabelEpisode).reshape(numBatch, -1)

        graph = actorModel.graph
        state_ = graph.get_tensor_by_name('inputs/state_:0') #_: tensor, no_: value
        actionLabel_ = graph.get_tensor_by_name('inputs/actionLabel_:0')
        advantages_ = graph.get_tensor_by_name('inputs/advantages_:0')
        loss_ = graph.get_tensor_by_name('outputs/loss_:0')
        trainOpt_ = graph.get_operation_by_name('train/adamOpt_')
        loss, trainOpt = actorModel.run([loss_, trainOpt_], feed_dict={state_: stateBatch,
                                                                       actionLabel_: actionLabelBatch,
                                                                       advantages_: advantages
                                                                       })
        return loss, actorModel


class OfflineAdvantageActorCritic():
    def __init__(self, numTrajectory, maxEpisode, render):
        self.numTrajectory = numTrajectory
        self.maxEpisode = maxEpisode
        self.render = render

    def __call__(self, actorModel, criticModel, approximatePolicy, sampleTrajectory, trainCritic, approximateValue,
                 estimateAdvantage, trainActor):
        for episodeIndex in range(self.maxEpisode):
            actor = lambda state: approximatePolicy(state, actorModel)
            episode = [sampleTrajectory(actor) for trajectoryIndex in range(self.numTrajectory)]
            valueLoss, criticModels = trainCritic(episode, criticModel)
            critic = lambda state: approximateValue(state, criticModel)
            advantages = estimateAdvantage(episode, critic)
            policyLoss, actorModel = trainActor(episode, advantages, actorModel)
            print(np.mean([len(trajectory) for trajectory in episode]))
            if episodeIndex % 1 == -1:
                for timeStep in episode[-1]:
                    self.render(timeStep[0])
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

    actorGraph = tf.Graph()
    with actorGraph.as_default():
        with tf.name_scope("inputs"):
            state_ = tf.placeholder(tf.float32, [None, numStateSpace], name="state_")
            actionLabel_ = tf.placeholder(tf.int32, [None, numActionSpace], name="actionLabel_")
            advantages_ = tf.placeholder(tf.float32, [None, ], name="advantages_")

        with tf.name_scope("hidden"):
            initWeight = tf.random_uniform_initializer(-0.03, 0.03)
            initBias = tf.constant_initializer(0.01)
            fullyConnected1_ = tf.layers.dense(inputs=state_, units=numActorFC1Unit, activation=tf.nn.relu,
                                               kernel_initializer=initWeight, bias_initializer=initBias)
            fullyConnected2_ = tf.layers.dense(inputs=fullyConnected1_, units=numActorFC2Unit, activation=tf.nn.relu,
                                               kernel_initializer=initWeight, bias_initializer=initBias)
            fullyConnected3_ = tf.layers.dense(inputs=fullyConnected2_, units=numActorFC2Unit, activation=tf.nn.relu,
                                               kernel_initializer=initWeight, bias_initializer=initBias)
            allActionActivation_ = tf.layers.dense(inputs=fullyConnected3_, units=numActionSpace, activation=None,
                                                   kernel_initializer=initWeight, bias_initializer=initBias)

        with tf.name_scope("outputs"):
            actionDistribution_ = tf.nn.softmax(allActionActivation_, name='actionDistribution_')
            actionEntropy_ = tf.multiply(tfp.distributions.Categorical(probs=actionDistribution_).entropy(), 1,
                                         name='actionEntropy_')
            negLogProb_ = tf.nn.softmax_cross_entropy_with_logits_v2(logits=allActionActivation_, labels=actionLabel_,
                                                                     name='negLogProb_')################
            loss_ = tf.reduce_mean(tf.multiply(negLogProb_, advantages_), name='loss_')
            actorLossSummary = tf.summary.scalar("ActorLoss", loss_)

        with tf.name_scope("train"):
            trainOpt_ = tf.train.AdamOptimizer(learningRateActor, name='adamOpt_').minimize(loss_)

        actorInit = tf.global_variables_initializer()

    actorModel = tf.Session(graph=actorGraph)
    actorModel.run(actorInit)

    criticGraph = tf.Graph()
    with criticGraph.as_default():
        with tf.name_scope("inputs"):
            state_ = tf.placeholder(tf.float32, [None, numStateSpace], name="state_")
            valueTarget_ = tf.placeholder(tf.float32, [None, 1], name="valueTarget_")

        with tf.name_scope("hidden"):
            initWeight = tf.random_uniform_initializer(-0.03, 0.03)
            initBias = tf.constant_initializer(0.001)
            fullyConnected1_ = tf.layers.dense(inputs=state_, units=numActorFC1Unit, activation=tf.nn.relu,
                                               kernel_initializer=initWeight, bias_initializer=initBias)
            fullyConnected2_ = tf.layers.dense(inputs=fullyConnected1_, units=numActorFC2Unit, activation=tf.nn.relu,
                                               kernel_initializer=initWeight, bias_initializer=initBias)
            fullyConnected3_ = tf.layers.dense(inputs=fullyConnected2_, units=numActorFC3Unit, activation=tf.nn.relu,
                                               kernel_initializer=initWeight, bias_initializer=initBias)
            fullyConnected4_ = tf.layers.dense(inputs=fullyConnected3_, units=numActorFC4Unit, activation=tf.nn.relu,
                                               kernel_initializer=initWeight, bias_initializer=initBias)

        with tf.name_scope("outputs"):
            value_ = tf.layers.dense(inputs=fullyConnected4_, units=1, activation=None, name='value_',
                                     kernel_initializer=initWeight, bias_initializer=initBias)
            diff_ = tf.subtract(valueTarget_, value_, name='diff_')
            loss_ = tf.reduce_mean(tf.square(diff_), name='loss_')
        criticLossSummary = tf.summary.scalar("CriticLoss", loss_)

        with tf.name_scope("train"):
            trainOpt_ = tf.train.AdamOptimizer(learningRateCritic, name='adamOpt_').minimize(loss_)

        criticInit = tf.global_variables_initializer()

    criticModel = tf.Session(graph=criticGraph)
    criticModel.run(criticInit)

    writer = tf.summary.FileWriter("logs/", criticModel.graph)


    xBoundary = [0, 360]
    yBoundary = [0, 360]
    checkBoundaryAndAdjust = ag.CheckBoundaryAndAdjust(xBoundary, yBoundary)

    initSheepPosition = np.array([180, 180])
    initWolfPosition = np.array([180, 180])
    initSheepVelocity = np.array([0, 0])
    initWolfVelocity = np.array([0, 0])
    initSheepPositionNoise = np.array([120, 120])
    initWolfPositionNoise = np.array([60, 60])
    sheepPositionReset = ag.SheepPositionReset(initSheepPosition, initSheepPositionNoise, checkBoundaryAndAdjust)
    wolfPositionReset = ag.WolfPositionReset(initWolfPosition, initWolfPositionNoise, checkBoundaryAndAdjust)

    numOneAgentState = 2
    positionIndex = [0, 1]

    sheepPositionTransition = ag.SheepPositionTransition(numOneAgentState, positionIndex, checkBoundaryAndAdjust)
    wolfPositionTransition = ag.WolfPositionTransition(numOneAgentState, positionIndex, checkBoundaryAndAdjust)

    numAgent = 2
    sheepId = 0
    wolfId = 1
    transitionFunction = env.TransitionFunction(sheepId, wolfId, sheepPositionReset, wolfPositionReset,
                                                sheepPositionTransition, wolfPositionTransition)
    minDistance = 15
    isTerminal = env.IsTerminal(sheepId, wolfId, numOneAgentState, positionIndex, minDistance)

    screen = pg.display.set_mode([xBoundary[1], yBoundary[1]])
    screenColor = [255, 255, 255]
    circleColorList = [[50, 255, 50], [50, 50, 50], [50, 50, 50], [50, 50, 50], [50, 50, 50], [50, 50, 50],
                       [50, 50, 50], [50, 50, 50], [50, 50, 50]]
    circleSize = 8
    saveImage = False
    saveImageFile = 'image'
    render = env.Render(numAgent, numOneAgentState, positionIndex, screen, screenColor, circleColorList, circleSize,
                        saveImage, saveImageFile)

    aliveBouns = -1
    deathPenalty = 20
    rewardDecay = 0.99
    rewardFunction = reward.RewardFunctionTerminalPenalty(sheepId, wolfId, numOneAgentState, positionIndex, aliveBouns,
                                                          deathPenalty, isTerminal)
    accumulateReward = AccumulateReward(rewardDecay, rewardFunction)

    maxTimeStep = 150
    sampleTrajectory = SampleTrajectory(maxTimeStep, transitionFunction, isTerminal)

    approximatePolicy = ApproximatePolicy(actionSpace)
    trainCritic = TrainCriticMonteCarloTensorflow(accumulateReward)
    estimateAdvantage = EstimateAdvantageMonteCarlo(accumulateReward)
    trainActor = TrainActorMonteCarloTensorflow(actionSpace)

    numTrajectory = 50
    maxEpisode = 602
    actorCritic = OfflineAdvantageActorCritic(numTrajectory, maxEpisode, render)

    trainedActorModel, trainedCriticModel = actorCritic(actorModel, criticModel, approximatePolicy, sampleTrajectory,
                                                        trainCritic,
                                                        approximateValue, estimateAdvantage, trainActor)

    savePathActor = 'data/tmpModelActor.ckpt'
    savePathCritic = 'data/tmpModelCritic.ckpt'
    # with actorModel.as_default():
    #     actorSaver.save(trainedActorModel, savePathActor)
    # with criticModel.as_default():
    #     criticSaver.save(trainedCriticModel, savePathCritic)


if __name__ == "__main__":
    main()