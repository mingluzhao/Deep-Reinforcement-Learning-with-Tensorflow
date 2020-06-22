import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import argparse
import tensorflow.contrib.layers as layers
from gym import spaces
import pickle

from environment.chasingEnv.multiAgentEnv import *
import maddpg.maddpgAlgor.common.tf_util as U


class BuildActorModel:
    def __init__(self, actionDim, obsDim, numUnits, lr, tau):
        self.actionDim = actionDim
        self.obsDim = obsDim
        self.numUnits = numUnits
        self.lr = lr
        self.tau = tau

    def __call__(self, agentID):
        agentStr = 'Agent'+ str(agentID) if agentID is not None else ''
        print("Generating Actor NN Model")
        graph = tf.Graph()
        with graph.as_default():
            with tf.variable_scope("input"):
                obsPlaceHolder = tf.placeholder(dtype=tf.float32, shape=[None, self.obsDim], name="obs")
                q_ = tf.placeholder(dtype = tf.float32, shape = [None, 1], name = 'targetQ')
                tf.add_to_collection("obsPlaceHolder", obsPlaceHolder)
                tf.add_to_collection("q_", q_)

            with tf.variable_scope("trainNetwork"):
                input = obsPlaceHolder
                outputShape = self.actionDim

                out = input
                out = layers.fully_connected(out, num_outputs=self.numUnits, activation_fn=tf.nn.relu)
                out = layers.fully_connected(out, num_outputs=self.numUnits, activation_fn=tf.nn.relu)
                policyTrainOutput = layers.fully_connected(out, num_outputs=outputShape, activation_fn=None)

            with tf.variable_scope("targetNetwork"):
                input = obsPlaceHolder
                outputShape = self.actionDim

                out = input
                out = layers.fully_connected(out, num_outputs=self.numUnits, activation_fn=tf.nn.relu)
                out = layers.fully_connected(out, num_outputs=self.numUnits, activation_fn=tf.nn.relu)
                policyTargetOutput = layers.fully_connected(out, num_outputs=outputShape, activation_fn=None)

            with tf.variable_scope("parameters"):
                policyNetVariables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='trainNetwork')
                targetNetVariables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='targetNetwork')

            with tf.variable_scope("replaceParam"):
                polyak = 1 - self.tau
                updates = [var_target.assign(polyak * var_target + (1.0 - polyak) * var) for var, var_target \
                           in zip(sorted(policyNetVariables, key=lambda v: v.name),
                                  sorted(targetNetVariables, key=lambda v: v.name))]
                updateParams_ = tf.group(*updates)
                tf.add_to_collection("updateParams_", updateParams_)

            with tf.variable_scope("optimize"):
                p_reg = tf.reduce_mean(tf.square(policyTrainOutput))
                pg_loss = -tf.reduce_mean(q_)
                loss_ = pg_loss + p_reg * 1e-3
                optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
                grad_norm_clipping = 0.5
                trainOpt_ = U.minimize_and_clip(optimizer, loss_, policyNetVariables, grad_norm_clipping)

                tf.add_to_collection("loss_", loss_)
                tf.add_to_collection("trainOpt_", trainOpt_)

            with tf.variable_scope("output"):
                sampleNoise = tf.random_uniform(tf.shape(policyTrainOutput))
                actionSample_ = U.softmax(policyTrainOutput - tf.log(-tf.log(sampleNoise)), axis=-1)
                tf.add_to_collection("actionSample_", actionSample_)

                sampleNoiseTarget = tf.random_uniform(tf.shape(policyTargetOutput))
                targetActionSample = U.softmax(policyTargetOutput - tf.log(-tf.log(sampleNoiseTarget)), axis=-1)
                tf.add_to_collection("actionSampleTarget_", targetActionSample)

            model = tf.Session(graph=graph)
            model.run(tf.global_variables_initializer())

        return model

def actByTargetPolicy(model, observation):
    graph = model.graph
    states_ = graph.get_collection_ref("obsPlaceHolder")[0]
    actionSampleTarget_ = graph.get_collection_ref("actionSampleTarget_")[0]
    actionSampleTarget = model.run(actionSampleTarget_, feed_dict={states_: observation})

    return actionSampleTarget

def updateParam(model):
    graph = model.graph
    updateParams_ = graph.get_collection_ref("updateParams_")[0]
    model.run(updateParams_)

    return model


def trainPolicyNet(model, observation, qVal):
    graph = model.graph
    states_ = graph.get_collection_ref("obsPlaceHolder")[0]
    q_ = graph.get_collection_ref("q_")[0]

    loss_ = graph.get_collection_ref("loss_")[0]
    trainOpt_ = graph.get_collection_ref("trainOpt_")[0]

    loss, trainOpt = model.run([loss_, trainOpt_], feed_dict={states_: observation, q_: qVal})

    return model

def actByTrainPolicy(model, observation):
    graph = model.graph
    states_ = graph.get_collection_ref("obsPlaceHolder")[0]
    actionSample_ = graph.get_collection_ref("actionSample_")[0]
    actionSample = model.run(actionSample_, feed_dict={states_: observation})

    return actionSample


class BuildCritic:
    def __init__(self, obsDim, actionDim, lr, tau, numUnits):
        self.obsDim = obsDim
        self.actionDim = actionDim
        self.lr = lr
        self.tau = tau
        self.numUnits = numUnits

    def __call__(self, agentID):
        agentStr = 'Agent'+ str(agentID) if agentID is not None else ''
        print("Generating Critic NN Model")
        graph = tf.Graph()

        with graph.as_default():
            with tf.variable_scope("input"):
                actionPlaceHolder = tf.placeholder(dtype=tf.float32, shape=[None, self.actionDim], name="action")
                yi_ = tf.placeholder(tf.float32, [None, 1], name="target")
                obsPlaceHolder = tf.placeholder(dtype = tf.float32, shape = [None, self.obsDim])

                tf.add_to_collection("actionPlaceHolder", actionPlaceHolder)
                tf.add_to_collection("yi_", yi_)
                tf.add_to_collection("obsPlaceHolder", obsPlaceHolder)

            with tf.variable_scope("trainQNet"):
                input = tf.concat([obsPlaceHolder, actionPlaceHolder], 1)
                outputShape = 1

                out = input
                out = layers.fully_connected(out, num_outputs=self.numUnits, activation_fn=tf.nn.relu)
                out = layers.fully_connected(out, num_outputs=self.numUnits, activation_fn=tf.nn.relu)

                qTrainOutput_ = layers.fully_connected(out, num_outputs=outputShape, activation_fn=None)

                tf.add_to_collection("qTrainOutput_", qTrainOutput_)

            with tf.variable_scope("targetQNet"):
                input = tf.concat([obsPlaceHolder, actionPlaceHolder], 1)
                outputShape = 1

                out = input
                out = layers.fully_connected(out, num_outputs=self.numUnits, activation_fn=tf.nn.relu)
                out = layers.fully_connected(out, num_outputs=self.numUnits, activation_fn=tf.nn.relu)
                qTargetOutput_ = layers.fully_connected(out, num_outputs=outputShape, activation_fn=None)

                tf.add_to_collection("qTargetOutput_", qTargetOutput_)

            with tf.variable_scope("parameters"):
                # qTrainParams = U.scope_vars(U.absolute_scope_name("trainQNet"))
                # qTargetParams = U.scope_vars(U.absolute_scope_name("targetQNet"))

                qTrainParams = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='trainQNet')
                qTargetParams = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='targetQNet')


            with tf.variable_scope("replaceParam"):
                polyak = 1 - self.tau
                updates = [var_target.assign(polyak * var_target + (1.0 - polyak) * var) for var, var_target \
                           in zip(sorted(qTrainParams, key=lambda v: v.name), sorted(qTargetParams, key=lambda v: v.name))]
                updateParams_ = tf.group(*updates)
                tf.add_to_collection("updateParams_", updateParams_)

            with tf.variable_scope("optimize"):
                loss_ = tf.reduce_mean(tf.square(qTrainOutput_ - yi_))
                optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
                grad_norm_clipping = 0.5
                optimize_expr = U.minimize_and_clip(optimizer, loss_, qTrainParams, grad_norm_clipping)

                tf.add_to_collection("loss_", loss_)
                tf.add_to_collection("trainOpt_", optimize_expr)

            model = tf.Session(graph=graph)
            model.run(tf.global_variables_initializer())

        return model

def evalTrainQ(model, observation, actions):
    graph = model.graph
    states_ = graph.get_collection_ref("obsPlaceHolder")[0]
    actionPlaceHolder = graph.get_collection_ref("actionPlaceHolder")[0]

    qTrainOutput_ = graph.get_collection_ref("qTrainOutput_")[0]
    qTrainOutput = model.run(qTrainOutput_, feed_dict = {states_: observation, actionPlaceHolder: actions})

    return qTrainOutput

def evalTargetQ(model, observation, actions):
    graph = model.graph
    states_ = graph.get_collection_ref("obsPlaceHolder")[0]
    actionPlaceHolder = graph.get_collection_ref("actionPlaceHolder")[0]

    qTargetOutput_ = graph.get_collection_ref("qTargetOutput_")[0]
    qTargetOutput = model.run(qTargetOutput_, feed_dict = {states_: observation, actionPlaceHolder: actions})

    return qTargetOutput

def trainCritic(model, observation, actions, yi):
    graph = model.graph
    states_ = graph.get_collection_ref("obsPlaceHolder")[0]
    actionPlaceHolder = graph.get_collection_ref("actionPlaceHolder")[0]
    yi_ = graph.get_collection_ref("yi_")[0]

    trainOpt_ = graph.get_collection_ref("trainOpt_")[0]
    loss_ = graph.get_collection_ref("loss_")[0]

    loss, trainOpt = model.run([loss_, trainOpt_], feed_dict={states_: observation, actionPlaceHolder: actions, yi_: yi})

    return model

import random

class ReplayBuffer(object):
    def __init__(self, size):
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def make_index(self, batch_size):
        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

    def sample_index(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def collect(self):
        idxes = range(0, len(self._storage))
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)


class SampleFromBuffer:
    def __init__(self, batchSize):
        self.batchSize = batchSize

    def __call__(self, buffer):
        self.replay_sample_index = buffer.make_index(self.batchSize)
        index = self.replay_sample_index
        obs, act, rew, obs_next, done = buffer.sample_index(index)

        return obs, act, rew, obs_next, done


# class UpdateAgentModel:
#     def __init__(self, agentActorModel, agentCriticModel, replayBuffer, learningStartStep, sampleFromBuffer, gamma):
#         self.actorModel = agentActorModel
#         self.criticModel = agentCriticModel
#         self.replayBuffer = replayBuffer
#         self.learningStartStep = learningStartStep
#         self.runTime = 0
#         self.sampleFromBuffer = sampleFromBuffer
#         self.gamma = gamma
#
#     def __call__(self):
#         self.runTime +=1
#
#         if len(self.replayBuffer) < self.learningStartStep: # replay buffer is not large enough
#             return
#         if not self.runTime % 100 == 0:  # only update every 100 steps
#             return
#
#         obs, act, rew, obs_next, done = self.sampleFromBuffer(self.replayBuffer)
#
#         targetNextAction = actByTargetPolicy(self.actorModel, obs)
#         targetNextQ = evalTargetQ(self.criticModel, obs_next, targetNextAction)
#         yi = rew.reshape(-1, 1) + self.gamma * targetNextQ
#
#         self.criticModel = trainCritic(self.criticModel, obs, act, yi)
#
#         # train p network
#         actionSample = actByTrainPolicy(self.actorModel, obs)
#         qVal = evalTrainQ(self.criticModel, obs, actionSample)
#         self.actorModel = trainPolicyNet(self.actorModel, obs, qVal)
#
#         updateParam(self.criticModel)
#         updateParam(self.actorModel)
#
#         return


class UpdateAgentModel:
    def __init__(self, replayBuffer, learningStartStep, sampleFromBuffer, gamma):
        self.replayBuffer = replayBuffer
        self.learningStartStep = learningStartStep
        self.runTime = 0
        self.sampleFromBuffer = sampleFromBuffer
        self.gamma = gamma

    def __call__(self, agentActorModel, agentCriticModel):
        self.runTime += 1

        if len(self.replayBuffer) < self.learningStartStep:  # replay buffer is not large enough
            return [agentActorModel, agentCriticModel]

        if not self.runTime % 100 == 0:  # only update every 100 steps
            return [agentActorModel, agentCriticModel]

        obs, act, rew, obs_next, done = self.sampleFromBuffer(self.replayBuffer)

        targetNextAction = actByTargetPolicy(agentActorModel, obs)
        targetNextQ = evalTargetQ(agentCriticModel, obs_next, targetNextAction)
        yi = rew.reshape(-1, 1) + self.gamma * targetNextQ

        agentCriticModel = trainCritic(agentCriticModel, obs, act, yi)

        # train p network
        actionSample = actByTrainPolicy(agentActorModel, obs)
        qVal = evalTrainQ(agentCriticModel, obs, actionSample)
        agentActorModel = trainPolicyNet(agentActorModel, obs, qVal)

        updatedCriticModel = updateParam(agentCriticModel)
        updatedActorModel = updateParam(agentActorModel)

        modelList = [updatedActorModel, updatedCriticModel]

        return modelList



ddpg = True
wolfSize = 0.075
sheepSize = 0.05
blockSize = 0.2

sheepMaxSpeed = 1.3
wolfMaxSpeed = 1.0
blockMaxSpeed = None

wolfColor = np.array([0.85, 0.35, 0.35])
sheepColor = np.array([0.35, 0.85, 0.35])
blockColor = np.array([0.25, 0.25, 0.25])


def main():
    wolvesID = [0]
    sheepsID = [1]
    blocksID = []

    numWolves = len(wolvesID)
    numSheeps = len(sheepsID)
    numBlocks = len(blocksID)

    numAgents = numWolves + numSheeps
    numEntities = numAgents + numBlocks

    entitiesSizeList = [wolfSize] * numWolves + [sheepSize] * numSheeps + [blockSize] * numBlocks
    entityMaxSpeedList = [wolfMaxSpeed] * numWolves + [sheepMaxSpeed] * numSheeps + [blockMaxSpeed] * numBlocks
    entitiesMovableList = [True] * numAgents + [False] * numBlocks
    massList = [1.0] * numEntities

    isCollision = IsCollision(getPosFromAgentState)
    rewardWolf = RewardWolf(wolvesID, sheepsID, entitiesSizeList, isCollision)
    punishForOutOfBound = PunishForOutOfBound()
    rewardSheep = RewardSheep(wolvesID, sheepsID, entitiesSizeList, getPosFromAgentState, isCollision,
                              punishForOutOfBound)

    rewardFunc = lambda state, action, nextState: \
        list(rewardWolf(state, action, nextState)) + list(rewardSheep(state, action, nextState))

    reset = ResetMultiAgentChasing(numAgents, numBlocks)
    observeOneAgent = lambda agentID: Observe(agentID, wolvesID, sheepsID, blocksID, getPosFromAgentState,
                                              getVelFromAgentState)
    observe = lambda state: [observeOneAgent(agentID)(state) for agentID in range(numAgents)]

    reshapeAction = ReshapeAction()
    getCollisionForce = GetCollisionForce()
    applyActionForce = ApplyActionForce(wolvesID, sheepsID, entitiesMovableList)
    applyEnvironForce = ApplyEnvironForce(numEntities, entitiesMovableList, entitiesSizeList,
                                          getCollisionForce, getPosFromAgentState)
    integrateState = IntegrateState(numEntities, entitiesMovableList, massList,
                                    entityMaxSpeedList, getVelFromAgentState, getPosFromAgentState)
    transit = TransitMultiAgentChasing(numEntities, reshapeAction, applyActionForce, applyEnvironForce, integrateState)

    isTerminal = lambda state: [False]* numAgents

    initObsForParams = observe(reset())
    obsShape = [initObsForParams[obsID].shape for obsID in range(len(initObsForParams))]
    worldDim = 2
    wolfObsDim = np.array(obsShape)[wolvesID[0]][0]
    sheepObsDim = np.array(obsShape)[sheepsID[0]][0]
    actionDim = worldDim * 2 + 1

    numUnits = 64
    lr = 0.01
    tau = 0.01

    buildWolfActorModel = BuildActorModel(actionDim, wolfObsDim, numUnits, lr, tau)
    wolfActorModel = buildWolfActorModel(agentID=0)

    buildSheepActorModel = BuildActorModel(actionDim, sheepObsDim, numUnits, lr, tau)
    sheepActorModel = buildSheepActorModel(agentID=1)

    buildWolfCritic = BuildCritic(wolfObsDim, actionDim, lr, tau, numUnits)
    wolfCriticModel = buildWolfCritic(agentID=0)

    buildSheepCritic = BuildCritic(sheepObsDim, actionDim, lr, tau, numUnits)
    sheepCriticModel = buildSheepCritic(agentID=1)

    maxEpisode = 30000
    maxTimeStep = 25

    gamma = 0.95  #
    bufferSize = 1e6  #
    minibatchSize = 32  #
    learningStartStep = minibatchSize * maxTimeStep  #

    sampleFromBuffer =  SampleFromBuffer(minibatchSize)

    replayBufferWolf = ReplayBuffer(bufferSize)
    replayBufferSheep = ReplayBuffer(bufferSize)

    updateWolf = UpdateAgentModel(replayBufferWolf, learningStartStep, sampleFromBuffer, gamma)
    updateSheep = UpdateAgentModel(replayBufferSheep, learningStartStep, sampleFromBuffer, gamma)


    episode_rewards = [0.0]  # sum of rewards for all agents
    agent_rewards = [[0.0] for _ in range(numAgents)]  # individual agent reward
    episode_step = 0
    train_step = 0

    state = reset()

    print('Starting iterations...')
    while True:
        obs_n = observe(state)
        action_n = [actByTrainPolicy(wolfActorModel, obs_n[0][None])[0], actByTrainPolicy(sheepActorModel, obs_n[1][None])[0]]
        nextState = transit(state, action_n)
        new_obs_n = observe(nextState)
        rew_n = rewardFunc(state, action_n, nextState)

        done_n = isTerminal(state)

        episode_step += 1
        done = all(done_n)
        terminal = (episode_step >= maxTimeStep)

        replayBufferWolf.add(obs_n[0], action_n[0], rew_n[0], new_obs_n[0], float(done_n[0]))
        replayBufferSheep.add(obs_n[1], action_n[1], rew_n[1], new_obs_n[1], float(done_n[1]))

        state = nextState

        for i, rew in enumerate(rew_n):
            episode_rewards[-1] += rew
            agent_rewards[i][-1] += rew

        if done or terminal:
            state = reset()
            episode_step = 0
            episode_rewards.append(0)
            for a in agent_rewards:
                a.append(0)

        # increment global step counter
        train_step += 1

        # updateWolf(wolfActorModel, wolfCriticModel)

        wolfModelList = updateWolf(wolfActorModel, wolfCriticModel)
        wolfActorModel = wolfModelList[0]
        wolfCriticModel = wolfModelList[1]

        sheepModelList = updateSheep(sheepActorModel, sheepCriticModel)
        sheepActorModel = sheepModelList[0]
        sheepCriticModel = sheepModelList[1]


        if terminal and (len(episode_rewards) % 1000 == 0):
            print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}".format(
                train_step, len(episode_rewards), np.mean(episode_rewards[-1000:]),
                [np.mean(rew[-1000:]) for rew in agent_rewards]))

        if len(episode_rewards) > maxEpisode:
            break




if __name__ == '__main__':
    main()










