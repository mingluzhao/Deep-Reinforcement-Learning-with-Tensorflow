import numpy as np
import random
import tensorflow as tf
import maddpg.maddpgAlgor.common.tf_util as U
from maddpg.maddpgAlgor.common.distributions import make_pdtype
from maddpg.maddpgAlgor import AgentTrainer
from maddpg.maddpgAlgor.trainer.replay_buffer import ReplayBuffer

tf.set_random_seed(1)

def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0-polyak) * var))
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])

def p_train(observPlaceHolderList, actionSpaceList, agentIndex, getMLPModel, q_func, optimizer,
            grad_norm_clipping=None, ddpg=False, num_units=64, scope="trainer", reuse=None):

    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        actionPlaceHolderTypeList = [make_pdtype(actionSpace) for actionSpace in actionSpaceList]

        # set up placeholders
        act_ph_n = [actionPlaceHolderTypeList[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(actionSpaceList))]

        p_input = observPlaceHolderList[agentIndex]

        p = getMLPModel(p_input, int(actionPlaceHolderTypeList[agentIndex].param_shape()[0]), scope="getMLPModel", num_units=num_units)
        p_func_vars = U.scope_vars(U.absolute_scope_name("getMLPModel"))

        # wrap parameters in distribution
        act_pd = actionPlaceHolderTypeList[agentIndex].pdfromflat(p)

        act_sample = act_pd.sample()
        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))

        act_input_n = act_ph_n + []
        act_input_n[agentIndex] = act_pd.sample()
        q_input = tf.concat(observPlaceHolderList + act_input_n, 1)
        if ddpg:
            q_input = tf.concat([observPlaceHolderList[agentIndex], act_input_n[agentIndex]], 1)
        q = q_func(q_input, 1, scope="q_func", reuse=True, num_units=num_units)[:,0]
        pg_loss = -tf.reduce_mean(q)

        loss = pg_loss + p_reg * 1e-3

        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=observPlaceHolderList + act_ph_n, outputs=loss, updates=[optimize_expr])
        act = U.function(inputs=[observPlaceHolderList[agentIndex]], outputs=act_sample)
        p_values = U.function([observPlaceHolderList[agentIndex]], p)

        # target network
        target_p = getMLPModel(p_input, int(actionPlaceHolderTypeList[agentIndex].param_shape()[0]), scope="target_p_func", num_units=num_units)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)

        target_act_sample = actionPlaceHolderTypeList[agentIndex].pdfromflat(target_p).sample()
        target_act = U.function(inputs=[observPlaceHolderList[agentIndex]], outputs=target_act_sample)

        return act, train, update_target_p, {'p_values': p_values, 'target_act': target_act}

def q_train(observPlaceHolderList, actionSpaceList, q_index, q_func, optimizer, grad_norm_clipping=None, ddpg=False, scope="trainer", reuse=None, num_units=64):
    with tf.variable_scope(scope, reuse=reuse):  #    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")

    # ddpg = False if maddpgAlgor, = true if ddpg
        # create distribtuions
        actionPlaceHolderTypeList = [make_pdtype(actionSpace) for actionSpace in actionSpaceList]

        # set up placeholders

        act_ph_n = [actionPlaceHolderTypeList[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(actionSpaceList))]
        target_ph = tf.placeholder(tf.float32, [None], name="target")

        q_input = tf.concat(observPlaceHolderList + act_ph_n, 1)
        if ddpg:
            q_input = tf.concat([observPlaceHolderList[q_index], act_ph_n[q_index]], 1)
        q = q_func(q_input, 1, scope="q_func", num_units=num_units)[:,0]
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        q_loss = tf.reduce_mean(tf.square(q - target_ph))

        # viscosity solution to Bellman differential equation in place of an initial condition
        q_reg = tf.reduce_mean(tf.square(q))
        loss = q_loss #+ 1e-3 * q_reg

        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=observPlaceHolderList + act_ph_n + [target_ph], outputs=loss, updates=[optimize_expr])
        q_values = U.function(observPlaceHolderList + act_ph_n, q)

        # target network
        target_q = q_func(q_input, 1, scope="target_q_func", num_units=num_units)[:,0]
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

        target_q_values = U.function(observPlaceHolderList + act_ph_n, target_q)

        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}

class MADDPGAgentTrainer(AgentTrainer):
    def __init__(self, name, model, observationShapeList, actionSpaceList, agent_index, args, ddpg=False):
        self.name = name
        self.numAgents = len(observationShapeList)
        self.agent_index = agent_index
        self.args = args
        observPlaceHolderList = []
        for i in range(self.numAgents):
            observPlaceHolderList.append(U.BatchInput(observationShapeList[i], name="observation"+str(i)).get())

        # Create all the functions necessary to train the model
        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.name,
            observPlaceHolderList=observPlaceHolderList,
            actionSpaceList=actionSpaceList,
            q_index=agent_index,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            ddpg=ddpg,
            num_units=args.num_units
        )
        self.act, self.p_train, self.p_update, self.p_debug = p_train(
            scope=self.name,
            observPlaceHolderList=observPlaceHolderList,
            actionSpaceList=actionSpaceList,
            agentIndex=agent_index,
            getMLPModel=model,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            ddpg=ddpg,
            num_units=args.num_units
        )
        # Create experience buffer
        self.replay_buffer = ReplayBuffer(1e6)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None

    def action(self, obs):
        return self.act(obs[None])[0]

    def experience(self, obs, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t):
        if len(self.replay_buffer) < self.max_replay_buffer_len: # replay buffer is not large enough
            return
        if not t % 100 == 0:  # only update every 100 steps
            return

        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index
        for i in range(self.numAgents):
            obs, act, rew, obs_next, done = agents[i].replay_buffer.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)

        # train q network
        num_sample = 1
        target_q = 0.0
        for i in range(num_sample):
            target_act_next_n = [agents[i].p_debug['target_act'](obs_next_n[i]) for i in range(self.numAgents)]
            target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))
            target_q += rew + self.args.gamma * (1.0 - done) * target_q_next
        target_q /= num_sample
        q_loss = self.q_train(*(obs_n + act_n + [target_q])) # number representing the q loss

        # train p network
        p_loss = self.p_train(*(obs_n + act_n))

        self.p_update()
        self.q_update()

        return [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]


class DDPGAgentTrainer(AgentTrainer):
    def __init__(self, name, model, observationShapeList, actionSpaceList, agent_index, args, ddpg=True):
        self.name = name
        self.numAgents = len(observationShapeList)
        self.agent_index = agent_index
        self.args = args
        observPlaceHolderList = []
        for i in range(self.numAgents):
            observPlaceHolderList.append(U.BatchInput(observationShapeList[i], name="observation"+str(i)).get())

        # Create all the functions necessary to train the model
        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.name,
            observPlaceHolderList=observPlaceHolderList,
            actionSpaceList=actionSpaceList,
            q_index=agent_index,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            ddpg=ddpg,
            num_units=args.num_units
        )
        self.act, self.p_train, self.p_update, self.p_debug = p_train(
            scope=self.name,
            observPlaceHolderList=observPlaceHolderList,
            actionSpaceList=actionSpaceList,
            agentIndex=agent_index,
            getMLPModel=model,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            ddpg=ddpg,
            num_units=args.num_units
        )
        # Create experience buffer
        self.replay_buffer = ReplayBuffer(1e6)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None

    def action(self, obs):
        return self.act(obs[None])[0]

    def experience(self, obs, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t):
        if len(self.replay_buffer) < self.max_replay_buffer_len: # replay buffer is not large enough
            return
        if not t % 100 == 0:  # only update every 100 steps
            return

        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index
        for i in range(self.numAgents):
            obs, act, rew, obs_next, done = agents[i].replay_buffer.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)

        # train q network
        num_sample = 1
        target_q = 0.0
        for i in range(num_sample):
            target_act_next_n = [agents[i].p_debug['target_act'](obs_next_n[i]) for i in range(self.numAgents)]
            target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))
            target_q += rew + self.args.gamma * (1.0 - done) * target_q_next
        target_q /= num_sample
        q_loss = self.q_train(*(obs_n + act_n + [target_q]))

        # train p network
        p_loss = self.p_train(*(obs_n + act_n))

        self.p_update()
        self.q_update()

        return [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]


def q_train_ddpg(observPlaceHolderList, actionSpaceList, q_index, q_func, optimizer, grad_norm_clipping=None, scope="trainer", reuse=None, num_units=64):
    with tf.variable_scope(scope, reuse=reuse):
        actionPlaceHolderTypeList = [make_pdtype(actionSpace) for actionSpace in actionSpaceList]

        # set up placeholders
        act_ph_n = [actionPlaceHolderTypeList[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(actionSpaceList))]
        target_ph = tf.placeholder(tf.float32, [None], name="target")

        # q_input = tf.concat(observPlaceHolderList + act_ph_n, 1)
        q_input = tf.concat([observPlaceHolderList[q_index], act_ph_n[q_index]], 1) # specific for ddpg
        q = q_func(q_input, 1, scope="q_func", num_units=num_units)[:,0]
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        q_loss = tf.reduce_mean(tf.square(q - target_ph))

        loss = q_loss #+ 1e-3 * q_reg

        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=observPlaceHolderList + act_ph_n + [target_ph], outputs=loss, updates=[optimize_expr])
        q_values = U.function(observPlaceHolderList + act_ph_n, q)

        # target network
        target_q = q_func(q_input, 1, scope="target_q_func", num_units=num_units)[:,0]
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

        target_q_values = U.function(observPlaceHolderList + act_ph_n, target_q)
        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}
