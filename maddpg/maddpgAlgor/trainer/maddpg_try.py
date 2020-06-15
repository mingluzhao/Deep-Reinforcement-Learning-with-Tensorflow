import numpy as np
import tensorflow as tf
import maddpg.maddpgAlgor.common.tf_util as U
from maddpg.maddpgAlgor import AgentTrainer
from maddpg.maddpgAlgor.trainer.replay_buffer import ReplayBuffer

def p_train(observPlaceHolderList, actionSpaceList, agentIndex, p_func, q_func, optimizer,grad_norm_clipping,
            ddpg, num_units=64, scope="trainer", reuse=None):

    with tf.variable_scope(scope, reuse=reuse):
        # set up placeholders
        actionPlaceHolderList = [tf.placeholder(dtype=tf.float32, shape=[None] + [actionSpaceList[i].n], name="action"+str(i)) for i in range(len(actionSpaceList))]

        policyNetInput = observPlaceHolderList[agentIndex] # personal observation 
        policyOutputShape = int(actionSpaceList[agentIndex].n)
        policyTrainOutput = p_func(policyNetInput, policyOutputShape, scope="p_func", num_units=num_units)
        policyNetVariables = U.scope_vars(U.absolute_scope_name("p_func"))

        sampleNoise = tf.random_uniform(tf.shape(policyTrainOutput), seed = 0)
        actionSample = U.softmax(policyTrainOutput - tf.log(-tf.log(sampleNoise)), axis=-1) # output of function act
        p_reg = tf.reduce_mean(tf.square(policyTrainOutput))
        
        actionInputPlaceHolderList = actionPlaceHolderList + []
        actionInputPlaceHolderList[agentIndex] = actionSample

        qNetInput = tf.concat(observPlaceHolderList + actionInputPlaceHolderList, 1)
        if ddpg:
            qNetInput = tf.concat([observPlaceHolderList[agentIndex], actionSample], 1)

        q = q_func(qNetInput, 1, scope="q_func", reuse=True, num_units=num_units)[:,0]
        pg_loss = -tf.reduce_mean(q)

        loss = pg_loss + p_reg * 1e-3  ####### didnt change this optimization process in my ddpg

        optimize_expr = U.minimize_and_clip(optimizer, loss, policyNetVariables, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=observPlaceHolderList + actionPlaceHolderList, outputs=loss, updates=[optimize_expr])
        act = U.function(inputs=[observPlaceHolderList[agentIndex]], outputs=actionSample)
        p_values = U.function([observPlaceHolderList[agentIndex]], policyTrainOutput)

        # target network
        target_p = p_func(policyNetInput, int(actionSpaceList[agentIndex].n), scope="target_p_func", num_units=num_units)
        targetNetVariables = U.scope_vars(U.absolute_scope_name("target_p_func"))
        update_target_p = make_update_exp(policyNetVariables, targetNetVariables)

        uTarget = tf.random_uniform(tf.shape(target_p))
        target_act_sample = U.softmax(target_p - tf.log(-tf.log(uTarget)), axis=-1)
        target_act = U.function(inputs=[observPlaceHolderList[agentIndex]], outputs=target_act_sample)

        return act, train, update_target_p, {'p_values': p_values, 'target_act': target_act}


def make_update_exp(vals, target_vals, polyak = 1.0 - 1e-2):
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0-polyak) * var))
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])



def q_train(observPlaceHolderList, actionSpaceList, agentIndex, q_func, optimizer,
            grad_norm_clipping=None, ddpg=False, scope="trainer", reuse=None, num_units=64):

    with tf.variable_scope(scope, reuse=reuse):
        actionPlaceHolderList = [tf.placeholder(dtype=tf.float32, shape=[None] + [actionSpaceList[i].n], name="action"+str(i)) for i in range(len(actionSpaceList))]
        yi_ = tf.placeholder(tf.float32, [None], name="target")

        q_input = tf.concat(observPlaceHolderList + actionPlaceHolderList, 1)# shape (?, 24)
        if ddpg:
            q_input = tf.concat([observPlaceHolderList[agentIndex], actionPlaceHolderList[agentIndex]], 1) # shape (?, 13)

        q = q_func(q_input, 1, scope="q_func", num_units=num_units)[:, 0] # drop a level: shape (?, 1) to shape (?,)
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        loss = tf.reduce_mean(tf.square(q - yi_))
        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=observPlaceHolderList + actionPlaceHolderList + [yi_], outputs=loss, updates=[optimize_expr])
        q_values = U.function(observPlaceHolderList + actionPlaceHolderList, q)

        # target network
        target_q = q_func(q_input, 1, scope="target_q_func", num_units=num_units)[:,0]
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

        target_q_values = U.function(observPlaceHolderList + actionPlaceHolderList, target_q)

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
            agentIndex=agent_index,
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
            p_func=model,
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

        sess = tf.Session()  # get session
        writer = tf.summary.FileWriter("logs/", sess.graph)

        self.learnStart = 0

    def action(self, obs):
        # obs = np.array([ 0., 0., -0.91739134,  0.96584283,  0.30922375, -0.95104116, 0. , 0.])
        actionOutput = self.act(obs[None])[0]
        actionWithoutNoise = self.p_debug['p_values'](obs[None])[0]
        # print(actionWithoutNoise, 'noisy', actionOutput)
        return actionOutput

    def experience(self, obs, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t):

        # everyone has a replay buffer, the buffer contains information for personal experience
        # and each time it samples an index, everyone act once, store information to the buffer
        # then use what i did in this step to update myself

        # when updating, has an index, take the experience from all agents at that specific index


        if len(self.replay_buffer) < self.max_replay_buffer_len: # replay buffer is not large enough
            return
        if not t % 100 == 0:  # only update every 100 steps
            return

        self.learnStart += 1
        if self.learnStart is 1:
            print('---------------------------------learning starts--------------------------------------------')


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
            # buffer information

        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)# personal buffer

        # train q network
        target_q = 0.0
        target_act_next_n = [agents[i].p_debug['target_act'](obs_next_n[i]) for i in range(self.numAgents)]
        target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))
        target_q += rew + self.args.gamma * (1.0 - done) * target_q_next
        q_loss = self.q_train(*(obs_n + act_n + [target_q]))

        # train p network
        p_loss = self.p_train(*(obs_n + act_n))

        self.p_update()
        self.q_update()

        return [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]
