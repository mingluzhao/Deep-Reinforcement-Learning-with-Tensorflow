import numpy as np
import random
import tensorflow as tf
import maddpg.maddpgAlgor.common.tf_util as U
from maddpg.maddpgAlgor.common.distributions import make_pdtype
from maddpg.maddpgAlgor import AgentTrainer
from maddpg.maddpgAlgor.trainer.replay_buffer import ReplayBuffer


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r
        r = r*(1.-done)
        discounted.append(r)
    return discounted[::-1]

def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2 #0.95
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0-polyak) * var))
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])     

def p_train(make_obs_ph_n, act_space_n, p_index, p_func, q_func, optimizer, grad_norm_clipping=None, scope="trainer", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]

        p_input = obs_ph_n[p_index]
        p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="p_func")
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))

        # wrap parameters in distribution
        act_pd = act_pdtype_n[p_index].pdfromflat(p)

        act_sample = act_pd.sample()
        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))

        act_input_n = act_ph_n + []
        act_input_n[p_index] = act_pd.sample()
        q_input = tf.concat(obs_ph_n + act_input_n, 1)
        q = q_func(q_input, 1, scope="q_func", reuse=True)[:,0]
        pg_loss = -tf.reduce_mean(q)

        loss = pg_loss + p_reg * 1e-3

        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n, outputs=loss, updates=[optimize_expr])
        act = U.function(inputs=[obs_ph_n[p_index]], outputs=act_sample) 
        p_values = U.function([obs_ph_n[p_index]], p)

        # target network
        target_p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="target_p_func", num_units=num_units)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))     
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)     

        target_act_sample = act_pdtype_n[p_index].pdfromflat(target_p).sample()
        target_act = U.function(inputs=[obs_ph_n[p_index]], outputs=target_act_sample)

        return act, train, update_target_p, {'p_values': p_values, 'target_act': target_act}

def q_train(make_obs_ph_n, act_space_n, q_func, optimizer, grad_norm_clipping=None, scope="trainer", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
        target_ph = tf.placeholder(tf.float32, [None], name="target")

        q_input = tf.concat(obs_ph_n + act_ph_n, 1)
        q = q_func(q_input, 1, scope="q_func")[:,0]
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))
        
        q_loss = tf.reduce_mean(tf.square(q - target_ph))

        # viscosity solution to Bellman differential equation in place of an initial condition
        q_reg = tf.reduce_mean(tf.square(q))
        loss = q_loss

        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss, updates=[optimize_expr])    
        q_values = U.function(obs_ph_n + act_ph_n, q)

        # target network
        target_q = q_func(q_input, 1, scope="target_q_func")[:,0]
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))     
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)    

        target_q_values = U.function(obs_ph_n + act_ph_n, target_q)       

        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}        

class MADDPGEnsembleAgentTrainer(AgentTrainer):
    def __init__(self, name, model, obs_shape_n, act_space_n, agent_index, args):
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        self.counter = 0

        obs_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation"+str(i)+"_ag"+str(agent_index)).get())
            
        # Create all the functions necessary to train the model
        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5
        )        
        self.act, self.p_train, self.p_update, self.p_debug = p_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            p_index=0,
            p_func=model,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5
        )        
        # Create experience buffer
        self.replay_buffer = [ReplayBuffer(1e6) for i in range(self.n)]
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len

    def action(self, obs):
        return self.act(obs[None])[0]

    def experience(self, obs_n, act_n, rew_n, new_obs_n, done_n, terminal):
        # Store transition in the replay buffer.
        for i in range(self.n):
            self.replay_buffer[i].add(obs_n[i], act_n[i], rew_n[i], new_obs_n[i], float(done_n[i]))

    def preupdate(self):
        self.counter += 1
            
    def update(self, agents):
        if len(self.replay_buffer[0]) < self.max_replay_buffer_len:
            return

        if not self.counter % 100 == 0:
            return

        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_buffer[0].make_index(self.args.batch_size)
        for i in range(self.n):  # replay_buffer[0] is self
            obs, act, rew, obs_next, done = self.replay_buffer[i].sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        obs, act, rew, obs_next, done = self.replay_buffer[0].sample_index(index)

        # train q network
        target_act_next_n = [self.p_debug['target_act'](obs_next_n[0])]
        ptr = 0
        for agent in agents:
            if agent is self: continue
            ptr += 1
            target_act_next_n.append(agent.p_debug['target_act'](obs_next_n[ptr]))
        target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))
        target_q = rew + self.args.gamma * (1.0 - done) * target_q_next
        q_loss = self.q_train(*(obs_n + act_n + [target_q]))

        # train p network
        p_loss = self.p_train(*(obs_n + act_n))

        self.p_update()
        self.q_update()

        return [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next)]