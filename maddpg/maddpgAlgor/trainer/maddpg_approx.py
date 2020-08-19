import numpy as np
import random
import tensorflow as tf
import multiagent_rl.common.tf_util as U

from multiagent_rl.common.distributions import make_pdtype
from multiagent_rl import AgentTrainer
from multiagent_rl.trainer.replay_buffer import ReplayBuffer

def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r
        r = r * (1. - done)
        discounted.append(r)
    return discounted[::-1]


def make_update_exp(vals, target_vals, rate=1e-2):
    polyak = 1.0 - rate  # 0.95
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        # expression.append(var_target.assign(var))
        expression.append(var_target.assign(polyak * var_target + (1.0 - polyak) * var))
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])

"""
Only update approximate policy network by online learning
"""
def p_approx_train(make_obs_ph_n, act_space_n, p_index, p_func, q_func, optimizer, grad_norm_clipping=None,
                    local_q_func=False, num_units=64, scope="trainer", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n  # [U.ensure_tf_input(make_obs_ph_n[i]("observation"+str(i))).get() for i in range(len(make_obs_ph_n))]
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action" + str(i)) for i in range(len(act_space_n))]
        act_logits_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action_mode" + str(i)) for i in range(len(act_space_n))]

        p_input = obs_ph_n[p_index]

        p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="p_func")
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))

        # wrap parameters in distribution
        act_pd = act_pdtype_n[p_index].pdfromflat(p)

        act_sample = act_pd.sample()
        #p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))
        p_reg = -tf.reduce_mean(act_pd.entropy())

        act_input_n = act_ph_n + []
        act_target = act_input_n[p_index]
        pg_loss = -tf.reduce_mean(act_pd.logp(act_target))

        loss = pg_loss + p_reg * 1e-3

        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n, outputs=loss, updates=[optimize_expr])
        act = U.function(inputs=[obs_ph_n[p_index]], outputs=act_sample)
        p_values = U.function([obs_ph_n[p_index]], p)

        # target network
        target_p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="target_p_func",
                          num_units=num_units)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)
        sync_target_p = make_update_exp(p_func_vars, target_p_func_vars, rate=1.0)

        target_pd = act_pdtype_n[p_index].pdfromflat(target_p)
        target_act_sample = target_pd.sample()
        target_act = U.function(inputs=[obs_ph_n[p_index]], outputs=target_act_sample)

        target_ph_pd = act_pdtype_n[p_index].pdfromflat(act_logits_ph_n[p_index])
        kl_loss = tf.reduce_mean(target_pd.kl(target_ph_pd))
        f_kl_loss = U.function(inputs=[obs_ph_n[p_index],act_logits_ph_n[p_index]],outputs=kl_loss)

        return act, train, update_target_p, sync_target_p, {'p_values': p_values, 'kl_loss': f_kl_loss, 'target_act': target_act}

def p_train(make_obs_ph_n, act_space_n, p_index, p_func, q_func, optimizer, grad_norm_clipping=None, local_q_func=False,
            num_units=64, scope="trainer", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n  # [U.ensure_tf_input(make_obs_ph_n[i]("observation"+str(i))).get() for i in range(len(make_obs_ph_n))]
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action" + str(i)) for i in range(len(act_space_n))]

        p_input = obs_ph_n[p_index]

        p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="p_func", num_units=num_units)
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))

        # wrap parameters in distribution
        act_pd = act_pdtype_n[p_index].pdfromflat(p)

        act_sample = act_pd.sample()
        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))

        act_input_n = act_ph_n + []
        act_input_n[p_index] = act_pd.sample()  # act_pd.mode() #
        q_input = tf.concat(obs_ph_n + act_input_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[p_index], act_input_n[p_index]], 1)
        q = q_func(q_input, 1, scope="q_func", reuse=True, num_units=num_units)[:, 0]
        pg_loss = -tf.reduce_mean(q)

        loss = pg_loss + p_reg * 1e-3

        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n, outputs=loss, updates=[optimize_expr])
        act = U.function(inputs=[obs_ph_n[p_index]], outputs=act_sample)
        p_values = U.function([obs_ph_n[p_index]], p)

        # target network
        target_p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="target_p_func",
                          num_units=num_units)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)
        sync_target_p = make_update_exp(p_func_vars, target_p_func_vars, rate=1.0)

        target_act_pd = act_pdtype_n[p_index].pdfromflat(target_p)
        target_act_sample = target_act_pd.sample()
        target_act_mode = target_act_pd.mode()
        target_act = U.function(inputs=[obs_ph_n[p_index]], outputs=target_act_sample)
        target_mode = U.function(inputs=[obs_ph_n[p_index]], outputs=target_act_mode)
        target_p_values = U.function([obs_ph_n[p_index]], target_p)

        return act, train, update_target_p, sync_target_p, {'p_values': p_values, 'target_p_values': target_p_values,
                                                            'target_mode': target_mode, 'target_act': target_act}


def q_train(make_obs_ph_n, act_space_n, q_index, q_func, optimizer, grad_norm_clipping=None, local_q_func=False,
            scope="trainer", reuse=None, num_units=64):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n  # [U.ensure_tf_input(make_obs_ph_n[i]("observation"+str(i))).get() for i in range(len(make_obs_ph_n))]
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action" + str(i)) for i in range(len(act_space_n))]
        target_ph = tf.placeholder(tf.float32, [None], name="target")

        q_input = tf.concat(obs_ph_n + act_ph_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[q_index], act_ph_n[q_index]], 1)
        q = q_func(q_input, 1, scope="q_func", num_units=num_units)[:, 0]
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        q_loss = tf.reduce_mean(tf.square(q - target_ph))
        # q_loss = tf.reduce_mean(U.huber_loss(q - target_ph))

        # TEMP: just want to give an viscosity solution to Bellman differential equation
        # in place of an initial condition
        q_reg = tf.reduce_mean(tf.square(q))
        loss = q_loss + 1e-3 * q_reg

        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss, updates=[optimize_expr])
        q_values = U.function(obs_ph_n + act_ph_n, q)

        # target network
        target_q = q_func(q_input, 1, scope="target_q_func", num_units=num_units)[:, 0]
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)
        sync_target_q = make_update_exp(q_func_vars, target_q_func_vars, rate=1.0)

        target_q_values = U.function(obs_ph_n + act_ph_n, target_q)

        return train, update_target_q, sync_target_q, {'q_values': q_values, 'target_q_values': target_q_values}


class MADDPGApproxAgentTrainer(AgentTrainer):
    def __init__(self, name, model, obs_shape_n, act_space_n, agent_index, args,
                 use_approx_policy = True,
                 sync_replay = True, local_q_func=False, update_gap=100):
        self.use_approx_policy = use_approx_policy
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        args.num_units = 64
        self.sync_replay = sync_replay
        self.counter = 0
        self.args = args
        self.update_gap = update_gap

        obs_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation" + str(i)).get())

        # Create all the functions necessary to train the model
        self.q_train, self.q_update, self.q_sync, self.q_debug = q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,  # [lambda name: U.BatchInput(obs_shape, name=name) for obs_shape in obs_shape_n],
            act_space_n=act_space_n,
            q_index=agent_index,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        self.act, self.p_train, self.p_update, self.p_sync, self.p_debug = p_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,  # [lambda name: U.BatchInput(obs_shape, name=name) for obs_shape in obs_shape_n],
            act_space_n=act_space_n,
            p_index=agent_index,
            p_func=model,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        self.approx_act, self.approx_p_train, self.approx_p_update, self.approx_p_sync, self.approx_p_debug = [],[],[],[],[]
        for i in range(self.n):
            if i == self.agent_index:
                t_act, t_p_train, t_p_update, t_p_sync, t_p_debug = self.act, self.p_train, self.p_update, self.p_sync, self.p_debug
            else:
                t_act, t_p_train, t_p_update, t_p_sync, t_p_debug = p_approx_train(
                    scope=self.name+'approx_p_%d'%i,
                    make_obs_ph_n=obs_ph_n,
                    # [lambda name: U.BatchInput(obs_shape, name=name) for obs_shape in obs_shape_n],
                    act_space_n=act_space_n,
                    p_index=i,
                    p_func=model,
                    q_func=model,
                    optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
                    grad_norm_clipping=0.5,
                    local_q_func=local_q_func,
                    num_units=args.num_units
                )
            self.approx_act.append(t_act)
            self.approx_p_train.append(t_p_train)
            self.approx_p_update.append(t_p_update)
            self.approx_p_sync.append(t_p_sync)
            self.approx_p_debug.append(t_p_debug)

        # Create experience buffer
        self.replay_buffer = ReplayBuffer(int(1e6))
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None

    def action(self, obs):
        return self.act(obs[None])[0]
        # return self.p_debug['target_act'](obs[None])[0]

    def experience(self, obs, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def sync_target_nets(self):
        for i in range(self.n):
            self.approx_p_sync[i]()
        self.q_sync()

    def preupdate(self):
        self.replay_sample_index = None
        self.counter += 1

    def update(self, agents):
        # replay buffer is not large enough
        if len(self.replay_buffer) < self.max_replay_buffer_len:
            return None

        if not self.counter % self.update_gap == 0:
            return None

        # agree on a replay samples across all agents
        # as in https://arxiv.org/abs/1703.06182
        if self.sync_replay:
            if agents[0].replay_sample_index is None:
                agents[0].replay_sample_index = agents[0].replay_buffer.make_index(agents[0].args.batch_size)
            self.replay_sample_index = agents[0].replay_sample_index
        else:
            self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)

        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index
        for i in range(self.n):
            obs, act, rew, obs_next, done = agents[i].replay_buffer.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)

        # evaluate kl divergence between approximate policy and the target policy
        target_logits_n = [agents[i].p_debug['target_p_values'](obs_n[i]) for i in range(self.n)]
        kl_loss = 0.0
        for i in range(self.n):
            if i == self.agent_index: continue
            kl_loss += self.approx_p_debug[i]['kl_loss'](obs_n[i], target_logits_n[i])

        # collect latest samples for approximate policy
        latest_obs_n = []
        latest_act_n = []
        latest_index = self.replay_buffer.make_latest_index(self.update_gap)
        for i in range(self.n):
            # TODO: now we approximate the *true policy*, but what we want is actually the target_policy!
            #       Shall we approximate the target net instead???
            t_obs, t_act, _, _, _ = agents[i].replay_buffer.sample_index(latest_index)
            #t_act = agents[i].p_debug['target_act'](t_obs)
            latest_obs_n.append(t_obs)
            latest_act_n.append(t_act)

        # train approximate p network
        for i in range(self.n):
            if i == self.agent_index: continue
            self.approx_p_train[i](*(latest_obs_n + latest_act_n))
            self.approx_p_update[i]()

        # train q network
        if self.use_approx_policy:
            target_act_next_n = [self.approx_p_debug[i]['target_act'](obs_next_n[i]) for i in range(self.n)]
        else:  # use true policy
            target_act_next_n = [agents[i].p_debug['target_act'](obs_next_n[i]) for i in range(self.n)]
        target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))
        target_q = rew + self.args.gamma * (1.0 - done) * target_q_next

        q_loss = self.q_train(*(obs_n + act_n + [target_q]))

        # train p network
        p_loss = self.p_train(*(obs_n + act_n))

        self.p_update()
        self.q_update()
        return [q_loss, p_loss, kl_loss]
