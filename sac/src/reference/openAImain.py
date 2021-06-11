from src.reference.openAIspinningup import *

import gym
import time
import numpy as np
import tensorflow as tf
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
"""
Soft Actor-Critic (SAC)
Args:
    env_fn : A function which creates a copy of the environment.
        The environment must satisfy the OpenAI Gym API.
    actor_critic: A function which takes in placeholder symbols
        for state, ``x_ph``, and action, ``a_ph``, and returns the main
        outputs from the agent's Tensorflow computation graph:
        ===========  ================  ======================================
        Symbol       Shape             Description
        ===========  ================  ======================================
        ``mu``       (batch, act_dim)  | Computes mean actions from policy
                                       | given states.
        ``pi``       (batch, act_dim)  | Samples actions from policy given
                                       | states.
        ``logp_pi``  (batch,)          | Gives log probability, according to
                                       | the policy, of the action sampled by
                                       | ``pi``. Critical: must be differentiable
                                       | with respect to policy parameters all
                                       | the way through action sampling.
        ``q1``       (batch,)          | Gives one estimate of Q* for
                                       | states in ``x_ph`` and actions in
                                       | ``a_ph``.
        ``q2``       (batch,)          | Gives another estimate of Q* for
                                       | states in ``x_ph`` and actions in
                                       | ``a_ph``.
        ===========  ================  ======================================
    ac_kwargs (dict): Any kwargs appropriate for the actor_critic
        function you provided to SAC.
    seed (int): Seed for random number generators.
    steps_per_epoch (int): Number of steps of interaction (state-action pairs)
        for the agent and the environment in each epoch.
    epochs (int): Number of epochs to run and train agent.
    replay_size (int): Maximum length of replay buffer.
    gamma (float): Discount factor. (Always between 0 and 1.)
    polyak (float): Interpolation factor in polyak averaging for target
        networks. Target networks are updated towards main networks
        according to:
        .. math:: \\theta_{\\text{targ}} \\leftarrow
            \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta
        where :math:`\\rho` is polyak. (Always between 0 and 1, usually
        close to 1.)
    lr (float): Learning rate (used for both policy and value learning).
    alpha (float): Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)
    batch_size (int): Minibatch size for SGD.
    start_steps (int): Number of steps for uniform-random action selection,
        before running real policy. Helps exploration.
    update_after (int): Number of env interactions to collect before
        starting to do gradient descent updates. Ensures replay buffer
        is full enough for useful updates.
    update_every (int): Number of env interactions that should elapse
        between gradient descent updates. Note: Regardless of how long
        you wait between updates, the ratio of env steps to gradient steps
        is locked to 1.
    num_test_episodes (int): Number of episodes to test the deterministic
        policy at the end of each epoch.
    max_ep_len (int): Maximum length of trajectory / episode / rollout.
    logger_kwargs (dict): Keyword args for EpochLogger.
    save_freq (int): How often (in terms of gap between epochs) to save
        the current policy and value function.
"""
def sac(env_fn, actor_critic=mlp_actor_critic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99,
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000,
        update_after=1000, update_every=50, max_ep_len=1000):


    tf.set_random_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Inputs to computation graph
    x_ph, a_ph, x2_ph, r_ph, d_ph = placeholders(obs_dim, act_dim, obs_dim, None, None)

    # Main outputs from computation graph
    with tf.variable_scope('main', reuse= tf.AUTO_REUSE):
        mu, pi, logp_pi, q1, q2 = actor_critic(x_ph, a_ph, **ac_kwargs)

    # with tf.variable_scope('main', reuse=True): # TODO........
        # compose q with pi, for pi-learning
        _, _, _, q1_pi, q2_pi = actor_critic(x_ph, pi, **ac_kwargs)

        # get actions and log probs of actions for next states, for Q-learning
        _, pi_next, logp_pi_next, _, _ = actor_critic(x2_ph, a_ph, **ac_kwargs)

    # Target value network
    with tf.variable_scope('target'):
        # target q values, using actions from *current* policy
        _, _, _, q1_targ, q2_targ = actor_critic(x2_ph, pi_next, **ac_kwargs)

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables
    # var_counts = tuple(count_vars(scope) for scope in ['main/pi', 'main/q1', 'main/q2', 'main'])
    # print('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d, \t total: %d\n' % var_counts)

    # Min Double-Q:
    min_q_pi = tf.minimum(q1_pi, q2_pi)
    min_q_targ = tf.minimum(q1_targ, q2_targ)

    # Entropy-regularized Bellman backup for Q functions, using Clipped Double-Q targets
    q_backup = tf.stop_gradient(r_ph + gamma * (1 - d_ph) * (min_q_targ - alpha * logp_pi_next))

    # Soft actor-critic losses
    pi_loss = tf.reduce_mean(alpha * logp_pi - min_q_pi)
    q1_loss = 0.5 * tf.reduce_mean((q_backup - q1) ** 2)
    q2_loss = 0.5 * tf.reduce_mean((q_backup - q2) ** 2)
    value_loss = q1_loss + q2_loss

    # Policy train op
    # (has to be separate from value train op, because q1_pi appears in pi_loss)
    pi_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))

    # Value train op
    # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
    value_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    value_params = get_vars('main/q')
    with tf.control_dependencies([train_pi_op]):
        train_value_op = value_optimizer.minimize(value_loss, var_list=value_params)

    # Polyak averaging for target variables
    # (control flow because sess.run otherwise evaluates in nondeterministic order)
    with tf.control_dependencies([train_value_op]):
        target_update = tf.group([tf.assign(v_targ, polyak * v_targ + (1 - polyak) * v_main)
                                  for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    # All ops to call during one training step
    step_ops = [pi_loss, q1_loss, q2_loss, q1, q2, logp_pi,
                train_pi_op, train_value_op, target_update]

    # Initializing targets to match main variables
    target_init = tf.group([tf.assign(v_targ, v_main)
                            for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)

    def get_action(o, deterministic=False):
        act_op = mu if deterministic else pi
        return sess.run(act_op, feed_dict={x_ph: o.reshape(1, -1)})[0]

    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    total_steps = steps_per_epoch * epochs

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.
        if t > start_steps:
            a = get_action(o)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            print("step{}, reward{}".format(t, ep_ret))
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                feed_dict = {x_ph: batch['obs1'],
                             x2_ph: batch['obs2'],
                             a_ph: batch['acts'],
                             r_ph: batch['rews'],
                             d_ph: batch['done'],
                             }
                outs = sess.run(step_ops, feed_dict)

        # End of epoch wrap-up
        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Pendulum-v0')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='sac')
    args = parser.parse_args()

    # sac(lambda: gym.make(args.env), actor_critic=mlp_actor_critic,
    #     ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
    #     gamma=args.gamma, seed=args.seed, epochs=args.epochs)

    sac(lambda: gym.make(args.env), actor_critic=mlp_actor_critic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        seed=0, steps_per_epoch=500, epochs=100, replay_size=int(1e6), gamma=0.99,
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=64, start_steps=64,
        update_after=64, update_every=1, max_ep_len=500)
