import time
import warnings
import numpy as np
import tensorflow as tf

from sac.src.policy import SACPolicy

import random

def unscale_action(action_space, scaled_action):
    low, high = action_space.low, action_space.high
    return low + (0.5 * (scaled_action + 1.0) * (high - low))

def scale_action(action_space, action):
    low, high = action_space.low, action_space.high
    return 2.0 * ((action - low) / (high - low)) - 1.0


class ReplayBuffer(object):
    def __init__(self, size: int):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self) -> int:
        return len(self._storage)

    @property
    def buffer_size(self) -> int:
        return self._maxsize

    def can_sample(self, n_samples: int) -> bool:
        return len(self) >= n_samples

    def is_full(self) -> int:
        return len(self) == self.buffer_size

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def extend(self, obs_t, action, reward, obs_tp1, done):
        for data in zip(obs_t, action, reward, obs_tp1, done):
            if self._next_idx >= len(self._storage):
                self._storage.append(data)
            else:
                self._storage[self._next_idx] = data
            self._next_idx = (self._next_idx + 1) % self._maxsize

    def sample(self, batch_size: int):
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return (np.array(obses_t), np.array(actions),np.array(rewards), np.array(obses_tp1), np.array(dones))


def get_trainable_vars(name):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)


def get_schedule_fn(value_schedule):
    if isinstance(value_schedule, (float, int)):
        def func(_):
            return value_schedule
        value_schedule = func
    else:
        assert callable(value_schedule)
    return value_schedule

class FeedForwardPolicy(SACPolicy):
    """
    Policy object that implements a DDPG-like actor critic, using a feed forward neural network.
    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) The size of the Neural network for the policy (if None, default to [64, 64])
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param layer_norm: (bool) enable layer normalisation
    :param reg_weight: (float) Regularization loss weight for the policy parameters
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, layers=None,
                 cnn_extractor=nature_cnn, feature_extraction="cnn", reg_weight=0.0,
                 layer_norm=False, act_fun=tf.nn.relu, **kwargs):
        super(FeedForwardPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch,
                                                reuse=reuse, scale=(feature_extraction == "cnn"))

        self._kwargs_check(feature_extraction, kwargs)
        self.layer_norm = layer_norm
        self.feature_extraction = feature_extraction
        self.cnn_kwargs = kwargs
        self.cnn_extractor = cnn_extractor
        self.reuse = reuse
        if layers is None:
            layers = [64, 64]
        self.layers = layers
        self.reg_loss = None
        self.reg_weight = reg_weight
        self.entropy = None

        assert len(layers) >= 1, "Error: must have at least one hidden layer for the policy."

        self.activ_fn = act_fun

    def make_actor(self, obs=None, reuse=False, scope="pi"):
        if obs is None:
            obs = self.processed_obs

        with tf.variable_scope(scope, reuse=reuse):
            pi_h = tf.layers.flatten(obs)
            pi_h = mlp(pi_h, self.layers, self.activ_fn, layer_norm=self.layer_norm)
            self.act_mu = mu_ = tf.layers.dense(pi_h, self.ac_space.shape[0], activation=None)
            log_std = tf.layers.dense(pi_h, self.ac_space.shape[0], activation=None)

        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)

        self.std = std = tf.exp(log_std)
        # Reparameterization trick
        pi_ = mu_ + tf.random_normal(tf.shape(mu_)) * std
        logp_pi = gaussian_likelihood(pi_, mu_, log_std)
        self.entropy = gaussian_entropy(log_std)
        deterministic_policy, policy, logp_pi = apply_squashing_func(mu_, pi_, logp_pi)
        self.policy = policy
        self.deterministic_policy = deterministic_policy

        return deterministic_policy, policy, logp_pi

    def make_critics(self, obs=None, action=None, reuse=False, scope="values_fn",
                     create_vf=True, create_qf=True):
        if obs is None:
            obs = self.processed_obs

        with tf.variable_scope(scope, reuse=reuse):
            critics_h = tf.layers.flatten(obs)

            if create_vf:
                # Value function
                with tf.variable_scope('vf', reuse=reuse):
                    vf_h = mlp(critics_h, self.layers, self.activ_fn, layer_norm=self.layer_norm)
                    value_fn = tf.layers.dense(vf_h, 1, name="vf")
                self.value_fn = value_fn

            if create_qf:
                # Concatenate preprocessed state and action
                qf_h = tf.concat([critics_h, action], axis=-1)

                # Double Q values to reduce overestimation
                with tf.variable_scope('qf1', reuse=reuse):
                    qf1_h = mlp(qf_h, self.layers, self.activ_fn, layer_norm=self.layer_norm)
                    qf1 = tf.layers.dense(qf1_h, 1, name="qf1")

                with tf.variable_scope('qf2', reuse=reuse):
                    qf2_h = mlp(qf_h, self.layers, self.activ_fn, layer_norm=self.layer_norm)
                    qf2 = tf.layers.dense(qf2_h, 1, name="qf2")

                self.qf1 = qf1
                self.qf2 = qf2

        return self.qf1, self.qf2, self.value_fn

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            return self.sess.run(self.deterministic_policy, {self.obs_ph: obs})
        return self.sess.run(self.policy, {self.obs_ph: obs})

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run([self.act_mu, self.std], {self.obs_ph: obs})


class OffPolicyRLModel(BaseRLModel):
    def __init__(self, policy, env, replay_buffer=None, _init_setup_model=False, verbose=0, *,
                 requires_vec_env=False, policy_base=None,
                 policy_kwargs=None, seed=None, n_cpu_tf_sess=None):
        super(OffPolicyRLModel, self).__init__(policy, env, verbose=verbose, requires_vec_env=requires_vec_env,
                                               policy_base=policy_base, policy_kwargs=policy_kwargs,
                                               seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)

        self.replay_buffer = replay_buffer

    @abstractmethod
    def setup_model(self):
        pass

    @abstractmethod
    def learn(self, total_timesteps, callback=None,
              log_interval=100, tb_log_name="run", reset_num_timesteps=True, replay_wrapper=None):
        pass

    @abstractmethod
    def predict(self, observation, state=None, mask=None, deterministic=False):
        pass

    @abstractmethod
    def action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        pass


class SAC(OffPolicyRLModel):
    def __init__(self, policy, env, gamma=0.99, learning_rate=3e-4, buffer_size=50000,
                 learning_starts=100, train_freq=1, batch_size=64,
                 tau=0.005, ent_coef='auto', target_update_interval=1,
                 gradient_steps=1, target_entropy='auto', action_noise=None,
                 random_exploration=0.0, verbose=0, tensorboard_log=None,
                 _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False,
                 seed=None, n_cpu_tf_sess=None):

        super(SAC, self).__init__(policy=policy, env=env, replay_buffer=None, verbose=verbose,
                                  policy_base=SACPolicy, requires_vec_env=False, policy_kwargs=policy_kwargs,
                                  seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)

        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.tau = tau
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.gradient_steps = gradient_steps
        self.gamma = gamma
        self.action_noise = action_noise
        self.random_exploration = random_exploration

        self.value_fn = None
        self.graph = None
        self.replay_buffer = None
        self.sess = None
        self.params = None
        self.summary = None
        self.policy_tf = None

        self.obs_target = None
        self.target_policy = None
        self.actions_ph = None
        self.rewards_ph = None
        self.terminals_ph = None
        self.observations_ph = None
        self.action_target = None
        self.next_observations_ph = None
        self.value_target = None
        self.step_ops = None
        self.target_update_op = None
        self.infos_names = None
        self.target_params = None
        self.learning_rate_ph = None
        self.processed_obs_ph = None
        self.processed_next_obs_ph = None
        self.setup_model()

    def setup_model(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session(graph=self.graph)
            self.replay_buffer = ReplayBuffer(self.buffer_size)

            with tf.variable_scope("input", reuse=False):
                # Create policy and target TF objects
                self.policy_tf = self.policy(self.sess, self.observation_space, self.action_space, **self.policy_kwargs)
                self.target_policy = self.policy(self.sess, self.observation_space, self.action_space, **self.policy_kwargs)

                # Initialize Placeholders
                self.observations_ph = self.policy_tf.obs_ph
                # Normalized observation for pixels
                self.processed_obs_ph = self.policy_tf.processed_obs

                self.next_observations_ph = self.target_policy.obs_ph
                self.processed_next_obs_ph = self.target_policy.processed_obs

                self.action_target = self.target_policy.action_ph
                self.terminals_ph = tf.placeholder(tf.float32, shape=(None, 1), name='terminals')
                self.rewards_ph = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
                self.actions_ph = tf.placeholder(tf.float32, shape=(None,) + self.action_space.shape, name='actions')
                self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")

            with tf.variable_scope("model", reuse=False):
                self.deterministic_action, policy_out, logp_pi = self.policy_tf.make_actor(self.processed_obs_ph)
                #  Use two Q-functions to improve performance by reducing overestimation bias.
                qf1, qf2, value_fn = self.policy_tf.make_critics(self.processed_obs_ph, self.actions_ph, create_qf=True, create_vf=True)
                qf1_pi, qf2_pi, _  = self.policy_tf.make_critics(self.processed_obs_ph, policy_out, create_qf=True, create_vf=False, reuse=True)
                self.ent_coef = float(self.ent_coef)

            with tf.variable_scope("target", reuse=False):
                # Create the value network
                _, _, value_target = self.target_policy.make_critics(self.processed_next_obs_ph, create_qf=False, create_vf=True)
                self.value_target = value_target

            with tf.variable_scope("loss", reuse=False):
                # Take the min of the two Q-Values (Double-Q Learning)
                min_qf_pi = tf.minimum(qf1_pi, qf2_pi)

                # Target for Q value regression
                q_backup = tf.stop_gradient(self.rewards_ph + (1 - self.terminals_ph) * self.gamma * self.value_target)

                # Compute Q-Function loss
                # TODO: test with huber loss (it would avoid too high values)
                qf1_loss = 0.5 * tf.reduce_mean((q_backup - qf1) ** 2)
                qf2_loss = 0.5 * tf.reduce_mean((q_backup - qf2) ** 2)

                # Compute the policy loss
                policy_loss = tf.reduce_mean(logp_pi - min_qf_pi)

                # Target for value fn regression
                # We update the vf towards the min of two Q-functions in order to reduce overestimation bias from function approximation error.
                v_backup = tf.stop_gradient(min_qf_pi - self.ent_coef * logp_pi)
                value_loss = 0.5 * tf.reduce_mean((value_fn - v_backup) ** 2)
                values_losses = qf1_loss + qf2_loss + value_loss

                # Policy train op
                # (has to be separate from value train op, because min_qf_pi appears in policy_loss)
                policy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                policy_train_op = policy_optimizer.minimize(policy_loss, var_list=get_trainable_vars('model/pi'))

                # Value train op
                value_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                values_params = get_trainable_vars('model/values_fn')

                source_params = get_trainable_vars("model/values_fn")
                target_params = get_trainable_vars("target/values_fn")

                # Polyak averaging for target variables
                self.target_update_op = [
                    tf.assign(target, (1 - self.tau) * target + self.tau * source)
                    for target, source in zip(target_params, source_params)
                ]
                # Initializing target to match source variables
                target_init_op = [ tf.assign(target, source) for target, source in zip(target_params, source_params)]

                # Control flow is used because sess.run otherwise evaluates in nondeterministic order
                # and we first need to compute the policy action before computing q values losses
                with tf.control_dependencies([policy_train_op]):
                    train_values_op = value_optimizer.minimize(values_losses, var_list=values_params)

                    self.infos_names = ['policy_loss', 'qf1_loss', 'qf2_loss', 'value_loss', 'entropy']
                    # All ops to call during one training step
                    self.step_ops = [policy_loss, qf1_loss, qf2_loss,
                                     value_loss, qf1, qf2, value_fn, logp_pi,
                                     policy_train_op, train_values_op]

            # Retrieve parameters that must be saved
            self.params = get_trainable_vars("model")
            self.target_params = get_trainable_vars("target/values_fn")

            # Initialize Variables and target network
            with self.sess.as_default():
                self.sess.run(tf.global_variables_initializer())
                self.sess.run(target_init_op)

            self.summary = tf.summary.merge_all()

    def _train_step(self, step, learning_rate):
        # Sample a batch from the replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones = batch

        feed_dict = {
            self.observations_ph: batch_obs,
            self.actions_ph: batch_actions,
            self.next_observations_ph: batch_next_obs,
            self.rewards_ph: batch_rewards.reshape(self.batch_size, -1),
            self.terminals_ph: batch_dones.reshape(self.batch_size, -1),
            self.learning_rate_ph: learning_rate
        }


        out = self.sess.run(self.step_ops, feed_dict)

        # Unpack to monitor losses and entropy
        policy_loss, qf1_loss, qf2_loss, value_loss, *values = out
        # qf1, qf2, value_fn, logp_pi, entropy, *_ = values
        entropy = values[4]
        return policy_loss, qf1_loss, qf2_loss, value_loss, entropy

    def learn(self, total_timesteps, callback=None,
              log_interval=4, tb_log_name="SAC", reset_num_timesteps=True):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        callback = self._init_callback(callback)
        self._setup_learn()

        # Transform to callable if needed
        self.learning_rate = get_schedule_fn(self.learning_rate)
        # Initial learning rate
        current_lr = self.learning_rate(1)

        start_time = time.time()
        episode_rewards = [0.0]
        episode_successes = []
        obs = self.env.reset()

        n_updates = 0
        infos_values = []

        callback.on_training_start(locals(), globals())
        callback.on_rollout_start()

        for step in range(total_timesteps):
            # Before training starts, randomly sample actions
            # from a uniform distribution for better exploration.
            # Afterwards, use the learned policy
            # if random_exploration is set to 0 (normal setting)
            if self.num_timesteps < self.learning_starts or np.random.rand() < self.random_exploration:
                # actions sampled from action space are from range specific to the environment
                # but algorithm operates on tanh-squashed actions therefore simple scaling is used
                unscaled_action = self.env.action_space.sample()
                action = scale_action(self.action_space, unscaled_action)
            else:
                action = self.policy_tf.step(obs[None], deterministic=False).flatten()
                # inferred actions need to be transformed to environment action_space before stepping
                unscaled_action = unscale_action(self.action_space, action)

            assert action.shape == self.env.action_space.shape

            new_obs, reward, done, info = self.env.step(unscaled_action)

            self.num_timesteps += 1

            # Only stop training if return value is False, not when it is None. This is for backwards
            # compatibility with callbacks that have no return statement.
            callback.update_locals(locals())
            if callback.on_step() is False:
                break

            obs_, new_obs_, reward_ = obs, new_obs, reward

            # Store transition in the replay buffer.
            self.replay_buffer.add(obs_, action, reward_, new_obs_, done, info)
            obs = new_obs

            # Retrieve reward and episode length if using Monitor wrapper
            maybe_ep_info = info.get('episode')
            if maybe_ep_info is not None:
                self.ep_info_buf.extend([maybe_ep_info])


            if self.num_timesteps % self.train_freq == 0:
                callback.on_rollout_end()

                mb_infos_vals = []
                # Update policy, critics and target networks
                for grad_step in range(self.gradient_steps):
                    if not self.replay_buffer.can_sample(self.batch_size) or self.num_timesteps < self.learning_starts:
                        break
                    n_updates += 1
                    # Compute current learning_rate
                    frac = 1.0 - step / total_timesteps
                    current_lr = self.learning_rate(frac)
                    # Update policy and critics (q functions)
                    mb_infos_vals.append(self._train_step(step, current_lr))
                    # Update target network
                    if (step + grad_step) % self.target_update_interval == 0:
                        # Update target network
                        self.sess.run(self.target_update_op)
                # Log losses and entropy, useful for monitor training
                if len(mb_infos_vals) > 0:
                    infos_values = np.mean(mb_infos_vals, axis=0)

                callback.on_rollout_start()

            episode_rewards[-1] += reward_
            if done:
                obs = self.env.reset()
                episode_rewards.append(0.0)

                maybe_is_success = info.get('is_success')
                if maybe_is_success is not None:
                    episode_successes.append(float(maybe_is_success))

            if len(episode_rewards[-101:-1]) == 0:
                mean_reward = -np.inf
            else:
                mean_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)

            # substract 1 as we appended a new term just now
            num_episodes = len(episode_rewards) - 1
            # Display training infos
        callback.on_training_end()
        return self

    def action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        if actions is not None:
            raise ValueError("Error: SAC does not have action probabilities.")

        warnings.warn("Even though SAC has a Gaussian policy, it cannot return a distribution as it "
                      "is squashed by a tanh before being scaled and outputed.")

        return None

    def predict(self, observation, state=None, mask=None, deterministic=True):
        observation = np.array(observation)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        actions = self.policy_tf.step(observation, deterministic=deterministic)
        actions = actions.reshape((-1,) + self.action_space.shape)  # reshape to the correct action shape
        actions = unscale_action(self.action_space, actions)  # scale the output for the prediction
        # actions = actions[0]

        return actions, None

    def get_parameter_list(self):
        return (self.params + self.target_params)
