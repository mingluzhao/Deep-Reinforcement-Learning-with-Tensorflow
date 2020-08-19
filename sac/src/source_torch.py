import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

import gym
import random
import numpy as np
from collections import deque

def mini_batch_train(env, agent, max_episodes, max_steps, batch_size):
    episode_rewards = []
    update_step = 0

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            episode_reward += reward

            if len(agent.replay_buffer) > batch_size:
                agent.update(batch_size)
                update_step += 1

            if done or step == max_steps - 1:
                episode_rewards.append(episode_reward)
                break

            state = next_state

        print("Episode " + str(episode) + ": " + str(episode_reward))

    return episode_rewards



class BasicBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)

    def __len__(self):
        return len(self.buffer)



class ValueNetwork(nn.Module):

    def __init__(self, input_dim, output_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class SoftQNetwork(nn.Module):

    def __init__(self, num_inputs, num_actions, hidden_size=256, init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):

    def __init__(self, num_inputs, num_actions, hidden_size=256, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)

        log_pi = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_pi = log_pi.sum(1, keepdim=True)

        return action, log_pi


class SACAgent:

    def __init__(self, env, gamma, tau, v_lr, q_lr, policy_lr, buffer_maxlen):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.env = env
        self.action_range = [env.action_space.low, env.action_space.high]

        # hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.update_step = 0
        self.delay_step = 2

        # initialize networks
        self.value_net = ValueNetwork(env.observation_space.shape[0], 1).to(self.device)
        self.target_value_net = ValueNetwork(env.observation_space.shape[0], 1).to(self.device)
        self.q_net1 = SoftQNetwork(env.observation_space.shape[0], env.action_space.shape[0]).to(self.device)
        self.q_net2 = SoftQNetwork(env.observation_space.shape[0], env.action_space.shape[0]).to(self.device)
        self.policy_net = PolicyNetwork(env.observation_space.shape[0], env.action_space.shape[0]).to(self.device)

        # copy params to target param
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param)

        # initialize optimizers
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=v_lr)
        self.q1_optimizer = optim.Adam(self.q_net1.parameters(), lr=q_lr)
        self.q2_optimizer = optim.Adam(self.q_net2.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

        self.replay_buffer = BasicBuffer(buffer_maxlen)

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, log_std = self.policy_net.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)
        action = action.cpu().detach().squeeze(0).numpy()

        return self.rescale_action(action)

    def rescale_action(self, action):
        return action * (self.action_range[1] - self.action_range[0]) / 2.0 + (
                    self.action_range[1] + self.action_range[0]) / 2.0

    def update(self, batch_size):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        dones = dones.view(dones.size(0), -1)

        next_actions, next_log_pi = self.policy_net.sample(next_states)
        next_q1 = self.q_net1(next_states, next_actions)
        next_q2 = self.q_net2(next_states, next_actions)
        next_v = self.target_value_net(next_states)

        # value Loss
        next_v_target = torch.min(next_q1, next_q2) - next_log_pi
        curr_v = self.value_net.forward(states)
        v_loss = F.mse_loss(curr_v, next_v_target.detach())

        # q loss
        curr_q1 = self.q_net1.forward(states, actions)
        curr_q2 = self.q_net2.forward(states, actions)
        expected_q = rewards + (1 - dones) * self.gamma * next_v
        q1_loss = F.mse_loss(curr_q1, expected_q.detach())
        q2_loss = F.mse_loss(curr_q2, expected_q.detach())

        # update value network and q networks
        self.value_optimizer.zero_grad()
        v_loss.backward()
        self.value_optimizer.step()

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # delayed update for policy net and target value nets
        if self.update_step % self.delay_step == 0:
            new_actions, log_pi = self.policy_net.sample(states)
            min_q = torch.min(
                self.q_net1.forward(states, new_actions),
                self.q_net2.forward(states, new_actions)
            )
            policy_loss = (log_pi - min_q).mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            # target networks
            for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        self.update_step += 1



# class SACAgent:
#
#     def __init__(self, env, gamma, tau, v_lr, q_lr, policy_lr, buffer_maxlen):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#         self.env = env
#         self.action_range = [env.action_space.low, env.action_space.high]
#
#         # hyperparameters
#         self.gamma = gamma
#         self.tau = tau
#         self.update_step = 0
#         self.delay_step = 2
#
#         # initialize networks
#         self.value_net = ValueNetwork(env.observation_space.shape[0], 1).to(self.device)
#         self.target_value_net = ValueNetwork(env.observation_space.shape[0], 1).to(self.device)
#         self.q_net1 = SoftQNetwork(env.observation_space.shape[0], env.action_space.shape[0]).to(self.device)
#         self.q_net2 = SoftQNetwork(env.observation_space.shape[0], env.action_space.shape[0]).to(self.device)
#         self.policy_net = PolicyNetwork(env.observation_space.shape[0], env.action_space.shape[0]).to(self.device)
#
#         # copy params to target param
#         for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
#             target_param.data.copy_(param)
#
#         # initialize optimizers
#         self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=v_lr)
#         self.q1_optimizer = optim.Adam(self.q_net1.parameters(), lr=q_lr)
#         self.q2_optimizer = optim.Adam(self.q_net2.parameters(), lr=q_lr)
#         self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
#
#         self.replay_buffer = BasicBuffer(buffer_maxlen)
#
#     def get_action(self, state):
#         state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
#         mean, log_std = self.policy_net.forward(state)
#         std = log_std.exp()
#
#         normal = Normal(mean, std)
#         z = normal.sample()
#         action = torch.tanh(z)
#         action = action.cpu().detach().squeeze(0).numpy()
#
#         return self.rescale_action(action)
#
#     def rescale_action(self, action):
#         return action * (self.action_range[1] - self.action_range[0]) / 2.0 + (
#                     self.action_range[1] + self.action_range[0]) / 2.0
#
#     def update(self, batch_size):
#         states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
#         states = torch.FloatTensor(states).to(self.device)
#         actions = torch.FloatTensor(actions).to(self.device)
#         rewards = torch.FloatTensor(rewards).to(self.device)
#         next_states = torch.FloatTensor(next_states).to(self.device)
#         dones = torch.FloatTensor(dones).to(self.device)
#         dones = dones.view(dones.size(0), -1)
#
#         next_actions, next_log_pi = self.policy_net.sample(states)
#         next_q1 = self.q_net1(states, next_actions)
#         next_q2 = self.q_net2(states, next_actions)
#         next_v = self.target_value_net(next_states)
#
#         # value Loss
#         next_v_target = torch.min(next_q1, next_q2) - next_log_pi
#         curr_v = self.value_net.forward(states)
#         v_loss = F.mse_loss(curr_v, next_v_target.detach())
#
#         # q loss
#         curr_q1 = self.q_net1.forward(states, actions)
#         curr_q2 = self.q_net2.forward(states, actions)
#         expected_q = rewards + (1 - dones) * self.gamma * next_v
#         q1_loss = F.mse_loss(curr_q1, expected_q.detach())
#         q2_loss = F.mse_loss(curr_q2, expected_q.detach())
#
#         # update value network and q networks
#         self.value_optimizer.zero_grad()
#         v_loss.backward()
#         self.value_optimizer.step()
#
#         self.q1_optimizer.zero_grad()
#         q1_loss.backward()
#         self.q1_optimizer.step()
#
#         self.q2_optimizer.zero_grad()
#         q2_loss.backward()
#         self.q2_optimizer.step()
#
#         # delayed update for policy net and target value nets
#         if self.update_step % self.delay_step == 0:
#             new_actions, log_pi = self.policy_net.sample(states)
#             min_q = torch.min(
#                 self.q_net1.forward(states, new_actions),
#                 self.q_net2.forward(states, new_actions)
#             )
#             policy_loss = (log_pi - min_q).mean()
#
#             self.policy_optimizer.zero_grad()
#             policy_loss.backward()
#             self.policy_optimizer.step()
#
#             # target networks
#             for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
#                 target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)
#
#         self.update_step += 1




env = gym.make("Pendulum-v0")

# SAC 2018 Params
tau = 0.005
gamma = 0.99
value_lr = 3e-3
q_lr = 3e-3
policy_lr = 3e-3
buffer_maxlen = 1000000

# 2018 agent
agent = SACAgent(env, gamma, tau, value_lr, q_lr, policy_lr, buffer_maxlen)

# train
episode_rewards = mini_batch_train(env, agent, 50, 500, 64)

# In[ ]:




