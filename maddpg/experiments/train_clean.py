import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import matplotlib.pyplot as plt

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
from functionTools.loadSaveModel import saveToPickle
import maddpg.maddpgAlgor.common.tf_util as U
from maddpg.maddpgAlgor.trainer.maddpg_ref import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers

modelNameList = ['policy3WolfMADDPG1SheepMADDPG', 'policy3WoolfMADDPG1SheepMADDPG',
                 'policy3WoolfMADDPG1SheepMADDPG11111', 'policy3WoolfMADDPG1SheepMADDPG111111', 'policy3WoolfMADDPG1SheepMADDPG11111111']
trajectoryPath = os.path.join(dirName, '..', 'trainedModels', 'sourceCodeModels', 'policy6WolfMADDPG1SheepMADDPG11111')

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=75, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=300, help="number of episodes") #60000
    parser.add_argument("--num-adversaries", type=int, default=3, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=128, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default='exp', help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default=trajectoryPath, help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default=os.path.join(dirName, '..', 'benchmark_files'), help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default=os.path.join(dirName, '..', 'learning_curves'), help="directory where plot data is saved")
    return parser.parse_args()


def mlp_model(input, num_outputs, scope, reuse=False, num_units=128, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None) #########
        return out

def make_env(scenario_name, arglist, benchmark=False):  #### why is arglist not used?????
    from maddpg.multiagent.environment import MultiAgentEnv
    import maddpg.multiagent.scenarios as scenarios

    # load scenario from script
    # scenario = scenarios.load(scenario_name + ".py").Scenario()
    scenario = scenarios.load("simple_tag.py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
        print(env.action_space)
    return env


def get_trainers(env, num_adversaries, obs_shape_n, arglist): # get_trainers(env, num_adversaries, obs_shape_n, arglist)
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer("agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            arglist.adv_policy=='ddpg'))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            arglist.good_policy=='ddpg'))
    return trainers


def train(arglist):
    with U.single_threaded_session():
        env = make_env(arglist.scenario, arglist, arglist.benchmark) # TODO: deal with personal environment
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)

        U.initialize()
        obs_n = env.reset()
        episode_step = 0
        train_step = 0

        print('Starting iterations...')
        while True:
            action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            episode_step += 1
            terminal = (episode_step >= arglist.max_episode_len)

            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n

            if terminal:
                obs_n = env.reset()
                episode_step = 0

            train_step += 1

            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                agent.update(trainers, train_step)



if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)

