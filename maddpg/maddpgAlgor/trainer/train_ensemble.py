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

import maddpg.maddpgAlgor.common.tf_util as U
from maddpg.maddpgAlgor.trainer.maddpg_ensemble import MADDPGEnsembleAgentTrainer
import tensorflow.contrib.layers as layers

def mlp_model(input, num_outputs, scope, reuse=False, num_units=128, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None) #########
        return out

class MyLogger:
    def __init__(self, logdir, clear_file = False):
        self.fname = logdir
        if clear_file:
            import os
            try:
                os.remove(self.fname)
            except OSError:
                pass

    def print(self, str, to_screen = True):
        if to_screen:
            print(str)
        with open(self.fname, 'a') as f:
            print(str, file=f)

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--seed", type=int, default=0, help="which seed to use")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--evaluate", action="store_true", default=False)
    parser.add_argument("--partition", choices=['rand','fix','mix','adv'], default='rand',
                        help='paritition type: <rand> random partition; <fix> fix partition;' +
                             '<mix> some partitions fix while other mixed; <adv> fix partition against rand partition')
    parser.add_argument("--partition-flag", type=int)
    parser.add_argument("--eval-output", type=str)
    parser.add_argument("--eval-pickle", type=str)
    parser.add_argument("--eval-episode", type=int)
    parser.add_argument("--measure-success", action="store_true", default=False)
    return parser.parse_args()

def make_env(scenario_name, args):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # set fixed partition
    if args.partition != 'rand':
        print('>> Ensemble Partition Type = {}!'.format(args.partition))
        scenario.partition = args.partition
    if args.partition_flag is not None:
        print('>> Partition Flag = {}!'.format(args.partition_flag))
        scenario.partition_flag = args.partition_flag
    if args.evaluate and args.measure_success:
        print('>> Evaluating Success Rate!')
        scenario.measure_success = True
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)  
    return env

def display(**vars):
    import pdb; pdb.set_trace()
    return

if __name__ == '__main__':
    args = parse_args()

    with U.single_threaded_session(): #U.make_session(8):
        # Create environment
        env = make_env(args.scenario, args)
        # Create agent trainers
        trainers = []
        for i,agent in enumerate(env.world.all_agents):
            # TODO: Super Hacky Now! Assume obs_shape_n[0] is own input
            #  >> current: adversary agents, then good agents
            if agent.adversary:
                obs_shape_n = [env.observation_space[j].shape for j in range(env.n)]  # adversary is the first
            else:
                obs_shape_n = [env.observation_space[env.n - 1].shape] \
                              + [env.observation_space[j].shape for j in range(env.n - 1)]  # adversary is the first
            trainers.append(MADDPGEnsembleAgentTrainer(
                "agent_%d" % i, mlp_model, obs_shape_n, env.action_space, i, args)) # TODO

        U.initialize()

        # Load previous results, if necessary
        if args.evaluate or args.restore:
            saver = U.load_state(args.save_dir)
        else:
            saver = tf.train.Saver()

        if args.display and (args.eval_output is not None):
            import matplotlib.pyplot as plt
            plt.ion()
            figure = plt.figure()            

        episode_rewards = [0.0]
        agent_rewards = [[0.0] for _ in range(env.n)]
        obs_n = env.reset()
        episode_step = 0
        t = 0

        logger = MyLogger(args.save_dir + (args.eval_output or 'progress.txt'), not args.restore)
        last_time = time.time()
        while True:
            # get action
            cur_trainers = [trainers[agent.index] for agent in env.world.agents]
            action_n = [agent.action(obs) for agent, obs in zip(cur_trainers,obs_n)]
            # environment step
            new_obs_n, rew_n, done_n, _ = env.step(action_n, episode_step)
            episode_step += 1                
            done = all(done_n)
            terminal = (episode_step >= args.max_episode_len)
            # collect experience
            def switch_list(a, i):
                # put a[i] in the front of a
                return [a[i]] + a[:i] + a[i+1:]
            for i, agent in enumerate(cur_trainers):
                agent.experience(switch_list(obs_n,i), switch_list(action_n,i),
                                 switch_list(rew_n,i), switch_list(new_obs_n,i),
                                 switch_list(done_n,i), terminal)
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if done or terminal:
                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)

            if args.evaluate:
                if args.eval_episode is None:
                    time.sleep(0.05)
                    env.render()
                if terminal and (args.eval_output is not None):
                    output_str = 'episodes:{}, mean episode reward: {}, agent episode reward: {}'.format(
                        len(episode_rewards), np.mean(episode_rewards), [np.mean(rew) for rew in agent_rewards]
                    )
                    logger.print(output_str)
                t += 1
                if (args.eval_episode is not None) and len(episode_rewards) >= args.eval_episode:
                    if args.eval_pickle is not None:
                        with open(args.save_dir + args.eval_pickle, 'wb') as file:
                            pickle.dump([episode_rewards, agent_rewards], file)
                    break
                continue

            if args.display:
                display(**locals())

            # update all trainers
            for agent in cur_trainers:
                agent.preupdate()
            for agent in cur_trainers:
                agent.update(cur_trainers)
            
            # save results
            if terminal and (len(episode_rewards) % args.save_rate == 0):
                U.save_state(args.save_dir, saver=saver)
                
            # display training output
            if terminal and (len(episode_rewards) % args.save_rate == 0):
                elap = time.time() - last_time
                t_elap = "%.2f min" % (elap / 60)
                output_str = "steps: {}, episodes: {}, time elapsed: {}, mean episode reward: {}, agent episode reward: {}".format(
                    t, len(episode_rewards), t_elap, np.mean(episode_rewards[-args.save_rate:]),
                    [np.mean(rew[-args.save_rate:]) for rew in agent_rewards])
                if len(episode_rewards) % args.save_rate == 0:
                    logger.print(output_str)
                if args.eval_output is not None:
                    with open(args.save_dir+args.eval_output, 'a') as f:
                        print(output_str, file=f)
                last_time = time.time()
            t += 1
