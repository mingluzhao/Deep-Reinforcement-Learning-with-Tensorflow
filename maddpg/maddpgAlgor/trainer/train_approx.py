import argparse
import gym
import numpy as np
import os
import tensorflow as tf
import time
import pickle

import multiagent_rl.common.tf_util as U

from multiagent_rl.trainer.ddpg import DDPGAgentTrainer
from multiagent_rl.trainer.maddpg_approx import MADDPGApproxAgentTrainer
from multiagent_rl.trainer.maddpg_ensemble import MADDPGEnsembleAgentTrainer
from multiagent_rl.trainer.qac import QACAgentTrainer
from multiagent_rl.trainer.pg import PGAgentTrainer, FeedbackPGAgentTrainer
from model import mlp_model

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
    parser.add_argument("--no-sync-replay", action="store_false", dest="sync_replay")
    parser.set_defaults(sync_deplay=True)
    parser.add_argument("--use-true-policy", action="store_false", dest="use_approx_policy")
    parser.set_defaults(use_approx_policy=True)
    parser.add_argument("--update-gap", type=int, default=100)
    parser.add_argument("--total-episodes", type=int, default=30000)
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--evaluate", action="store_true", default=False)
    parser.add_argument("--eval-output", type=str)
    parser.add_argument("--pickle-file", type=str)
    return parser.parse_args()

def make_env(scenario_name, args):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)  
    return env

def display(**vars):
    import pdb; pdb.set_trace()
    return        
    # if terminal and (len(episode_rewards) % args.save_rate == 0):
    # # if len(trainers[0].replay_buffer) > (args.batch_size * args.max_episode_len):
    #     plt.clf()
    #     w = 16
    #     X,Y = np.meshgrid(np.linspace(-1, +1, w), np.linspace(-1, +1, w))
    #     obs = np.stack([X.flatten(),Y.flatten()], 1)
    #     act = trainers[0].act(obs)
    #     Q = trainers[0].q_debug['q_values'](obs, act)
    #     target_Q = trainers[0].q_debug['target_q_values'](obs, act)
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(np.reshape(Q,[w,w]))
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(np.reshape(target_Q,[w,w]))
        
    #     # for i in range(5):
    #     #     plt.subplot(1, 5, 1+i)
    #     #     plt.imshow(np.reshape(Q[:,i],[w,w]))

    #     plt.draw()
    #     # writer.grab_frame()
    #     time.sleep(1e-5)
    #     plt.pause(1e-5)

if __name__ == '__main__':
    args = parse_args()

    if args.sync_replay:
        print(">> Replay Index Synced!")

    with U.single_threaded_session(): #U.make_session(8):
        # Create environment
        env = make_env(args.scenario, args)
        # Create agent trainers
        trainers = []
        #trainers.append(MADDPGAgentTrainer(
        #    "agent_%d" % 0, mlp_model, obs_shape_n, env.action_space, 0, args))
        #trainers.append(DDPGAgentTrainer(
        #    "agent_%d" % 0, mlp_model, obs_shape_n[0], env.action_space[0], args))
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        for i in range(env.n):
            # TODO: Super Hacky Now! Assume obs_shape_n[0] is own input
            #  >> current: adversary agents, then good agents
            trainers.append(MADDPGApproxAgentTrainer(
                "agent_%d" % i, mlp_model, obs_shape_n, env.action_space, i, args,
                use_approx_policy=args.use_approx_policy,
                sync_replay=args.sync_replay, update_gap=args.update_gap))
            #trainers.append(DDPGAgentTrainer(
            #    "agent_%d" % i, mlp_model, obs_shape_n[i], env.action_space[i], args))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if args.evaluate or args.restore:
            saver = U.load_state(args.save_dir)
        else:
            saver = tf.train.Saver()
            for trainer in trainers:
                trainer.sync_target_nets()

        if args.display:
            import matplotlib.pyplot as plt
            plt.ion()
            figure = plt.figure()            

        episode_rewards = [0.0]
        agent_rewards = [[0.0] for _ in range(env.n)]
        agent_kl = [0.0] * env.n
        obs_n = env.reset()
        episode_step = 0
        t = 0

        logger = MyLogger(args.save_dir + (args.eval_output or 'progress.txt'), not args.restore)
        t_elap = time.time()
        counter = [0] * env.n

        if (args.pickle_file is not None) and not args.evaluate:
            train_stats = dict(reward=[])
            for i in range(env.n):
                train_stats["kl_agent{}".format(i)] = []
        else:
            train_stats = None

        while True:
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
            # environment step
            new_obs_n, rew_n, done_n, _ = env.step(action_n, episode_step)
            episode_step += 1                
            done = all(done_n)
            terminal = (episode_step >= args.max_episode_len)
            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
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
                time.sleep(0.05)
                env.render()
                if terminal and (args.eval_output is not None):
                    output_str = 'episodes:{}, mean episode reward: {}, agent episode reward: {}'.format(
                        len(episode_rewards), np.mean(episode_rewards), [np.mean(rew) for rew in agent_rewards]
                    )
                    logger.print(output_str)
                continue

            if args.display:
                display(**locals())

            # update all trainers
            for agent in trainers:
                agent.preupdate()
            has_update = False
            for i,agent in enumerate(trainers):
                info = agent.update(trainers)
                if info is not None:
                    has_update = True
                    counter[i] += 1
                    _, _, kl = info
                    agent_kl[i] += kl
                    if train_stats is not None:
                        train_stats["kl_agent{}".format(i)].append(kl)
            if has_update and (train_stats is not None):
                train_stats["reward"].append(np.mean(episode_rewards[-args.save_rate:]))

            # save results
            if terminal and (len(episode_rewards) % args.save_rate == 0):
                U.save_state(args.save_dir, saver=saver)
                
            # display training output
            if terminal and ((len(episode_rewards) % args.save_rate == 0) or args.eval_output is not None):
                # save train stats
                if train_stats is not None:
                    with open(args.save_dir+args.pickle_file,'wb') as file:
                        pickle.dump(train_stats, file)
                dur = (time.time() - t_elap) / 60
                if counter[0] > 0:
                    for i in range(env.n):
                        agent_kl[i] /= counter[i]
                    avg_kl = np.mean(agent_kl)
                else:
                    avg_kl = -1
                output_str = "steps: {}, episodes: {}, time elapsed: {} min, mean episode reward: {}, agent episode reward: {}, avg approx kl: {}".format(
                    t, len(episode_rewards), dur, np.mean(episode_rewards[-args.save_rate:]),
                    [np.mean(rew[-args.save_rate:]) for rew in agent_rewards], avg_kl)
                if len(episode_rewards) % args.save_rate == 0:
                    logger.print(output_str)
                if args.eval_output is not None:
                    with open(args.save_dir+args.eval_output, 'a') as f:
                        print(output_str, file=f)
                counter = [0] * env.n
                agent_kl = [0.0] * env.n
                t_elap = time.time()
            t += 1

            if len(episode_rewards) >= args.total_episodes: break