import os
import sys
import argparse

os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
# sys.path.append(os.path.join(dirName, '..'))
# sys.path.append(os.path.join(dirName, '..', '..'))

from environment.gymEnv.multiAgentEnv_func import *
from functionTools.loadSaveModel import loadFromPickle, saveToPickle
from functionTools.trajectory import SampleTrajectory

from maddpg.multiagent.environment import MultiAgentEnv
import maddpg.maddpg.common.tf_util as U
from maddpg.maddpg.trainer.maddpg import MADDPGAgentTrainer
import maddpg.multiagent.scenarios as scenarios
import tensorflow.contrib.layers as layers
import time

trajectoryPath = os.path.join(dirName, '..', 'policy')

wolfSize = 0.075
sheepSize = 0.05
blockSize = 0.2

sheepMaxSpeed = 1.3
wolfMaxSpeed = 1.0
blockMaxSpeed = None

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=10000, help="number of episodes") #60000
    parser.add_argument("--num-adversaries", type=int, default=3, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default='exp', help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default= trajectoryPath, help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=True)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default=os.path.join(dirName, '..', 'benchmark_files'), help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default=os.path.join(dirName, '..', 'learning_curves'), help="directory where plot data is saved")
    return parser.parse_args()


def main():
    wolvesID = [0, 1]
    sheepsID = [2]
    blocksID = [3]
    
    numWolves = len(wolvesID)
    numSheeps = len(sheepsID)
    numBlocks = len(blocksID)
    
    numAgents = numWolves + numSheeps
    numEntities = numAgents + numBlocks
    
    entitiesSizeList = [wolfSize]* numWolves + [sheepSize] * numSheeps + [blockSize]* numBlocks
    entityMaxSpeedList = [wolfMaxSpeed]* numWolves + [sheepMaxSpeed] * numSheeps + [blockMaxSpeed]* numBlocks
    entitiesMovableList = [True]* numAgents + [False] * numBlocks
    massList = [1.0] * numEntities
    
    isCollision = IsCollision(getPosFromAgentState)
    rewardWolf = RewardWolf(wolvesID, sheepsID, entitiesSizeList, isCollision)
    punishForOutOfBound = PunishForOutOfBound()
    rewardSheep = RewardSheep(wolvesID, sheepsID, entitiesSizeList, getPosFromAgentState, isCollision, punishForOutOfBound)

    rewardFunc = lambda state, action, nextState: \
        list(rewardWolf(state, action, nextState)) + list(rewardSheep(state, action, nextState))

    resetState = ResetMultiAgentChasing(numAgents, numBlocks)
    observeOneAgent = lambda agentID: Observe(agentID, wolvesID, sheepsID, blocksID, getPosFromAgentState, getVelFromAgentState)
    observe = lambda state: [observeOneAgent(agentID)(state) for agentID in range(numAgents)]
    reset = lambda: resetState()

    reshapeAction = ReshapeAction()
    getCollisionForce = GetCollisionForce()
    applyActionForce = ApplyActionForce(wolvesID, sheepsID, entitiesMovableList)
    applyEnvironForce = ApplyEnvironForce(numEntities, entitiesMovableList, entitiesSizeList,
                                          getCollisionForce, getPosFromAgentState)
    integrateState = IntegrateState(numEntities, entitiesMovableList, massList,
                                    entityMaxSpeedList, getVelFromAgentState, getPosFromAgentState)
    transit = TransitMultiAgentChasing(numEntities, reshapeAction, applyActionForce, applyEnvironForce, integrateState)

    useDDPG = False
    isTerminal = lambda state: False

    maxRunningSteps = 25
    sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, rewardFunc, reset)


    with U.single_threaded_session():
        scenario = scenarios.load("simple_tag.py").Scenario()
        world = scenario.make_world()
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        trainers = []
        model = mlp_model
        trainer = MADDPGAgentTrainer
        arglist = parse_args()

        for i in range(numWolves):
            trainers.append(trainer("agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist, local_q_func= useDDPG))
        for i in range(numWolves, numAgents):
            trainers.append(trainer("agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist, local_q_func= useDDPG))

        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        print(reset())

        policy = lambda state: [agent.action(obs) for agent, obs in zip(trainers, observe(state))]

        traj = sampleTrajectory(policy)

        # for episodeID in range(numEpisodes):
        #     action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)] # shape = 3* 5
        #
        #     getStateFromEnv = lambda entityID: list(env.world.entities[entityID].state.p_pos) + list(env.world.entities[entityID].state.p_vel)
        #     numEntities = len(env.world.entities)
        #     currentState = [getStateFromEnv(entityID) for entityID in range(numEntities)]
        #
        #     # environment step
        #     new_obs_n, rew_n, done_n, info_n = env.step(action_n)
        #
        #     nextState = [getStateFromEnv(entityID) for entityID in range(numEntities)]
        #     currentTimeStepInfo = [currentState, action_n, rew_n, nextState]
        #     trajectory.append(np.array(currentTimeStepInfo))
        #
        #
        #     episode_step += 1
        #     terminal = (episode_step >= arglist.max_episode_len)
        #     # collect experience
        #     for i, agent in enumerate(trainers):
        #         agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
        #     obs_n = new_obs_n
        #
        #     for i, rew in enumerate(rew_n):
        #         episode_rewards[-1] += rew
        #         agent_rewards[i][-1] += rew
        #
        #     if terminal:
        #         obs_n = env.reset()
        #         episode_step = 0
        #         episode_rewards.append(0)
        #         for a in agent_rewards:
        #             a.append(0)
        #         trajectoryPath = os.path.join(dirName, '..', 'trajectoryFull.pickle')
        #         saveToPickle(trajectory, trajectoryPath)
        #
        #         break
        #
        #
        #     train_step += 1
        #
        #     if arglist.display:
        #         time.sleep(0.1)
        #         env.render()
        #         continue


if __name__ == '__main__':
    main()
