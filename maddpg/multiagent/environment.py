import gym
from gym import spaces
import numpy as np

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True):

        self.world = world
        self.agents = self.world.policy_agents
        # set required vectorized gym env property
        self.n = len(world.policy_agents) # 4
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback # None
        self.done_callback = done_callback # None
        # environment parameters
        self.discrete_action_space = True
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector

        self.time = 0

        # configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            total_action_space = []
            # physical action space
            u_action_space = spaces.Discrete(world.dim_p * 2 + 1) # 5
            total_action_space.append(u_action_space)
            self.action_space.append(total_action_space[0])
            # observation space
            obs_dim = len(observation_callback(agent, self.world)) # 16 16 16 14
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            agent.action.c = np.zeros(self.world.dim_c)

        # rendering
        self.viewer = None
        self._reset_render()

    def step(self, action_n): # takes actions of all agents
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        self.agents = self.world.policy_agents
        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
        # advance world state
        self.world.step()
        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))

            info_n['n'].append(self._get_info(agent))

        return obs_n, reward_n, done_n, info_n

    def reset(self):
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        return {}
    # get observation for a particular agent
    def _get_obs(self, agent):
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        return False
    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        action = [action]

        agent.action.u[0] += action[0][1] - action[0][2]
        agent.action.u[1] += action[0][3] - action[0][4]

        sensitivity = 5.0
        agent.action.u *= sensitivity
        action = action[1:]

        # make sure we used all elements of action
        assert len(action) == 0

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    # render environment
    # render environment
    # def render(self, mode='human'):
    #     from maddpgAlgor.multiagent import rendering
    #     self.viewers = rendering.Viewer(700,700)
    #
    #     # create rendering geometry
    #     if self.render_geoms is None:
    #         # import rendering only if we need it (and don't import for headless machines)
    #         #from gym.envs.classic_control import rendering
    #         from maddpgAlgor.multiagent import rendering
    #         self.render_geoms = []
    #         self.render_geoms_xform = []
    #         for entity in self.world.entities:
    #             geom = rendering.make_circle(entity.size)
    #             xform = rendering.Transform()
    #             if 'agent' in entity.name:
    #                 geom.set_color(*entity.color, alpha=0.5)
    #             else:
    #                 geom.set_color(*entity.color)
    #             geom.add_attr(xform)
    #             self.render_geoms.append(geom)
    #             self.render_geoms_xform.append(xform)
    #
    #         # add geoms to viewer
    #         self.viewers.geoms = []
    #         for geom in self.render_geoms:
    #             self.viewers.add_geom(geom)
    #
    #     results = []
    #     # update bounds to center around agent
    #     cam_range = 1
    #     pos = np.zeros(self.world.dim_p)
    #     self.viewers.set_bounds(pos[0]-cam_range,pos[0]+cam_range,pos[1]-cam_range,pos[1]+cam_range)
    #     # update geometry positions
    #     for e, entity in enumerate(self.world.entities):
    #         self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
    #     # render to display or array
    #     results.append(self.viewers.render(return_rgb_array = mode=='rgb_array'))
    #
    #     return results


    # create receptor field locations in local coordinate frame

    def render(self, mode='human'):
        from maddpg.multiagent import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(700,700)

        # create rendering geometry
        if self.render_geoms is None:
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.world.entities:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)
                else:
                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            self.viewer.geoms = []
            for geom in self.render_geoms:
                self.viewer.add_geom(geom)

        results = []
        # update bounds to center around agent
        cam_range = 1
        pos = np.zeros(self.world.dim_p)
        self.viewer.set_bounds(pos[0]-cam_range, pos[0]+cam_range, pos[1]-cam_range, pos[1]+cam_range)
        # update geometry positions

        currentState = []

        for e, entity in enumerate(self.world.entities): # 4
            self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
            currentState.append(list(entity.state.p_pos) + list(entity.state.p_vel))
        # print(currentState)
        # render to display or array

        results.append(self.viewer.render(return_rgb_array = mode=='rgb_array'))

        return currentState

# trajectory[0] = [allAgentsStates, allAgentsActions, allAgentsRewards, allAgentsNextState],
# agentState = [agentPos1, agentPos2, agentVel1, agentVel2]