import numpy as np
from maddpgAlgor.maddpg_openai.multiagent.core import World, Agent, Landmark
from maddpgAlgor.maddpg_openai.multiagent.scenario import BaseScenario
# predator-prey environment
# good agents are prey, adversaries are predators, now set good = 0, bad = 1



# state = [position, velocity, comm]

class ResetMultiAgentChasing:
    def __init__(self):
        self.positionDimension = 2
        self.commChanelDim = 2 # communication channel dimensionality

    def __call__(self):
        # set random initial states
        getRandomPos = lambda: np.random.uniform(-1, +1, self.positionDimension)
        getRandomVel = lambda: np.zeros(self.positionDimension)
        getRandomComm = lambda: np.zeros(self.commChanelDim)

        position = getRandomPos()
        velocity = getRandomVel()
        comm = getRandomComm()
        return [position, velocity, comm]


class RewardGymChasing:
    def __init__(self, isTerminal):
        self.isTerminal = isTerminal

    def __call__(self, predatorState, preyState, action):
        predatorPos = predatorState[0]
        preyPos = preyState[0]
        reward = -0.1 * np.sqrt(np.sum(np.square(preyPos - predatorPos))) # L2norm from the sheep
        done = self.isTerminal(predatorState)
        if done:
            reward += 10
        return reward

class IsTerminal:
    def __init__(self):
        self.predatorSize = 0.075
        self.preySize = 0.05

    def __call__(self, predatorState, preyState):
        predatorPos = predatorState[0]
        preyPos = preyState[0]
        delta_pos = predatorPos - preyPos
        dist = np.sqrt(np.sum(np.square(delta_pos))) # L2 norm
        dist_min = self.predatorSize + self.preySize

        return True if dist < dist_min else False


class TransitMultiAgentChasing:
    def __init__(self):


    def __call__(self, state, actions):



class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_good_agents = 1
        num_adversaries = 3
        num_agents = num_adversaries + num_good_agents
        num_landmarks = 2
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.075 if agent.adversary else 0.05
            agent.accel = 3.0 if agent.adversary else 4.0
            #agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = 1.0 if agent.adversary else 1.3
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False
        # make initial conditions
        self.reset_world(world)
        return world


    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])





