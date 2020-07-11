import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class InvertedDoublePendulumEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'inverted_double_pendulum.xml', 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        x, _, y = self.sim.data.site_xpos[0]
        dist_penalty = 0.01 * x ** 2 + (y - 2) ** 2
        v1, v2 = self.sim.data.qvel[1:3]
        vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2
        alive_bonus = 10
        r = alive_bonus - dist_penalty - vel_penalty
        done = bool(y <= 1)
        return ob, r, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos[:1],  # cart x pos
            np.sin(self.sim.data.qpos[1:]),  # link angles
            np.cos(self.sim.data.qpos[1:]),
            np.clip(self.sim.data.qvel, -10, 10),
            np.clip(self.sim.data.qfrc_constraint, -10, 10)
        ]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.randn(self.model.nv) * .1
        )
        return self._get_obs()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 0.5
        v.cam.lookat[2] = 0.12250000000000005  # v.model.stat.center[2]

import numpy as np
import mujoco_py as mujoco

class RewardFunctionTerminalPenalty():
    def __init__(self, aliveBouns, deathPenalty, isTerminal):
        self.aliveBouns = aliveBouns
        self.deathPenalty = deathPenalty
        self.isTerminal = isTerminal
    def __call__(self, state, action):
        reward = self.aliveBouns
        if self.isTerminal(state):
            reward = self.deathPenalty
        return reward

class RewardFunction():
    def __init__(self, aliveBouns):
        self.aliveBouns = aliveBouns
    def __call__(self, state, action):
        reward = self.aliveBouns
        return reward

def euclideanDistance(pos1, pos2):
    return np.sqrt(np.sum(np.square(pos1 - pos2)))

class RewardFunctionCompete():
    def __init__(self, aliveBouns, catchReward, disDiscountFactor, minXDis):
        self.aliveBouns = aliveBouns
        self.catchReward = catchReward
        self.disDiscountFactor = disDiscountFactor
        self.minXDis = minXDis
    def __call__(self, state, action):
        pos0 = state[0][2:4]
        pos1 = state[1][2:4]
        distance = euclideanDistance(pos0, pos1)

        if distance <= 2 * self.minXDis:
            catchReward = self.catchReward
        else:
            catchReward = 0

        distanceReward = self.disDiscountFactor * distance

        reward = np.array([distanceReward - catchReward, -distanceReward + catchReward])
        # print("reward", reward)
        return reward


class InvDblPendulumRewardFunction:
    def __init__(self, aliveBonus, deathPenalty, isTerminal):
        self.aliveBonus = aliveBonus
        self.deathPenalty = deathPenalty
        self.isTerminal = isTerminal
        modelName = 'inverted_double_pendulum'
        model = mujoco.load_model_from_path('xmls/' + modelName + '.xml')
        self.simulation = mujoco.MjSim(model)
        self.numQPos = len(self.simulation.data.qpos)
        self.numQVel = len(self.simulation.data.qvel)

    def __call__(self, state, action, nextState):
        reward = self.aliveBonus
        if self.isTerminal(nextState):
            reward = self.deathPenalty

        oldQPos = state[0: self.numQPos]
        oldQVel = state[self.numQPos: self.numQPos + self.numQVel]
        self.simulation.data.qpos[:] = oldQPos
        self.simulation.data.qvel[:] = oldQVel
        self.simulation.data.ctrl[:] = action
        self.simulation.forward()

        x, _, y = self.simulation.data.site_xpos[0]
        distPenalty = 0.01 * x ** 2 + (y - 2) ** 2

        v1, v2 = self.simulation.data.qvel[1:3]
        velPenalty = 1e-3 * v1 ** 2 + 5e-3 * v2 ** 2

        return reward - distPenalty - velPenalty


class Reset():
    def __init__(self, modelName, qPosInitNoise, qVelInitNoise):
        model = mujoco.load_model_from_path('xmls/' + modelName + '.xml')
        self.simulation = mujoco.MjSim(model)
        self.qPosInitNoise = qPosInitNoise
        self.qVelInitNoise = qVelInitNoise
    def __call__(self):
        qPos = self.simulation.data.qpos + np.random.uniform(low = -self.qPosInitNoise, high = self.qPosInitNoise, size = len(self.simulation.data.qpos))
        qVel = self.simulation.data.qvel + np.random.uniform(low = -self.qVelInitNoise, high = self.qVelInitNoise, size = len(self.simulation.data.qvel))
        startState = np.concatenate([qPos, qVel])
        return startState

class TransitionFunction():
    def __init__(self, modelName, renderOn):
        model = mujoco.load_model_from_path('xmls/' + modelName + '.xml')
        self.simulation = mujoco.MjSim(model)
        self.numQPos = len(self.simulation.data.qpos)
        self.numQVel = len(self.simulation.data.qvel)
        self.renderOn = renderOn
        if self.renderOn:
            self.viewer = mujoco.MjViewer(self.simulation)
    def __call__(self, oldState, action, numSimulationFrames = 1):
        oldQPos = oldState[0 : self.numQPos]
        oldQVel = oldState[self.numQPos : self.numQPos + self.numQVel]
        self.simulation.data.qpos[:] = oldQPos
        self.simulation.data.qvel[:] = oldQVel
        self.simulation.data.ctrl[:] = action

        for i in range(numSimulationFrames):
            self.simulation.step()
            if self.renderOn:
                self.viewer.render()
        newQPos, newQVel = self.simulation.data.qpos, self.simulation.data.qvel
        newState = np.concatenate([newQPos, newQVel])
        #print("From\n\t qpos: {}; qvel: {}\nTo\n\t qpos: {}; qvel: {}\n".format(oldQPos, oldQVel, newQPos, newQVel))
        return newState

class InvDblPendulumIsTerminal():
    def __init__(self, minHeight):
        self.minHeight = minHeight
        self.model = mujoco.load_model_from_path('xmls/inverted_double_pendulum.xml')
        self.simulation = mujoco.MjSim(self.model)
        self.numQPos = len(self.simulation.data.qpos)
        self.numQVel = len(self.simulation.data.qvel)
    def __call__(self, state):
        oldQPos = state[0: self.numQPos]
        oldQVel = state[self.numQPos: self.numQPos + self.numQVel]
        self.simulation.data.qpos[:] = oldQPos
        self.simulation.data.qvel[:] = oldQVel
        self.simulation.forward()

        x, _, y = self.simulation.data.site_xpos[0]
        terminal = bool(y <= self.minHeight)
        return terminal


if __name__ == '__main__':
    modelName = "inverted_double_pendulum"
    qPosInitNoise = 0.001
    qVelInitNoise = 0.001
    minHeight = -100
    reset = Reset(modelName, qPosInitNoise, qVelInitNoise)
    transition = TransitionFunction(modelName, renderOn=True)
    isTerminal = InvDblPendulumIsTerminal(minHeight)

    aliveBonus = 10
    deathPenalty = 10
    rewardFunc = reward.InvDblPendulumRewardFunction(aliveBonus, deathPenalty, isTerminal)

    episodeLen = 1000
    state = reset()
    for i in range(episodeLen):
        if isTerminal(state):
            break
        action = 0.05*(np.random.randn() - 0.5)
        if i % 50 == 0:
            r = rewardFunc(state, action)
            print(r)
        nextState = transition(state, action)
        state = nextState