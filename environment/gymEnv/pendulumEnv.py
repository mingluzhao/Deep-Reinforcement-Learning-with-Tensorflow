from gym.utils import seeding
import numpy as np
from os import path


class TransitGymPendulum:
    def __init__(self, processAction = None):
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        self.g = 10.0
        self.m = 1.
        self.l = 1.
        self.processAction = processAction

    def __call__(self, state, action):
        action = self.processAction(action) if self.processAction is not None else action
        th, thdot = state  # th := theta
        action = np.clip(action, -self.max_torque, self.max_torque)[0]

        newthdot = thdot + (-3 * self.g / (2 * self.l) * np.sin(th + np.pi) + 3. / (self.m * self.l ** 2) * action) * self.dt
        newth = th + newthdot * self.dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        nextState = np.array([newth, newthdot])
        return nextState


class RewardGymPendulum:
    def __init__(self, angle_normalize, processAction = None):
        self.angle_normalize = angle_normalize
        self.max_torque = 2.
        self.processAction = processAction

    def __call__(self, state, action, nextState):
        action = self.processAction(action) if self.processAction is not None else action
        action = np.clip(action, -self.max_torque, self.max_torque)[0]
        th, thdot = state  # th := theta
        costs = self.angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (action ** 2)
        reward = -costs
        return reward


def isTerminalGymPendulum(stateInfo):
    return False


def observe(state):
    theta, thetadot = state
    return np.array([np.cos(theta), np.sin(theta), thetadot])


class ResetGymPendulum:
    def __init__(self, seed):
        self.seed = seed

    def __call__(self):
        np_random, seed = seeding.np_random(self.seed )
        high = np.array([np.pi, 1])
        state = np_random.uniform(low=-high, high=high)
        return state


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)


class ProcessDiscretePendulumAction:
    def __init__(self, discreteActionSpaceSize):
        self.actionSpace = discreteActionSpaceSize

    def __call__(self, action):
        floatAction = (np.array(action) - (self.actionSpace - 1) / 2) / ((self.actionSpace - 1) / 4)
        # [-2 ~ 2] float actions
        return np.array(floatAction)


class VisualizeGymPendulum:
    def __init__(self):
        self.viewer = None
        self.max_torque = 2.

    def __call__(self, trajectory):
        mode = 'human'
        for timeStep in range(len(trajectory)):
            state = trajectory[timeStep][0]
            action = trajectory[timeStep][1]
            action = np.clip(action, -self.max_torque, self.max_torque)[0]

            if self.viewer is None:
                from gym.envs.classic_control import rendering
                self.viewer = rendering.Viewer(500, 500)
                self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
                rod = rendering.make_capsule(1, .2)
                rod.set_color(.8, .3, .3)
                self.pole_transform = rendering.Transform()
                rod.add_attr(self.pole_transform)
                self.viewer.add_geom(rod)
                axle = rendering.make_circle(.05)
                axle.set_color(0, 0, 0)
                self.viewer.add_geom(axle)
                fname = path.join(path.dirname(__file__), "assets/clockwise.png")
                self.img = rendering.Image(fname, 1., 1.)
                self.imgtrans = rendering.Transform()
                self.img.add_attr(self.imgtrans)

            self.viewer.add_onetime(self.img)
            self.pole_transform.set_rotation(state[0] + np.pi / 2)
            self.imgtrans.scale = (-action/ 2, np.abs(action) / 2)

            self.viewer.render(return_rgb_array=mode == 'rgb_array')

        if self.viewer:
            self.viewer.close()
            self.viewer = None
        return
