import math

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

class TransitMountCarDiscrete:
    def __init__(self):
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07

        self.force = 0.001
        self.gravity = 0.0025
        self.action_space = spaces.Discrete(3)

    def __call__(self, state, action):
        action = action[0]
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        position, velocity = state
        velocity += (action - 1) * self.force + math.cos(3 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if (position == self.min_position and velocity < 0): velocity = 0

        nextState = np.array([position, velocity])
        return nextState


def rewardMountCarDiscrete(state, action, nextState):
    return -1.0

def modifiedRewardMountCarDiscrete(state, action):
    position, velocity = state
    reward = abs(position - (-0.5))
    return reward

class IsTerminalMountCarDiscrete:
    def __init__(self):
        self.goal_position = 0.5
        self.goal_velocity = 0

    def __call__(self, state):
        position, velocity = state
        done = bool(position >= self.goal_position and velocity >= self.goal_velocity)
        return done


class ResetMountCarDiscrete:
    def __init__(self, seed, low =-0.6, high=-0.4):
        self.seed = seed
        self.low = low
        self.high = high

    def __call__(self):
        np_random, seed = seeding.np_random(self.seed)
        # state = np.array([np_random.uniform(low=-0.6, high=-0.4), 0])
        state = np.array([np_random.uniform(self.low, self.high), 0])
        return np.array(state)


class VisualizeMountCarDiscrete:
    def __init__(self):
        self._height = lambda xs: np.sin(3 * xs) * .45 + .55
        self.min_position = -1.2
        self.max_position = 0.6
        self.goal_position = 0.5
        self.viewer = None
        self.get_keys_to_action = lambda : {(): 1, (276,): 0, (275,): 2, (275, 276): 1}

    def __call__(self, trajectory):
        mode = 'human'
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width / world_width
        carwidth = 40
        carheight = 20

        for timeStep in range(len(trajectory)):
            state = trajectory[timeStep][0]

            if self.viewer is None:
                from gym.envs.classic_control import rendering
                self.viewer = rendering.Viewer(screen_width, screen_height)
                xs = np.linspace(self.min_position, self.max_position, 100)
                ys = self._height(xs)
                xys = list(zip((xs - self.min_position) * scale, ys * scale))

                self.track = rendering.make_polyline(xys)
                self.track.set_linewidth(4)
                self.viewer.add_geom(self.track)

                clearance = 10

                l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
                car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                car.add_attr(rendering.Transform(translation=(0, clearance)))
                self.cartrans = rendering.Transform()
                car.add_attr(self.cartrans)
                self.viewer.add_geom(car)
                frontwheel = rendering.make_circle(carheight / 2.5)
                frontwheel.set_color(.5, .5, .5)
                frontwheel.add_attr(rendering.Transform(translation=(carwidth / 4, clearance)))
                frontwheel.add_attr(self.cartrans)
                self.viewer.add_geom(frontwheel)
                backwheel = rendering.make_circle(carheight / 2.5)
                backwheel.add_attr(rendering.Transform(translation=(-carwidth / 4, clearance)))
                backwheel.add_attr(self.cartrans)
                backwheel.set_color(.5, .5, .5)
                self.viewer.add_geom(backwheel)
                flagx = (self.goal_position - self.min_position) * scale
                flagy1 = self._height(self.goal_position) * scale
                flagy2 = flagy1 + 50
                flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
                self.viewer.add_geom(flagpole)
                flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)])
                flag.set_color(.8, .8, 0)
                self.viewer.add_geom(flag)

            pos = state[0]
            self.cartrans.set_translation((pos - self.min_position) * scale, self._height(pos) * scale)
            self.cartrans.set_rotation(math.cos(3 * pos))

            self.viewer.render(return_rgb_array=mode == 'rgb_array')


        if self.viewer:
            self.viewer.close()
            self.viewer = None

        return