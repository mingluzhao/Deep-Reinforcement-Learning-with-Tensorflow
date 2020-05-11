from gym.utils import seeding
import numpy as np
import random
import math

class TransitGymMountCarContinuous:
    def __init__(self):
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.power = 0.0015

    def __call__(self, state, action):
        position = state[0]
        velocity = state[1]

        force = min(max(action[0], self.min_action), self.max_action)
        velocity += force * self.power - 0.0025 * math.cos(3 * position)
        if (velocity > self.max_speed): velocity = self.max_speed
        if (velocity < -self.max_speed): velocity = -self.max_speed
        position += velocity
        if (position > self.max_position): position = self.max_position
        if (position < self.min_position): position = self.min_position
        if (position == self.min_position and velocity < 0): velocity = 0

        nextState = np.array([position, velocity])
        return nextState


class RewardMountCarContin:
    def __init__(self, isTerminal):
        self.isTerminal = isTerminal

    def __call__(self, state, action, nextState):
        done = self.isTerminal(nextState)
        reward = 100.0 if done else 0
        reward -= math.pow(action[0], 2) * 0.1

        return reward


class IsTerminalMountCarContin:
    def __init__(self):
        self.goal_position = 0.45
        self.goal_velocity = 0

    def __call__(self, state):
        position = state[0]
        velocity = state[1]
        done = bool(
            position >= self.goal_position and velocity >= self.goal_velocity
        )
        return done


class ResetMountCarContin:
    def __init__(self, seed = None, low = -0.6, high = -0.4):
        self.seed = seed
        self.low = low
        self.high = high

    def __call__(self):
        if self.seed is not None:
            np_random, seed = seeding.np_random(self.seed)
            state = np.array([np_random.uniform(self.low, self.high), 0])
        else:
            state = np.array([random.uniform(self.low, self.high), 0])
        return state


class VisualizeMountCarContin:
    def __init__(self):
        self.min_position = -1.2
        self.max_position = 0.6
        self._height = lambda xs: np.sin(3 * xs)*.45+.55
        self.goal_position = 0.45
        self.viewer = None

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
                frontwheel.add_attr(
                    rendering.Transform(translation=(carwidth / 4, clearance))
                )
                frontwheel.add_attr(self.cartrans)
                self.viewer.add_geom(frontwheel)
                backwheel = rendering.make_circle(carheight / 2.5)
                backwheel.add_attr(
                    rendering.Transform(translation=(-carwidth / 4, clearance))
                )
                backwheel.add_attr(self.cartrans)
                backwheel.set_color(.5, .5, .5)
                self.viewer.add_geom(backwheel)
                flagx = (self.goal_position - self.min_position) * scale
                flagy1 = self._height(self.goal_position) * scale
                flagy2 = flagy1 + 50
                flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
                self.viewer.add_geom(flagpole)
                flag = rendering.FilledPolygon(
                    [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
                )
                flag.set_color(.8, .8, 0)
                self.viewer.add_geom(flag)

            pos = state[0]
            self.cartrans.set_translation(
                (pos - self.min_position) * scale, self._height(pos) * scale
            )
            self.cartrans.set_rotation(math.cos(3 * pos))
            self.viewer.render(return_rgb_array=mode == 'rgb_array')

        if self.viewer:
            self.viewer.close()
            self.viewer = None

        return