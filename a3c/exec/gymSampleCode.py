import gym
import numpy as np
import time

"""
A simple sample code for running gym environment:
Please refer to the website for a comprehensive list of available environments:
http://gym.openai.com/envs/#classic_control

To install gym, please refer to the page:
http://gym.openai.com/docs/

For detailed implementation of the environments:
https://github.com/openai/gym/tree/master/gym/envs

Some environments (like the one in this code sample) required a mujoco license, 
and you can apply for a free 30-day license here:
https://www.roboti.us/license.html

"""

def main():
    game = 'InvertedDoublePendulum-v2'
    env = gym.make(game)
    stateDim = env.observation_space.shape[0]
    actionDim = env.action_space.shape[0]
    actionLow = env.action_space.low[0]
    actionHigh = env.action_space.high[0]
    print('state dimension {}, action dimension {}, action range from {} to {}'.format(stateDim, actionDim, actionLow, actionHigh))

    # reset the environment
    env.reset()
    for i in range(100):
        # a random action just for illustrating purposes
        action = np.random.standard_normal(actionDim) * actionHigh

        # with each call of env.step(action), env will update itself and output step information
        nextState, reward, done, info = env.step(action)

        # for visualizing the current state
        env.render()
        time.sleep(.05)

if __name__ == '__main__':
    main()