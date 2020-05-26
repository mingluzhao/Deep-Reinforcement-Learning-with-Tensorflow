import os
import sys
import gym
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))
sys.path.append(os.path.join(dirName, '..'))

from dqn.src.dqn import BuildModel, getTrainQValue
from dqn.src.policy import ActGreedyByModel
from functionTools.trajectory import SampleTrajectory
from functionTools.loadSaveModel import restoreVariables
from environment.gymEnv.discreteMountainCarEnv import TransitMountCarDiscrete, IsTerminalMountCarDiscrete, \
    rewardMountCarDiscrete, ResetMountCarDiscrete, VisualizeMountCarDiscrete

ENV_NAME = 'MountainCar-v0'
env = gym.make(ENV_NAME)
env = env.unwrapped


def main():
    stateDim = env.observation_space.shape[0]
    actionDim = env.action_space.n
    buildModel = BuildModel(stateDim, actionDim)
    layersWidths = [30]
    writer, model = buildModel(layersWidths)

    dirName = os.path.dirname(__file__)
    modelPath = os.path.join(dirName, '..', 'trainedDQNModels', 'epsilonIncrease=0.0002_epsilonMin=0_gamma=0.9_learningRate=0.001_maxEpisode=1000_maxTimeStep=2000_minibatchSize=128.ckpt')
    restoreVariables(model, modelPath)
    policy = lambda state: [ActGreedyByModel(getTrainQValue, model)(state)]

    isTerminal = IsTerminalMountCarDiscrete()
    transit = TransitMountCarDiscrete()
    getReward = rewardMountCarDiscrete
    reset = ResetMountCarDiscrete(seed = None, low= -1, high= 0.4)

    for i in range(20):
        maxRunningSteps = 2000
        sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, getReward, reset)
        trajectory = sampleTrajectory(policy)

        # plots& plot
        showDemo = True
        if showDemo:
            visualize = VisualizeMountCarDiscrete()
            visualize(trajectory)


if __name__ == '__main__':
    main()
