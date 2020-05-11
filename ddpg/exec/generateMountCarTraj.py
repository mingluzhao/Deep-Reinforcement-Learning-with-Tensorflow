import os
import sys
import gym
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))
sys.path.append(os.path.join(dirName, '..'))

from src.ddpg import actByPolicyTrain, BuildActorModel
from functionTools.trajectory import SampleTrajectory
from functionTools.loadSaveModel import restoreVariables
from environment.gymEnv.continousMountainCarEnv import IsTerminalMountCarContin, TransitGymMountCarContinuous, \
    RewardMountCarContin, ResetMountCarContin, VisualizeMountCarContin
from src.policy import ActDDPGOneStep

ENV_NAME = 'MountainCarContinuous-v0'
env = gym.make(ENV_NAME)
env = env.unwrapped


def main():
    stateDim = env.observation_space.shape[0]
    actionDim = env.action_space.shape[0]
    actionHigh = env.action_space.high
    actionLow = env.action_space.low
    actionBound = (actionHigh - actionLow)/2

    buildActorModel = BuildActorModel(stateDim, actionDim, actionBound)
    actorLayerWidths = [30]
    actorWriter, actorModel = buildActorModel(actorLayerWidths)

    dirName = os.path.dirname(__file__)
    actorModelPath = os.path.join(dirName, '..', 'trainedDDPGModels', 'Eps=300_High=0.4_actorModel=0_batch=128_env=MountainCarContinuous-v0_gam=0.9_lrActor=0.001_lrCritic=0.001_noiseVar=1_resetLow=-1_timeStep=2000_varDiscout=0.99995.ckpt')

    restoreVariables(actorModel, actorModelPath)
    policy = ActDDPGOneStep(actionLow, actionHigh, actByPolicyTrain, actorModel, getNoise = None)

    isTerminal = IsTerminalMountCarContin()
    reset = ResetMountCarContin(seed= None, low= -1, high= 0.4)

    transit = TransitGymMountCarContinuous()
    rewardFunc = RewardMountCarContin(isTerminal)


    for i in range(20):
        maxRunningSteps = 2000
        sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, rewardFunc, reset)
        trajectory = sampleTrajectory(policy)

        # plots& plot
        showDemo = True
        if showDemo:
            visualize = VisualizeMountCarContin()
            visualize(trajectory)


if __name__ == '__main__':
    main()
