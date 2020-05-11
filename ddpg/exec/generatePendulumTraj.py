import os
import sys
import gym
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))
sys.path.append(os.path.join(dirName, '..'))

from src.ddpg import actByPolicyTrain, BuildActorModel
from src.policy import ActDDPGOneStep
from functionTools.trajectory import SampleTrajectory
from functionTools.loadSaveModel import restoreVariables
from environment.gymEnv.pendulumEnv import TransitGymPendulum, RewardGymPendulum, isTerminalGymPendulum, \
    observe, angle_normalize, VisualizeGymPendulum, ResetGymPendulum

seed = 1
ENV_NAME = 'Pendulum-v0'
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

    actorModelPath = os.path.join(dirName, '..', 'trainedDDPGModels', 'Eps=200_actorModel=0_batch=128_env=Pendulum-v0_gam=0.9_lrActor=0.001_lrCritic=0.001_noiseVar=3_timeStep=200_varDiscout=0.9995.ckpt')
    restoreVariables(actorModel, actorModelPath)
    actOneStep = ActDDPGOneStep(actionLow, actionHigh, actByPolicyTrain, actorModel, getNoise = None)
    policy = lambda state: actOneStep(observe(state))

    isTerminal = isTerminalGymPendulum
    reset = ResetGymPendulum(seed)
    transit = TransitGymPendulum()
    rewardFunc = RewardGymPendulum(angle_normalize)

    for i in range(10):
        maxRunningSteps = 200
        sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, rewardFunc, reset)
        trajectory = sampleTrajectory(policy)

        # plots& plot
        showDemo = True
        if showDemo:
            visualize = VisualizeGymPendulum()
            visualize(trajectory)


if __name__ == '__main__':
    main()
