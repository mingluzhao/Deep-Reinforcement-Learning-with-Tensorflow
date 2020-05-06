import os

DIRNAME = os.path.dirname(__file__)
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from src.ddpg import *
from functionTools.trajectory import *
from functionTools.loadSaveModel import *
from environment.gymEnv.continousMountainCarEnv import *
import gym
os.environ['KMP_DUPLICATE_LIB_OK']='True'


maxEpisode = 150
maxTimeStep = 1000
learningRateActor = 1e-2    # learning rate for actor
learningRateCritic = 5e-3  # learning rate for critic
gamma = 0.99     # reward discount
tau=0.001
bufferSize = 100000
minibatchSize = 64

seed = 1

ENV_NAME = 'MountainCarContinuous-v0'
env = gym.make(ENV_NAME)
env = env.unwrapped

class ActWithoutNoise:
    def __init__(self, actByPolicyTrain, actorModel):
        self.actByPolicyTrain = actByPolicyTrain
        self.actorModel = actorModel

    def __call__(self, states):
        stateBatch = np.asarray(states).reshape(1, -1)
        action = self.actByPolicyTrain(self.actorModel, stateBatch)[0]
        return action

def main():
    stateDim = env.observation_space.shape[0]
    actionDim = env.action_space.shape[0]
    actionHigh = env.action_space.high
    actionLow = env.action_space.low
    actionBound = (actionHigh - actionLow)/2

    buildActorModel = BuildActorModel(stateDim, actionDim, actionBound)
    actorLayerWidths = [256, 128]
    actorWriter, actorModel = buildActorModel(actorLayerWidths)

    dirName = os.path.dirname(__file__)
    actorModelPath = os.path.join(dirName, '..', 'trainedDDPGModels', 'Eps=300_High=0.4_actorModel=0_batch=64_env=MountainCarContinuous-v0_gam=0.99_lrActor=0.01_lrCritic=0.005_noiseVar=2_resetLow=0_timeStep=1000_varDiscout=0.99995.ckpt')
    restoreVariables(actorModel, actorModelPath)
    policy = ActWithoutNoise(actByPolicyTrain, actorModel)

    isTerminal = IsTerminalMountCarContin()
    reset = ResetMountCarContin(seed = 1, low= -0.8, high = -0.6)

    transit = TransitGymMountCarContinuous()

    maxRunningSteps = 20000        # max possible length of the trajectory/episode
    sampleTrajectory = SampleTrajectoryWithActions(maxRunningSteps, transit, isTerminal, reset)
    trajectory = sampleTrajectory(policy) # trajectory only contains states, need to add actions for each state

    # demo& plot
    showDemo = True
    if showDemo:
        visualize = VisualizeMountCarContin()
        visualize(trajectory)

if __name__ == '__main__':
    main()
