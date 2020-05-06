import os

DIRNAME = os.path.dirname(__file__)
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from src.ddpg import *
from functionTools.trajectory import *
from functionTools.loadSaveModel import *
from environment.gymEnv.pendulumEnv import *
import gym
os.environ['KMP_DUPLICATE_LIB_OK']='True'


maxEpisode = 200
maxTimeStep = 200
learningRateActor = 0.001    # learning rate for actor
learningRateCritic = 0.001    # learning rate for critic
gamma = 0.9     # reward discount
tau=0.01
bufferSize = 20000
minibatchSize = 128
seed = None

ENV_NAME = 'Pendulum-v0'
env = gym.make(ENV_NAME)
env = env.unwrapped

class ActWithoutNoise:
    def __init__(self, actByPolicyTrain, actorModel, observe = None):
        self.actByPolicyTrain = actByPolicyTrain
        self.actorModel = actorModel
        self.observe = observe

    def __call__(self, states):
        states = self.observe(states) if self.observe is not None else states
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
    actorLayerWidths = [30]
    actorWriter, actorModel = buildActorModel(actorLayerWidths)

    dirName = os.path.dirname(__file__)
    actorModelPath = os.path.join(dirName, '..', 'trainedDDPGModels', 'actorModel=0_gamma=0.9_learningRateActor=0.001_learningRateCritic=0.001_maxEpisode=200_maxTimeStep=200_minibatchSize=128_noiseVar=3_varDiscout=0.9995.ckpt')
    restoreVariables(actorModel, actorModelPath)
    policy = ActWithoutNoise(actByPolicyTrain, actorModel, observe)

    isTerminal = isTerminalGymPendulum
    reset = ResetGymPendulum(seed, observe)
    transit = TransitGymPendulum()

    maxRunningSteps = 10000        # max possible length of the trajectory/episode
    sampleTrajectory = SampleTrajectoryWithActions(maxRunningSteps, transit, isTerminal, reset)
    trajectory = sampleTrajectory(policy) # trajectory only contains states, need to add actions for each state

    # demo& plot
    showDemo = True
    if showDemo:
        visualize = VisualizeGymPendulum()
        visualize(trajectory)

if __name__ == '__main__':
    main()
