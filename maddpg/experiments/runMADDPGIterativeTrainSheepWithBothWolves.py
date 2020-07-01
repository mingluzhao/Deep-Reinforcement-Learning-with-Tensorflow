import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import numpy as np
import json

from maddpg.maddpgAlgor.trainer.myMADDPG import BuildMADDPGModels, TrainCritic, TrainActor, TrainCriticBySASR, \
    TrainActorFromSA, TrainMADDPGModelsWithBuffer, ActOneStep, actByPolicyTrainNoisy, actByPolicyTargetNoisyForNextState
from RLframework.RLrun_MultiAgent import UpdateParameters, SampleOneStep, SampleFromMemory,\
    RunTimeStep, RunEpisode, RunAlgorithm, getBuffer, SaveModel, StartLearn
from functionTools.loadSaveModel import saveVariables
from environment.chasingEnv.multiAgentEnv import TransitMultiAgentChasing, ApplyActionForce, ApplyEnvironForce, \
    ResetMultiAgentChasing, ReshapeAction, RewardSheep, RewardWolf, Observe, GetCollisionForce, IntegrateState, \
    IsCollision, PunishForOutOfBound, getPosFromAgentState, getVelFromAgentState
from environment.chasingEnv.multiAgentEnvWithIndividReward import RewardWolfIndividual

# fixed training parameters
maxEpisode = 120000
learningRateActor = 0.01#
learningRateCritic = 0.01#
gamma = 0.95 #
tau=0.01 #
bufferSize = 1e6#
minibatchSize = 1024#


# arguments: numWolves numSheeps numBlocks saveAllmodels = True or False

class RunAlgorithmWithIterSheep:
    def __init__(self, runEpisodeIndivid, runEpisodeShared, maxEpisode, saveModelsIndivid, saveModelsShared, numAgents = 1, printEpsFrequency = 1000):
        self.runEpisodeIndivid = runEpisodeIndivid
        self.runEpisodeShared = runEpisodeShared
        self.maxEpisode = maxEpisode
        self.saveModelsIndivid = saveModelsIndivid
        self.saveModelsShared = saveModelsShared
        self.numAgents = numAgents
        self.printEpsFrequency = printEpsFrequency

    def __call__(self, replayBufferShared, replayBufferIndivid):
        episodeRewardList = []
        meanRewardList = []
        lastTimeSpanMeanRewardList = []

        trajectory = []
        agentsEpsRewardList = [list() for agentID in range(self.numAgents)]

        shared = 0
        for episode in range(self.maxEpisode):
            if shared % 10 < 5:
                replayBufferShared, episodeReward, trajectory = self.runEpisodeShared(replayBufferShared, trajectory)
                episodeRewardList.append(np.sum(episodeReward))
                [agentRewardList.append(agentEpsReward) for agentRewardList, agentEpsReward in zip(agentsEpsRewardList, episodeReward)]
                meanRewardList.append(np.mean(episodeRewardList))
                [saveModel() for saveModel in self.saveModelsShared]
            else:
                replayBufferIndivid, episodeReward, trajectory = self.runEpisodeIndivid(replayBufferIndivid, trajectory)
                episodeRewardList.append(np.sum(episodeReward))
                [agentRewardList.append(agentEpsReward) for agentRewardList, agentEpsReward in zip(agentsEpsRewardList, episodeReward)]
                meanRewardList.append(np.mean(episodeRewardList))
                [saveModel() for saveModel in self.saveModelsIndivid]

            if episode % self.printEpsFrequency == 0:
                lastTimeSpanMeanReward = np.mean(episodeRewardList[-self.printEpsFrequency:])
                lastTimeSpanMeanRewardList.append(lastTimeSpanMeanReward)

                print("steps: {} and {}, episodes: {}, last {} eps mean episode reward: {}, agent episode reward: {}".format(
                    len(replayBufferShared), len(replayBufferIndivid), len(episodeRewardList), self.printEpsFrequency, lastTimeSpanMeanReward,
                    [np.mean(rew[-self.printEpsFrequency:]) for rew in agentsEpsRewardList]))

            shared += 1

        return meanRewardList, trajectory

def main():
    debug = 1
    if debug:
        numWolves = 3
        numSheeps = 1
        numBlocks = 2
        saveAllmodels = False
        maxTimeStep = 25
        sheepSpeedMultiplier = 1
        individualRewardWolf = int(False)

    else:
        print(sys.argv)
        condition = json.loads(sys.argv[1])
        numWolves = int(condition['numWolves'])
        numSheeps = int(condition['numSheeps'])
        numBlocks = int(condition['numBlocks'])

        maxTimeStep = int(condition['maxTimeStep'])
        sheepSpeedMultiplier = float(condition['sheepSpeedMultiplier'])
        individualRewardWolf = int(condition['individualRewardWolf'])

        saveAllmodels = False

    print("maddpg: {} wolves, {} sheep, {} blocks, {} episodes with {} steps each eps, sheepSpeed: {}x, wolfIndividualReward: {}, save all models: {}".
          format(numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, individualRewardWolf, str(saveAllmodels)))


    numAgents = numWolves + numSheeps
    numEntities = numAgents + numBlocks
    wolvesID = list(range(numWolves))
    sheepsID = list(range(numWolves, numAgents))
    blocksID = list(range(numAgents, numEntities))

    wolfSize = 0.075
    sheepSize = 0.05
    blockSize = 0.2
    entitiesSizeList = [wolfSize] * numWolves + [sheepSize] * numSheeps + [blockSize] * numBlocks

    wolfMaxSpeed = 1.0
    blockMaxSpeed = None
    sheepMaxSpeedOriginal = 1.3
    sheepMaxSpeed = sheepMaxSpeedOriginal * sheepSpeedMultiplier

    entityMaxSpeedList = [wolfMaxSpeed] * numWolves + [sheepMaxSpeed] * numSheeps + [blockMaxSpeed] * numBlocks
    entitiesMovableList = [True] * numAgents + [False] * numBlocks
    massList = [1.0] * numEntities

    isCollision = IsCollision(getPosFromAgentState)
    punishForOutOfBound = PunishForOutOfBound()
    rewardSheep = RewardSheep(wolvesID, sheepsID, entitiesSizeList, getPosFromAgentState, isCollision,
                              punishForOutOfBound)

    rewardWolfIndivid = RewardWolfIndividual(wolvesID, sheepsID, entitiesSizeList, isCollision)
    rewardWolfShared = RewardWolf(wolvesID, sheepsID, entitiesSizeList, isCollision)

    rewardFuncIndividWolf = lambda state, action, nextState: \
        list(rewardWolfIndivid(state, action, nextState)) + list(rewardSheep(state, action, nextState))
    rewardFuncSharedWolf = lambda state, action, nextState: \
        list(rewardWolfShared(state, action, nextState)) + list(rewardSheep(state, action, nextState))

    reset = ResetMultiAgentChasing(numAgents, numBlocks)
    observeOneAgent = lambda agentID: Observe(agentID, wolvesID, sheepsID, blocksID, getPosFromAgentState,
                                              getVelFromAgentState)
    observe = lambda state: [observeOneAgent(agentID)(state) for agentID in range(numAgents)]

    reshapeAction = ReshapeAction()
    getCollisionForce = GetCollisionForce()
    applyActionForce = ApplyActionForce(wolvesID, sheepsID, entitiesMovableList)
    applyEnvironForce = ApplyEnvironForce(numEntities, entitiesMovableList, entitiesSizeList,
                                          getCollisionForce, getPosFromAgentState)
    integrateState = IntegrateState(numEntities, entitiesMovableList, massList,
                                    entityMaxSpeedList, getVelFromAgentState, getPosFromAgentState)
    transit = TransitMultiAgentChasing(numEntities, reshapeAction, applyActionForce, applyEnvironForce, integrateState)

    isTerminal = lambda state: [False] * numAgents
    initObsForParams = observe(reset())
    obsShape = [initObsForParams[obsID].shape[0] for obsID in range(len(initObsForParams))]

    worldDim = 2
    actionDim = worldDim * 2 + 1

    layerWidth = [128, 128]

#------------ models ------------------------

    buildMADDPGModels = BuildMADDPGModels(actionDim, numAgents, obsShape)
    modelsListShared = [buildMADDPGModels(layerWidth, agentID) for agentID in range(numAgents)]
    sheepModel = [modelsListShared[sheepID] for sheepID in sheepsID]
    modelsListIndivid = [buildMADDPGModels(layerWidth, agentID) for agentID in wolvesID] + sheepModel


    trainCriticBySASR = TrainCriticBySASR(actByPolicyTargetNoisyForNextState, learningRateCritic, gamma)
    trainCritic = TrainCritic(trainCriticBySASR)
    trainActorFromSA = TrainActorFromSA(learningRateActor)
    trainActor = TrainActor(trainActorFromSA)

    paramUpdateInterval = 1 #
    updateParameters = UpdateParameters(paramUpdateInterval, tau)
    sampleBatchFromMemory = SampleFromMemory(minibatchSize)

    learnInterval = 100
    learningStartBufferSize = minibatchSize * maxTimeStep
    startLearn = StartLearn(learningStartBufferSize, learnInterval)

    trainMADDPGModelsIndivid = TrainMADDPGModelsWithBuffer(updateParameters, trainActor, trainCritic, sampleBatchFromMemory, startLearn, modelsListIndivid)
    trainMADDPGModelsShared = TrainMADDPGModelsWithBuffer(updateParameters, trainActor, trainCritic, sampleBatchFromMemory, startLearn, modelsListShared)

    actOneStepOneModel = ActOneStep(actByPolicyTrainNoisy)
    actOneStepIndivid = lambda allAgentsStates, runTime: [actOneStepOneModel(model, allAgentsStates) for model in modelsListIndivid]
    actOneStepShared = lambda allAgentsStates, runTime: [actOneStepOneModel(model, allAgentsStates) for model in modelsListShared]

    sampleOneStepIndivid = SampleOneStep(transit, rewardFuncIndividWolf)
    sampleOneStepShared = SampleOneStep(transit, rewardFuncSharedWolf)

    runDDPGTimeStepIndivid = RunTimeStep(actOneStepIndivid, sampleOneStepIndivid, trainMADDPGModelsIndivid, observe = observe)
    runDDPGTimeStepShared = RunTimeStep(actOneStepShared, sampleOneStepShared, trainMADDPGModelsShared, observe = observe)

    runEpisodeIndivid = RunEpisode(reset, runDDPGTimeStepIndivid, maxTimeStep, isTerminal)
    runEpisodeShared = RunEpisode(reset, runDDPGTimeStepShared, maxTimeStep, isTerminal)

    getAgentModelIndivid = lambda agentId: lambda: trainMADDPGModelsIndivid.getTrainedModels()[agentId]
    getModelListIndivid = [getAgentModelIndivid(i) for i in range(numAgents)]
    modelSaveRate = 10000
    individStr = 'individ'
    fileName = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}{}_agent".format(
        numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, individStr)
    modelPath = os.path.join(dirName, '..', 'trainedModels', '3wolvesMaddpg_iterTrainSheep', fileName)
    saveModelsIndivid = [SaveModel(modelSaveRate, saveVariables, getTrainedModel, modelPath+ str(i), saveAllmodels) for i, getTrainedModel in enumerate(getModelListIndivid)]

    getAgentModelShared = lambda agentId: lambda: trainMADDPGModelsShared.getTrainedModels()[agentId]
    getModelListShared = [getAgentModelShared(i) for i in range(numAgents)]
    individStr = 'shared'
    fileName = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}{}_agent".format(
        numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, individStr)
    modelPath = os.path.join(dirName, '..', 'trainedModels', '3wolvesMaddpg_iterTrainSheep', fileName)
    saveModelsShared = [SaveModel(modelSaveRate, saveVariables, getTrainedModel, modelPath+ str(i), saveAllmodels) for i, getTrainedModel in enumerate(getModelListShared)]

    maddpgIterSheep = RunAlgorithmWithIterSheep(runEpisodeIndivid, runEpisodeShared, maxEpisode, saveModelsIndivid, saveModelsShared, numAgents)

    replayBufferIndivid = getBuffer(bufferSize)
    replayBufferShared = getBuffer(bufferSize)

    meanRewardList, trajectory = maddpgIterSheep(replayBufferShared, replayBufferIndivid)





if __name__ == '__main__':
    main()


