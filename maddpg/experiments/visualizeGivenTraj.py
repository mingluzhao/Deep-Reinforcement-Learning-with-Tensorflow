import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
from gym import spaces
import matplotlib.pyplot as plt

from maddpg.maddpgAlgor.trainer.myMADDPG import *
from functionTools.loadSaveModel import loadFromPickle


def calcTrajRewardWithSharedWolfReward(traj):
    rewardIDinTraj = 2
    rewardList = [timeStepInfo[rewardIDinTraj][0] for timeStepInfo in traj]
    trajReward = np.sum(rewardList)
    return trajReward

def calcTrajRewardWithIndividualWolfReward(traj, wolvesID):
    rewardIDinTraj = 2
    getWolfReward = lambda allAgentsReward: np.sum([allAgentsReward[wolfID] for wolfID in wolvesID])
    rewardList = [getWolfReward(timeStepInfo[rewardIDinTraj]) for timeStepInfo in traj]
    trajReward = np.sum(rewardList)
    return trajReward


def main():
    numWolves = 2
    numSheeps = 1
    numBlocks = 3
    maxEpisode = 60000

    numAgents = numWolves + numSheeps
    numEntities = numAgents + numBlocks
    wolvesID = list(range(numWolves))
    sheepsID = list(range(numWolves, numAgents))
    blocksID = list(range(numAgents, numEntities))


    trajFileName = "maddpg{}wolves{}sheep{}blocks{}epsTrajectory.pickle".format(numWolves, numSheeps, numBlocks, maxEpisode)
    trajSavePath = os.path.join(dirName, '..', 'trajectory', trajFileName)
    trajectoryList = loadFromPickle(trajSavePath)

    meanTrajReward = np.mean([calcTrajRewardWithSharedWolfReward(traj) for traj in trajectoryList])

    numSheepsList = [1, 2, 4]
    plotResult = True
    if plotResult:
        plt.plot(numSheepsList, meanRewardList)
        plt.show()

if __name__ == '__main__':
    main()
