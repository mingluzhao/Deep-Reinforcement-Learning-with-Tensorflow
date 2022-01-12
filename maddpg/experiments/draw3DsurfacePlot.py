import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))

from functionTools.loadSaveModel import saveToPickle, restoreVariables, loadFromPickle
import itertools as it
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Modify:
    def __init__(self, resultDF):
        self.resultDF = resultDF

    def __call__(self, df):
        sheepSpeedMultiplier = df.index.get_level_values('sheepSpeedMultiplier')[0]
        costActionRatio = df.index.get_level_values('costActionRatio')[0]
        rewardSensitivityToDistance = df.index.get_level_values('rewardSensitivityToDistance')[0]
        if rewardSensitivityToDistance == 3.0:
            rewardSensitivityToDistance = 10000.0
        numWolves = df.index.get_level_values('numWolves')[0]

        meanTrajKill = self.resultDF.loc[numWolves, sheepSpeedMultiplier, costActionRatio, rewardSensitivityToDistance]['meanKill']
        seKill = self.resultDF.loc[numWolves, sheepSpeedMultiplier, costActionRatio, rewardSensitivityToDistance]['seKill']
        meanTrajAction = self.resultDF.loc[numWolves, sheepSpeedMultiplier, costActionRatio, rewardSensitivityToDistance]['meanTrajAction']
        seTrajAction = self.resultDF.loc[numWolves, sheepSpeedMultiplier, costActionRatio, rewardSensitivityToDistance]['seTrajAction']
        meanAgentAction = self.resultDF.loc[numWolves, sheepSpeedMultiplier, costActionRatio, rewardSensitivityToDistance]['meanAgentAction']
        seAgentAction = self.resultDF.loc[numWolves, sheepSpeedMultiplier, costActionRatio, rewardSensitivityToDistance]['seAgentAction']

        return pd.Series({'meanKill': meanTrajKill, 'seKill': seKill,
                          'meanTrajAction': meanTrajAction, 'seTrajAction': seTrajAction,
                          'meanAgentAction': meanAgentAction, 'seAgentAction': seAgentAction})


def main():
    resultPath = os.path.join(dirName, '..', 'evalResults')
    resultLoc = os.path.join(resultPath, 'evalRewardWithKillProbAndDistSensitiveNoBiteRewKillInfo_allConditions_EasytoPlot.pkl')
    resultDF = loadFromPickle(resultLoc)

    independentVariables = dict()
    independentVariables['numWolves'] = [2, 3, 4, 5, 6]
    independentVariables['sheepSpeedMultiplier'] = [0.5, 0.625, 0.75, 0.875, 1.0]
    independentVariables['costActionRatio'] = [.0, .005, .01, .015, .02, .025, .03]
    independentVariables['rewardSensitivityToDistance'] = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    # sensitivityTickLabels = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, '10k']
    sensitivityTickLabels = ['unselfish', '', '', '', '', '', 'selfish']
    costTickLabels = ['.0', '.005', '.01', '.015', '.02', '.025', '.03']
    speedTickLabels = ['0.65x', '0.82x', '1x', '1.15x', '1.3x']
    levelNames = resultDF.index.names


    for levelCombo in it.combinations(levelNames, 2):
        summaryDF = resultDF.groupby(list(levelCombo)).mean().reset_index()

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_trisurf(summaryDF[levelCombo[0]], summaryDF[levelCombo[1]], summaryDF['meanKill'],
                        cmap=plt.cm.viridis, linewidth=0.2)

        if levelCombo[0] == 'rewardSensitivityToDistance':
            plt.xticks(independentVariables['rewardSensitivityToDistance'], sensitivityTickLabels)
            ax.set_xlabel('Selfish Index')

        elif levelCombo[0] == 'sheepSpeedMultiplier':
            plt.xticks(independentVariables['sheepSpeedMultiplier'], speedTickLabels)
            ax.set_xlabel('Prey Speed Multiplier')

        elif levelCombo[0] == 'costActionRatio':
            plt.xticks(independentVariables['costActionRatio'], costTickLabels)
            ax.set_xlabel('Cost-Action Ratio')

        else:
            plt.xticks(independentVariables[levelCombo[0]])
            ax.set_xlabel('Number of Predators')

        if levelCombo[1] == 'rewardSensitivityToDistance':
            plt.yticks(independentVariables['rewardSensitivityToDistance'], sensitivityTickLabels)
            ax.set_ylabel('Selfish Index')

        elif levelCombo[1] == 'sheepSpeedMultiplier':
            plt.yticks(independentVariables['sheepSpeedMultiplier'], speedTickLabels)
            ax.set_ylabel('Prey Speed Multiplier')

        elif levelCombo[1] == 'costActionRatio':
            plt.yticks(independentVariables['costActionRatio'], costTickLabels)
            ax.set_ylabel('Cost-Action Ratio')

        else:
            plt.yticks(independentVariables[levelCombo[1]])
            ax.set_ylabel('Number of Predators')


        ax.set_zlabel('Mean Episode Kill')
        ax.set_title("Mean Episode Kill with 75 steps/eps")
        n_x = len(independentVariables[levelCombo[0]])
        n_y = len(independentVariables[levelCombo[1]])
        ax.set_box_aspect([1, (n_y-1)/(n_x-1), 1])

        # fig.colorbar(surf, shrink=0.5, aspect=5)
        # plt.savefig(os.path.join(resultPath, 'eval{}With{}_meanKill_surface.pdf'.format(levelCombo[0], levelCombo[1])), dpi = 600)
        fig.savefig(os.path.join(resultPath, 'eval{}With{}_meanKill_surface.svg'.format(levelCombo[0], levelCombo[1])), format='svg')
        # plt.show()
        plt.close()

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_trisurf(summaryDF[levelCombo[0]], summaryDF[levelCombo[1]], summaryDF['meanTrajAction'],
                               cmap=plt.cm.viridis, linewidth=0.2)

        ax.set_xlabel(levelCombo[0])
        ax.set_ylabel(levelCombo[1])
        ax.set_zlabel('Mean Moving Distance')

        ax.set_title("Mean Episode Agents Total Moving Distance with 75 steps/eps")

        if levelCombo[0] == 'rewardSensitivityToDistance':
            plt.xticks(independentVariables['rewardSensitivityToDistance'], sensitivityTickLabels)
            ax.set_xlabel('Selfish Index')

        elif levelCombo[0] == 'sheepSpeedMultiplier':
            plt.xticks(independentVariables['sheepSpeedMultiplier'], speedTickLabels)
            ax.set_xlabel('Prey Speed Multiplier')

        elif levelCombo[0] == 'costActionRatio':
            plt.xticks(independentVariables['costActionRatio'], costTickLabels)
            ax.set_xlabel('Cost-Action Ratio')

        else:
            plt.xticks(independentVariables[levelCombo[0]])
            ax.set_xlabel('Number of Predators')

        if levelCombo[1] == 'rewardSensitivityToDistance':
            plt.yticks(independentVariables['rewardSensitivityToDistance'], sensitivityTickLabels)
            ax.set_ylabel('Selfish Index')

        elif levelCombo[1] == 'sheepSpeedMultiplier':
            plt.yticks(independentVariables['sheepSpeedMultiplier'], speedTickLabels)
            ax.set_ylabel('Prey Speed Multiplier')

        elif levelCombo[1] == 'costActionRatio':
            plt.yticks(independentVariables['costActionRatio'], costTickLabels)
            ax.set_ylabel('Cost-Action Ratio')

        else:
            plt.yticks(independentVariables[levelCombo[1]])
            ax.set_ylabel('Number of Predators')

        n_x = len(independentVariables[levelCombo[0]])
        n_y = len(independentVariables[levelCombo[1]])
        ax.set_box_aspect([1, (n_y-1)/(n_x-1), 1])
        # fig.colorbar(surf, shrink=0.5, aspect=5)
        # plt.savefig(os.path.join(resultPath, 'eval{}With{}_moveDist_surface.pdf'.format(levelCombo[0], levelCombo[1])), dpi=600)

        fig.savefig(os.path.join(resultPath, 'eval{}With{}_moveDist_surface.svg'.format(levelCombo[0], levelCombo[1])), format='svg')

        # plt.show()
        plt.close()

# def main():
#     resultPath = os.path.join(dirName, '..', 'evalResults')
#     resultLoc = os.path.join(resultPath,
#                              'evalRewardWithKillProbAndDistSensitiveNoBiteRewKillInfo_allConditions_EasytoPlot.pkl')
#     resultDF = loadFromPickle(resultLoc)
#
#     independentVariables = dict()
#     independentVariables['numWolves'] = [2, 3, 4, 5, 6]
#     independentVariables['sheepSpeedMultiplier'] = [0.5, 0.625, 0.75, 0.875, 1.0]
#     independentVariables['costActionRatio'] = [.0, .005, .01, .015, .02, .025, .03]
#     independentVariables['rewardSensitivityToDistance'] = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
#
#     # sensitivityTickLabels = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, '10k']
#     sensitivityTickLabels = ['unselfish', '', '', '', '', '', 'selfish']
#     costTickLabels = ['.0', '.005', '.01', '.015', '.02', '.025', '.03']
#     speedTickLabels = ['0.65x', '0.82x', '1x', '1.15x', '1.3x']
#     levelNames = resultDF.index.names
#
#     # fig.savefig('filename.eps', format='eps')
#
#     levelCombo = list(it.combinations(levelNames, 2))[1]
#     summaryDF = resultDF.groupby(list(levelCombo)).mean().reset_index()
#
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
#     surf = ax.plot_trisurf(summaryDF[levelCombo[0]], summaryDF[levelCombo[1]], summaryDF['meanKill'],
#                            cmap=plt.cm.viridis, linewidth=0.2)
#
#     if levelCombo[0] == 'rewardSensitivityToDistance':
#         plt.xticks(independentVariables['rewardSensitivityToDistance'], sensitivityTickLabels)
#         ax.set_xlabel('selfishIndex')
#
#     elif levelCombo[0] == 'sheepSpeedMultiplier':
#         plt.xticks(independentVariables['sheepSpeedMultiplier'], speedTickLabels)
#         ax.set_xlabel(levelCombo[0])
#
#     elif levelCombo[0] == 'costActionRatio':
#         plt.xticks(independentVariables['costActionRatio'], costTickLabels)
#         ax.set_xlabel(levelCombo[0])
#
#     else:
#         plt.xticks(independentVariables[levelCombo[0]])
#         ax.set_xlabel(levelCombo[0])
#
#     if levelCombo[1] == 'rewardSensitivityToDistance':
#         plt.yticks(independentVariables['rewardSensitivityToDistance'], sensitivityTickLabels)
#         ax.set_ylabel('selfishIndex')
#
#     elif levelCombo[1] == 'sheepSpeedMultiplier':
#         plt.yticks(independentVariables['sheepSpeedMultiplier'], speedTickLabels)
#         ax.set_ylabel(levelCombo[1])
#
#     elif levelCombo[1] == 'costActionRatio':
#         plt.yticks(independentVariables['costActionRatio'], costTickLabels)
#         ax.set_ylabel(levelCombo[1])
#
#     else:
#         plt.yticks(independentVariables[levelCombo[1]])
#         ax.set_ylabel(levelCombo[1])
#
#     ax.set_zlabel('Mean Episode Kill')
#     ax.set_title("Mean Episode Kill with 75 steps/eps")
#
#
#     n_y = len(independentVariables[levelCombo[0]])
#     n_x = len(independentVariables[levelCombo[1]])
#     ax.set_box_aspect([1, (n_x-1)/(n_y-1), 1])
#     plt.savefig(os.path.join(resultPath, 'eval{}With{}_moveDist_surface.svg'.format(levelCombo[0], levelCombo[1])), format='svg', dpi = 600)
#
#     plt.show()


if __name__ == '__main__':
    main()