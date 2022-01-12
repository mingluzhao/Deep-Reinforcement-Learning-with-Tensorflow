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

#
# def main():
#     resultPath = os.path.join(dirName, '..', 'evalResults')
#     resultLoc = os.path.join(resultPath, 'evalRewardWithKillProbAndDistSensitiveNoBiteRewKillInfo_allConditions.pkl')
#     resultDF = loadFromPickle(resultLoc)
#
#     independentVariables = dict()
#     independentVariables['numWolves'] = [2, 3, 4, 5, 6]
#     independentVariables['sheepSpeedMultiplier'] = [0.5, 0.625, 0.75, 0.875, 1.0]
#     independentVariables['costActionRatio'] = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
#     independentVariables['rewardSensitivityToDistance'] = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
#
#     modify = Modify(resultDF)
#     levelNames = list(independentVariables.keys())
#     levelValues = list(independentVariables.values())
#     levelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
#     toSplitFrame = pd.DataFrame(index=levelIndex)
#     resultDFEasy = toSplitFrame.groupby(levelNames).apply(modify)
#
#     resultLoc = os.path.join(resultPath, 'evalRewardWithKillProbAndDistSensitiveNoBiteRewKillInfo_allConditions_EasytoPlot.pkl')
#     saveToPickle(resultDFEasy, resultLoc)


def main():
    resultPath = os.path.join(dirName, '..', 'evalResults')
    resultLoc = os.path.join(resultPath, 'evalRewardWithKillProbAndDistSensitiveNoBiteRewKillInfo_allConditions_EasytoPlot.pkl')
    resultDF = loadFromPickle(resultLoc)

    independentVariables = dict()
    independentVariables['numWolves'] = [2, 3, 4, 5, 6]
    independentVariables['sheepSpeedMultiplier'] = [0.5, 0.625, 0.75, 0.875, 1.0]
    independentVariables['costActionRatio'] = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
    independentVariables['rewardSensitivityToDistance'] = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    sensitivityTickLabels = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, '10k']
    speedTickLabels = ['0.65x', '0.82x', '1x', '1.15x', '1.3x']
    levelNames = resultDF.index.names

### heatmap-------------------------------------------------------------------
    # df = resultDF.reset_index().pivot_table(index='costActionRatio', columns='rewardSensitivityToDistance',
    #                                         values='meanKill', aggfunc=np.sum)
    # sns.heatmap(df, annot=True, fmt=".1f")
    # plt.show()

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np


### 3d bar plot-------------------------------------------------------------------

    for levelCombo in it.combinations(levelNames, 2):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # levelCombo = list( it.combinations(levelNames, 2))[0]
        summaryDF = resultDF.groupby(list(levelCombo)).mean().reset_index()
        yValues = independentVariables[levelCombo[1]]
        colors = ['y', 'g', 'b', 'r', 'o']
        for col, yValue in zip(colors[:len(yValues)], yValues):
            zValues = summaryDF[summaryDF[levelCombo[1]] == yValue]['meanKill']
            xValue = independentVariables[levelCombo[0]]
            cs = [col] * len(yValues)
            if levelCombo[0] == 'numWolves':
                binwidth = 0.8
            elif levelCombo[0] == 'sheepSpeedMultiplier':
                binwidth = 0.2
            else:
                binwidth = 0.008
            ax.bar(xValue, zValues, zs= yValue, zdir='y', color=cs, alpha=0.8, width = binwidth)

        ax.set_xlabel(levelCombo[0])
        ax.set_ylabel(levelCombo[1])
        ax.set_zlabel('meanKill')

        if levelCombo[0] == 'rewardSensitivityToDistance':
            plt.xticks(independentVariables['rewardSensitivityToDistance'], sensitivityTickLabels)

        elif levelCombo[0] == 'sheepSpeedMultiplier':
            plt.xticks(independentVariables['sheepSpeedMultiplier'], speedTickLabels)

        else:
            plt.xticks(independentVariables[levelCombo[0]])

        if levelCombo[1] == 'rewardSensitivityToDistance':
            plt.yticks(independentVariables['rewardSensitivityToDistance'], sensitivityTickLabels)

        elif levelCombo[1] == 'sheepSpeedMultiplier':
            plt.yticks(independentVariables['sheepSpeedMultiplier'], speedTickLabels)

        else:
            plt.yticks(independentVariables[levelCombo[1]])

        plt.yticks(independentVariables[levelCombo[1]])
        plt.show()



########bar plot
    # for levelCombo in it.combinations(levelNames, 2):
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     # levelCombo = list( it.combinations(levelNames, 2))[0]
    #     summaryDF = resultDF.groupby(list(levelCombo)).mean().reset_index()
    #     yValues = independentVariables[levelCombo[1]]
    #     colors = ['y', 'g', 'b', 'r', 'o']
    #     for col, yValue in zip(colors[:len(yValues)], yValues):
    #         zValues = summaryDF[summaryDF[levelCombo[1]] == yValue]['meanAgentAction']
    #         xValue = independentVariables[levelCombo[0]]
    #         cs = [col] * len(yValues)
    #         if levelCombo[0] == 'numWolves':
    #             binwidth = 0.8
    #         elif levelCombo[0] == 'sheepSpeedMultiplier':
    #             binwidth = 0.2
    #         else:
    #             binwidth = 0.008
    #         ax.bar(xValue, zValues, zs= yValue, zdir='y', color=cs, alpha=0.8, width = binwidth)
    #
    #     ax.set_xlabel(levelCombo[0])
    #     ax.set_ylabel(levelCombo[1])
    #     ax.set_zlabel('meanAgentAction')
    #
    #     if levelCombo[0] == 'rewardSensitivityToDistance':
    #         plt.xticks(independentVariables['rewardSensitivityToDistance'], sensitivityTickLabels)
    #
    #     elif levelCombo[0] == 'sheepSpeedMultiplier':
    #         plt.xticks(independentVariables['sheepSpeedMultiplier'], speedTickLabels)
    #
    #     else:
    #         plt.xticks(independentVariables[levelCombo[0]])
    #
    #     if levelCombo[1] == 'rewardSensitivityToDistance':
    #         plt.yticks(independentVariables['rewardSensitivityToDistance'], sensitivityTickLabels)
    #
    #     elif levelCombo[1] == 'sheepSpeedMultiplier':
    #         plt.yticks(independentVariables['sheepSpeedMultiplier'], speedTickLabels)
    #
    #     else:
    #         plt.yticks(independentVariables[levelCombo[1]])
    #
    #     plt.yticks(independentVariables[levelCombo[1]])
    #     plt.show()




### 3d surface plot-------------------------------------------------------------------

    for levelCombo in it.combinations(levelNames, 2):
        summaryDF = resultDF.groupby(list(levelCombo)).mean().reset_index()

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_trisurf(summaryDF[levelCombo[0]], summaryDF[levelCombo[1]], summaryDF['meanKill'],
                        cmap=plt.cm.viridis, linewidth=0.2)

        ax.set_xlabel(levelCombo[0])
        ax.set_ylabel(levelCombo[1])
        ax.set_title("Mean Episode Kill with 75 steps/eps")

        if levelCombo[0] == 'rewardSensitivityToDistance':
            numXRepeat = int(len(summaryDF[levelCombo[0]]) / len(independentVariables[levelCombo[0]]))
            xtickValue = sensitivityTickLabels * numXRepeat
            plt.xticks(summaryDF[levelCombo[0]], xtickValue)
        elif levelCombo[0] == 'sheepSpeedMultiplier':
            numXRepeat = int(len(summaryDF[levelCombo[0]]) / len(independentVariables[levelCombo[0]]))
            xtickValue = speedTickLabels * numXRepeat
            plt.xticks(summaryDF[levelCombo[0]], xtickValue)
        else:
            plt.xticks(independentVariables[levelCombo[0]])

        if levelCombo[1] == 'rewardSensitivityToDistance':
            numYRepeat = int(len(summaryDF[levelCombo[1]]) / len(independentVariables[levelCombo[1]]))
            ytickValue = sensitivityTickLabels * numYRepeat
            plt.yticks(summaryDF[levelCombo[1]], ytickValue)

        elif levelCombo[1] == 'sheepSpeedMultiplier':
            numYRepeat = int(len(summaryDF[levelCombo[1]]) / len(independentVariables[levelCombo[1]]))
            ytickValue = speedTickLabels * numYRepeat
            plt.yticks(summaryDF[levelCombo[1]], ytickValue)
        else:
            plt.yticks(independentVariables[levelCombo[1]])

        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
        # plt.savefig(os.path.join(resultPath, 'eval{}With{}_meanKill_surface.pdf'.format(levelCombo[0], levelCombo[1])), dpi = 600)
        plt.close()



# ####---------------------------------------------------
#         fig = plt.figure()
#         ax = fig.gca(projection='3d')
#         surf = ax.plot_trisurf(summaryDF[levelCombo[0]], summaryDF[levelCombo[1]], summaryDF['meanAgentAction'],
#                                cmap=plt.cm.viridis, linewidth=0.2)
#
#         ax.set_xlabel(levelCombo[0])
#         ax.set_ylabel(levelCombo[1])
#         ax.set_title("Mean Episode Single Agent Moving Distance with 75 steps/eps")
#
#         if levelCombo[0] == 'rewardSensitivityToDistance':
#             plt.xticks(independentVariables['rewardSensitivityToDistance'], sensitivityTickLabels)
#
#         elif levelCombo[0] == 'sheepSpeedMultiplier':
#             plt.xticks(independentVariables['sheepSpeedMultiplier'], speedTickLabels)
#
#         else:
#             plt.xticks(independentVariables[levelCombo[0]])
#
#         if levelCombo[1] == 'rewardSensitivityToDistance':
#             plt.yticks(independentVariables['rewardSensitivityToDistance'], sensitivityTickLabels)
#         elif levelCombo[1] == 'sheepSpeedMultiplier':
#             plt.yticks(independentVariables['sheepSpeedMultiplier'], speedTickLabels)
#         else:
#             plt.yticks(independentVariables[levelCombo[1]])
#
#         fig.colorbar(surf, shrink=0.5, aspect=5)
#         plt.savefig(os.path.join(resultPath, 'eval{}With{}_meanAgentDist_surface.pdf'.format(levelCombo[0], levelCombo[1])),
#                     dpi=600)
#         plt.close()
#
# ####---------------------------------------------------
#         fig = plt.figure()
#         line = sns.lineplot(x=levelCombo[0], y="meanKill", hue=levelCombo[1], style=levelCombo[1],
#                             ci='sd', data=resultDF.reset_index())
#
#         if levelCombo[1] == 'rewardSensitivityToDistance':
#             plt.legend(title='rewardSensitivityToDistance', labels=['0', '1', '2', '10k'])
#
#         if levelCombo[1] == 'sheepSpeedMultiplier':
#             plt.legend(title='sheepSpeedMultiplier', labels=['0.65x', '1x', '1.3x'])
#
#         if levelCombo[0] == 'sheepSpeedMultiplier':
#             plt.xticks([0.5, 0.75, 1.0], ['0.65x', '1x', '1.3x'])
#         else:
#             plt.xticks(independentVariables[levelCombo[0]])
#         plt.savefig(os.path.join(resultPath, 'eval{}With{}_meanKill_line.pdf'.format(levelCombo[0], levelCombo[1])), dpi = 600)
#         plt.close()
#
# ####---------------------------------------------------
#         fig = plt.figure()
#         line = sns.lineplot(x=levelCombo[0], y="meanAgentAction", hue=levelCombo[1], style=levelCombo[1],
#                             ci='sd', data=resultDF.reset_index())
#
#         if levelCombo[1] == 'rewardSensitivityToDistance':
#             plt.legend(title='rewardSensitivityToDistance', labels=['0', '1', '2', '10k'])
#
#         if levelCombo[1] == 'sheepSpeedMultiplier':
#             plt.legend(title='sheepSpeedMultiplier', labels=['0.65x', '1x', '1.3x'])
#
#         if levelCombo[0] == 'sheepSpeedMultiplier':
#             plt.xticks([0.5, 0.75, 1.0], ['0.65x', '1x', '1.3x'])
#         else:
#             plt.xticks(independentVariables[levelCombo[0]])
#
#         plt.savefig(os.path.join(resultPath, 'eval{}With{}_meanAgentDist_line.pdf'.format(levelCombo[0], levelCombo[1])),
#                     dpi=600)
#         plt.close()
#
# ####---------------------------------------------------
#         fig = plt.figure()
#         line = sns.lmplot(x=levelCombo[0], y="meanKill", hue=levelCombo[1],
#                             data=resultDF.reset_index())
#         if levelCombo[1] == 'rewardSensitivityToDistance':
#             new_labels = ['0', '1', '2', '10k']
#             for t, l in zip(line._legend.texts, new_labels): t.set_text(l)
#
#         if levelCombo[1] == 'sheepSpeedMultiplier':
#             new_labels = ['0.65x', '1x', '1.3x']
#             for t, l in zip(line._legend.texts, new_labels): t.set_text(l)
#
#         if levelCombo[0] == 'sheepSpeedMultiplier':
#             plt.xticks([0.5, 0.75, 1.0], ['0.65x', '1x', '1.3x'])
#         else:
#             plt.xticks(independentVariables[levelCombo[0]])
#         plt.savefig(os.path.join(resultPath, 'eval{}With{}_meanKill_lm.pdf'.format(levelCombo[0], levelCombo[1])), dpi = 600)
#         plt.close()
#
# ####---------------------------------------------------
#         fig = plt.figure()
#         line = sns.lmplot(x=levelCombo[0], y="meanAgentAction", hue=levelCombo[1],
#                           data=resultDF.reset_index())
#         if levelCombo[1] == 'rewardSensitivityToDistance':
#             new_labels = ['0', '1', '2', '10k']
#             for t, l in zip(line._legend.texts, new_labels): t.set_text(l)
#
#         if levelCombo[1] == 'sheepSpeedMultiplier':
#             new_labels = ['0.65x', '1x', '1.3x']
#             for t, l in zip(line._legend.texts, new_labels): t.set_text(l)
#
#         if levelCombo[0] == 'sheepSpeedMultiplier':
#             plt.xticks([0.5, 0.75, 1.0], ['0.65x', '1x', '1.3x'])
#         else:
#             plt.xticks(independentVariables[levelCombo[0]])
#         plt.savefig(os.path.join(resultPath, 'eval{}With{}_meanAgentDist_lm.pdf'.format(levelCombo[0], levelCombo[1])),
#                     dpi=600)
#         plt.close()


if __name__ == '__main__':
    main()