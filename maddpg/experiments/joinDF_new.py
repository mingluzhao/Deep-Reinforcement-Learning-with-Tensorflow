import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
from functionTools.loadSaveModel import saveToPickle, restoreVariables, loadFromPickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import sys
sys.setrecursionlimit(1000000)
import seaborn as sns

# def main():
#     # independentVariables = dict()
#     # independentVariables['numWolves'] = [2, 3, 4]
#     # independentVariables['trainingSequence'] = [('individ', 'individ'), ('individ', 'shared'), ('shared', 'shared'), ('shared', 'individ')]
#     # independentVariables['costActionRatio'] = [0.0, 0.02]
#     # independentVariables['epsID'] = np.linspace(5000, 120000, 24)
#
#     resultPath = os.path.join(dirName, '..', 'evalResults','evalTestEqui_resetAtKill')
#     #
#     # resultDF0 = loadFromPickle(os.path.join(resultPath, 'evalEquiWithKill' + '.pkl'))
#     # resultDF1 = loadFromPickle(os.path.join(resultPath, 'evalEquiWithKill1' + '.pkl'))
#     # resultDF2 = loadFromPickle(os.path.join(resultPath, 'evalEquiWithKill2' + '.pkl'))
#     # resultDF3 = loadFromPickle(os.path.join(resultPath, 'evalEquiWithKill3' + '.pkl'))
#     #
#     # resultDF2wolves = resultDF3.combine_first(resultDF2.combine_first(resultDF0.combine_first(resultDF1)))
#     #
#     # resultDF4 = loadFromPickle(os.path.join(resultPath, 'evalEquiWithKill4' + '.pkl'))
#     # resultDF5 = loadFromPickle(os.path.join(resultPath, 'evalEquiWithKill5' + '.pkl'))
#     # resultDF6 = loadFromPickle(os.path.join(resultPath, 'evalEquiWithKill6' + '.pkl'))
#     # resultDF7 = loadFromPickle(os.path.join(resultPath, 'evalEquiWithKill7' + '.pkl'))
#     #
#     # resultDF4wolves0 = resultDF7.combine_first(resultDF6.combine_first(resultDF5.combine_first(resultDF4)))
#     #
#     #
#     # resultDF8 = loadFromPickle(os.path.join(resultPath, 'evalEquiWithKill8' + '.pkl'))
#     # resultDF9 = loadFromPickle(os.path.join(resultPath, 'evalEquiWithKill9' + '.pkl'))
#     # resultDF10 = loadFromPickle(os.path.join(resultPath, 'evalEquiWithKill10' + '.pkl'))
#     # resultDF11 = loadFromPickle(os.path.join(resultPath, 'evalEquiWithKill11' + '.pkl'))
#     #
#     # resultDF4wolves1 = resultDF11.combine_first(resultDF10.combine_first(resultDF9.combine_first(resultDF8)))
#     #
#     # resultDF4wolves = resultDF4wolves0.combine_first(resultDF4wolves1)
#     #
#     # resultDF = resultDF2wolves.combine_first(resultDF4wolves)
#     #
#     # saveToPickle(resultDF, os.path.join(resultPath, 'evalEquiWithKill_10forEachCondition.pkl'))
#     #
#     evalResult = loadFromPickle(os.path.join(resultPath, 'evalEquiWithKill_10forEachCondition.pkl'))
#     #
#     result = evalResult.reset_index()
#     result['laterSelfishIndex'] = np.where(result['fileID'] < 10, 10000.0, 0.0)
#     result['recoveredSelfishIndex'] = result['recoveredSelfishIndex'].astype(str)
#     result['laterSelfishIndex'] = result['laterSelfishIndex'].astype(str)
#     result['rewardMechanism'] = result['recoveredSelfishIndex'] + ',' + result['laterSelfishIndex']
#
#     independentVariables = dict()
#     independentVariables['numWolves'] = [2, 4]
#     independentVariables['recoveredSelfishIndex'] = [0.0, 10000.0]
#     independentVariables['costActionRatio'] = [0.0, 0.02]
#     independentVariables['fileID'] = list(range(20))
#     independentVariables['epsID'] = np.linspace(10000, 120000, 12)
#
#     def stdErr(x):
#         return np.std(x) / np.sqrt(len(x) - 1)
#
#     resultDF = result.groupby(list(['numWolves', 'rewardMechanism', 'costActionRatio', 'epsID'])).agg(['mean', stdErr])['meanKill']
#
#     epsIDList = np.linspace(10000, 120000, 12)
#     figure = plt.figure(figsize=(11, 15))
#     plotCounter = 1
#
#     numRows = len(independentVariables['numWolves'])
#     numColumns = len(independentVariables['costActionRatio'])
#
#     for key, outmostSubDf in resultDF.groupby('numWolves'):
#         outmostSubDf.index = outmostSubDf.index.droplevel('numWolves')
#         for keyCol, outterSubDf in outmostSubDf.groupby('costActionRatio'):
#             outterSubDf.index = outterSubDf.index.droplevel('costActionRatio')
#             axForDraw = figure.add_subplot(numRows, numColumns, plotCounter)
#             for keyRow, innerSubDf in outterSubDf.groupby('rewardMechanism'):
#                 innerSubDf.index = innerSubDf.index.droplevel('rewardMechanism')
#                 plt.ylim([0, 5])
#
#                 innerSubDf.plot.line(ax = axForDraw, y='mean', yerr='stdErr', label = keyRow, uplims=True, lolims=True, capsize=3)
#                 if plotCounter <= numColumns:
#                     axForDraw.title.set_text('Action cost = ' + str(keyCol))
#                 if plotCounter% numColumns == 1:
#                     axForDraw.set_ylabel('Number of Wolves = ' + str(key))
#                 axForDraw.set_xlabel('epsID')
#
#             plotCounter += 1
#             plt.xticks(epsIDList, rotation='vertical')
#             plt.legend(title='Training Sequence', title_fontsize = 8, prop={'size': 8})
#
#     figure.text(x=0.03, y=0.5, s='Mean Episode Bite', ha='center', va='center', rotation=90)
#     plt.suptitle('MADDPG Evaluate Equilibrium')
#     plt.savefig(os.path.join(resultPath, 'evalEquilibriumFigureBiteAllWolves_resetAtKill'))
#     plt.show()




def main():
    resultPath = os.path.join(dirName, '..', 'evalResults','evalTestEqui_NoReset')
    evalResult = loadFromPickle(os.path.join(resultPath, 'evalEquiWithKill_10forEachCondition.pkl'))
    #
    result = evalResult.reset_index()
    result['laterSelfishIndex'] = np.where(result['fileID'] < 10, 10000.0, 0.0)
    result['recoveredSelfishIndex'] = result['recoveredSelfishIndex'].astype(str)
    result['laterSelfishIndex'] = result['laterSelfishIndex'].astype(str)
    result['rewardMechanism'] = result['recoveredSelfishIndex'] + ',' + result['laterSelfishIndex']

    independentVariables = dict()
    independentVariables['numWolves'] = [2, 4]
    independentVariables['recoveredSelfishIndex'] = [0.0, 10000.0]
    independentVariables['costActionRatio'] = [0.0, 0.02]
    independentVariables['fileID'] = list(range(20))
    independentVariables['epsID'] = np.linspace(10000, 120000, 12)

    def stdErr(x):
        return np.std(x) / np.sqrt(len(x) - 1)

    resultDF = result.groupby(list(['numWolves', 'rewardMechanism', 'costActionRatio', 'epsID'])).agg(['mean', stdErr])['meanKill']

    epsIDList = np.linspace(10000, 120000, 12)
    figure = plt.figure(figsize=(11, 15))
    plotCounter = 1

    numRows = len(independentVariables['numWolves'])
    numColumns = len(independentVariables['costActionRatio'])

    for key, outmostSubDf in resultDF.groupby('numWolves'):
        outmostSubDf.index = outmostSubDf.index.droplevel('numWolves')
        for keyCol, outterSubDf in outmostSubDf.groupby('costActionRatio'):
            outterSubDf.index = outterSubDf.index.droplevel('costActionRatio')
            axForDraw = figure.add_subplot(numRows, numColumns, plotCounter)
            for keyRow, innerSubDf in outterSubDf.groupby('rewardMechanism'):
                innerSubDf.index = innerSubDf.index.droplevel('rewardMechanism')
                plt.ylim([0, 15])

                innerSubDf.plot.line(ax = axForDraw, y='mean', yerr='stdErr', label = keyRow, uplims=True, lolims=True, capsize=3)
                if plotCounter <= numColumns:
                    axForDraw.title.set_text('Action cost = ' + str(keyCol))
                if plotCounter% numColumns == 1:
                    axForDraw.set_ylabel('Number of Wolves = ' + str(key))
                axForDraw.set_xlabel('epsID')

            plotCounter += 1
            plt.xticks(epsIDList, rotation='vertical')
            plt.legend(title='Reward Sequence', title_fontsize = 8, prop={'size': 8})

    figure.text(x=0.03, y=0.5, s='Mean Episode Bite', ha='center', va='center', rotation=90)
    plt.suptitle('MADDPG Evaluate Equilibrium')
    plt.savefig(os.path.join(resultPath, 'evalEquilibriumFigureBiteAllWolves_resetAtKill'))
    plt.show()

    # ax = sns.lineplot(x="timeStep", y="meanReward", hue="person", style="person", ci='sd', data=resultDF.reset_index())
    # plt.suptitle('Gym-Mujoco: Ant-v2 with DDPG')
    # plt.xlabel('Episode ID')
    # plt.savefig(os.path.join(evalResultDir, fileName))
    # plt.show()



# resultDF0 = loadFromPickle(os.path.join(resultPath, 'evalEquiWithKill' + '.pkl'))
# resultDF1 = loadFromPickle(os.path.join(resultPath, 'evalEquiWithKill1' + '.pkl'))
# resultDF2 = loadFromPickle(os.path.join(resultPath, 'evalEquiWithKill2' + '.pkl'))
# resultDF3 = loadFromPickle(os.path.join(resultPath, 'evalEquiWithKill3' + '.pkl'))
#
# resultDF2wolves = resultDF3.combine_first(resultDF2.combine_first(resultDF0.combine_first(resultDF1)))
#
# resultDF4 = loadFromPickle(os.path.join(resultPath, 'evalEquiWithKill4' + '.pkl'))
# resultDF5 = loadFromPickle(os.path.join(resultPath, 'evalEquiWithKill5' + '.pkl'))
# resultDF6 = loadFromPickle(os.path.join(resultPath, 'evalEquiWithKill6' + '.pkl'))
# resultDF7 = loadFromPickle(os.path.join(resultPath, 'evalEquiWithKill7' + '.pkl'))
#
# resultDF4wolves0 = resultDF7.combine_first(resultDF6.combine_first(resultDF5.combine_first(resultDF4)))
#
#
# resultDF8 = loadFromPickle(os.path.join(resultPath, 'evalEquiWithKill8' + '.pkl'))
# resultDF9 = loadFromPickle(os.path.join(resultPath, 'evalEquiWithKill9' + '.pkl'))
# resultDF10 = loadFromPickle(os.path.join(resultPath, 'evalEquiWithKill10' + '.pkl'))
# resultDF11 = loadFromPickle(os.path.join(resultPath, 'evalEquiWithKill11' + '.pkl'))
#
# resultDF4wolves1 = resultDF11.combine_first(resultDF10.combine_first(resultDF9.combine_first(resultDF8)))
#
# resultDF4wolves = resultDF4wolves0.combine_first(resultDF4wolves1)
#
# resultDF = resultDF2wolves.combine_first(resultDF4wolves)
#
# saveToPickle(resultDF, os.path.join(resultPath, 'evalEquiWithKill_10forEachCondition.pkl'))


if __name__ == '__main__':
    main()
