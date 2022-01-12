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

def main():
    independentVariables = dict()
    independentVariables['numWolves'] = [2, 3, 4]
    independentVariables['trainingSequence'] = [('individ', 'individ'), ('individ', 'shared'), ('shared', 'shared'), ('shared', 'individ')]
    independentVariables['costActionRatio'] = [0.0, 0.02]
    independentVariables['epsID'] = np.linspace(5000, 120000, 24)

    resultPath = os.path.join(dirName, '..', 'evalResults')

    resultLoc2 = os.path.join(resultPath, 'evalEquilibrium' + str(2) + 'wolves.pkl')
    resultDF2 = loadFromPickle(resultLoc2)
    resultDF2 = resultDF2.reorder_levels(['numWolves', 'costActionRatio', 'trainingSequence', 'epsID']).sort_index()

    resultLoc3 = os.path.join(resultPath, 'evalEquilibrium' + str(3) + 'wolves.pkl')
    resultDF3 = loadFromPickle(resultLoc3)
    resultDF3 = resultDF3.reorder_levels(['numWolves', 'costActionRatio', 'trainingSequence', 'epsID']).sort_index()

    resultLoc4 = os.path.join(resultPath, 'evalEquilibrium' + str(4) + 'wolves.pkl')
    resultDF4 = loadFromPickle(resultLoc4)
    resultDF4 = resultDF4.reorder_levels(['numWolves', 'costActionRatio', 'trainingSequence', 'epsID']).sort_index()

    resultDF23 = resultDF2.combine_first(resultDF3)
    resultDF = resultDF23.combine_first(resultDF4)


    print(resultDF)


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
            for keyRow, innerSubDf in outterSubDf.groupby('trainingSequence'):
                innerSubDf.index = innerSubDf.index.droplevel('trainingSequence')
                plt.ylim([0, 15])

                innerSubDf.plot.line(ax = axForDraw, y='meanBite', yerr='seBite', label = keyRow, uplims=True, lolims=True, capsize=3)
                if plotCounter <= numColumns:
                    axForDraw.title.set_text('Action cost = ' + str(keyCol))
                if plotCounter% numColumns == 1:
                    axForDraw.set_ylabel('Number of Wolves = ' + str(key))
                axForDraw.set_xlabel('epsID')

            plotCounter += 1
            plt.xticks(epsIDList, rotation='vertical')
            plt.legend(title='Training Sequence', title_fontsize = 8, prop={'size': 8})

    figure.text(x=0.03, y=0.5, s='Mean Episode Bite', ha='center', va='center', rotation=90)
    plt.suptitle('MADDPG Evaluate Equilibrium')
    plt.savefig(os.path.join(resultPath, 'evalEquilibriumFigureBiteAllWolves'))
    plt.show()






if __name__ == '__main__':
    main()
