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


def main():
    resultPath = os.path.join(dirName, '..', 'evalResults','evalTestEqui_NoReset')
    evalResult = loadFromPickle(os.path.join(resultPath, 'evalEquiWithKill_10forEachCondition.pkl'))
    #
    result = evalResult.reset_index()
    result = result[result['recoveredSelfishIndex'] == 10000.0]
    result['laterSelfishIndex'] = np.where(result['fileID'] < 10, 10000.0, 0.0)
    result['recoveredSelfishIndex'] = result['recoveredSelfishIndex'].astype(str)
    result['laterSelfishIndex'] = result['laterSelfishIndex'].astype(str)
    result['rewardMechanism'] = result['recoveredSelfishIndex'] + ',' + result['laterSelfishIndex']

    resultToPlot = result[result['numWolves'] == 4]

    independentVariables = dict()
    independentVariables['numWolves'] = [4]
    independentVariables['costActionRatio'] = [0.0, 0.02]
    independentVariables['fileID'] = list(range(20))
    independentVariables['epsID'] = np.linspace(10000, 120000, 12)


    sns_plot = sns.relplot(data=resultToPlot, x="epsID", y="meanKill",col="costActionRatio", hue="rewardMechanism", style="rewardMechanism", kind="line")
    # ax = sns.lineplot(x="epsID", y="meanKill", hue="rewardMechanism", style="rewardMechanism", data=resultToPlot)
    plt.ylabel('Mean Episode Kill')
    plt.xlabel('Episode ID')
    # plt.savefig(os.path.join(resultPath, 'evalEquilibriumFigureBite4Wolves_noResetAtKill.pdf'))
    # plt.show()
    sns_plot.savefig(os.path.join(resultPath, 'evalEquilibriumFigureBite4WolvesAllCost_noResetAtKill.pdf'), dpi=300)

    # figure = ax.get_figure()
    # figure.savefig(os.path.join(resultPath, 'evalEquilibriumFigureBite4WolvesAllCost_noResetAtKill.pdf'), dpi=300)

    from scipy.stats import ttest_ind

    selfishUnselfish0cost60k = np.array(resultToPlot[(resultToPlot['epsID'] == 60000.0)& (resultToPlot['laterSelfishIndex'] == '0.0') & (resultToPlot['costActionRatio'] == 0.0)]['meanKill'])
    selfishUnselfish0cost120k = np.array(resultToPlot[(resultToPlot['epsID'] == 120000.0)& (resultToPlot['laterSelfishIndex'] == '0.0') & (resultToPlot['costActionRatio'] == 0.0)]['meanKill'])
    stat, p = ttest_ind(selfishUnselfish0cost60k, selfishUnselfish0cost120k)
    print('compare 0 cost, us, 60k vs 120k: t={}, p={}'.format(stat, p))

    selfishUnselfish2cost60k = np.array(resultToPlot[(resultToPlot['epsID'] == 60000.0)& (resultToPlot['laterSelfishIndex'] == '0.0') & (resultToPlot['costActionRatio'] == 0.02)]['meanKill'])
    selfishUnselfish2cost120k = np.array(resultToPlot[(resultToPlot['epsID'] == 120000.0)& (resultToPlot['laterSelfishIndex'] == '0.0') & (resultToPlot['costActionRatio'] == 0.02)]['meanKill'])
    stat, p = ttest_ind(selfishUnselfish2cost60k, selfishUnselfish2cost120k)
    print('compare 0.02 cost, us, 60k vs 120k: t={}, p={}'.format(stat, p))

    # (t = 4.83, p = 0.001)
    # (t = 81.28, p < 0.001)


    selfishselfish0cost60k = np.array(resultToPlot[(resultToPlot['epsID'] == 60000.0)& (resultToPlot['laterSelfishIndex'] == '10000.0') & (resultToPlot['costActionRatio'] == 0.0)]['meanKill'])
    selfishselfish0cost120k = np.array(resultToPlot[(resultToPlot['epsID'] == 120000.0)& (resultToPlot['laterSelfishIndex'] == '10000.0') & (resultToPlot['costActionRatio'] == 0.0)]['meanKill'])
    stat, p = ttest_ind(selfishselfish0cost60k, selfishselfish0cost120k)
    print('compare 0 cost, us, 60k vs 120k: t={}, p={}'.format(stat, p))

    selfishselfish2cost60k = np.array(resultToPlot[(resultToPlot['epsID'] == 60000.0)& (resultToPlot['laterSelfishIndex'] == '10000.0') & (resultToPlot['costActionRatio'] == 0.02)]['meanKill'])
    selfishselfish2cost120k = np.array(resultToPlot[(resultToPlot['epsID'] == 120000.0)& (resultToPlot['laterSelfishIndex'] == '10000.0') & (resultToPlot['costActionRatio'] == 0.02)]['meanKill'])
    stat, p = ttest_ind(selfishselfish2cost60k, selfishselfish2cost120k)
    print('compare 0.02 cost, us, 60k vs 120k: t={}, p={}'.format(stat, p))

    # (t=-0.7155791126163047, p=0.48343124414381033)
    # (t=-1.9724522117960896, p=0.06412070677158731)


if __name__ == '__main__':
    main()
