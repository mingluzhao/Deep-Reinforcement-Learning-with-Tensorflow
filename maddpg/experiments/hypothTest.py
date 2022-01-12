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
        sheepSpeedMultiplier = float(df.index.get_level_values('sheepSpeedMultiplier')[0][:-1])
        costActionRatio = float(df.index.get_level_values('costActionRatio')[0][:-1])
        rewardSensitivityToDistance = float(df.index.get_level_values('rewardSensitivityToDistance')[0][:-1])
        if rewardSensitivityToDistance == 3.0:
            rewardSensitivityToDistance = 10000.0
        numWolves = float(df.index.get_level_values('numWolves')[0][:-1])

        meanKill = self.resultDF.loc[numWolves, sheepSpeedMultiplier, costActionRatio, rewardSensitivityToDistance]['meanKill']
        seKill = self.resultDF.loc[numWolves, sheepSpeedMultiplier, costActionRatio, rewardSensitivityToDistance]['seKill']
        meanTrajAction = self.resultDF.loc[numWolves, sheepSpeedMultiplier, costActionRatio, rewardSensitivityToDistance]['meanTrajAction']
        seTrajAction = self.resultDF.loc[numWolves, sheepSpeedMultiplier, costActionRatio, rewardSensitivityToDistance]['seTrajAction']

        return pd.Series({'meanKill': meanKill})


def main():
    resultPath = os.path.join(dirName, '..', 'evalResults')
    resultLoc = os.path.join(resultPath, 'evalRewardWithKillProbAndDistSensitiveNoBiteRewKillInfo_allConditions.pkl')
    resultDF = loadFromPickle(resultLoc)

    independentVariables = dict()
    independentVariables['numWolves'] = ['2x', '3x', '4x', '5x', '6x']
    independentVariables['sheepSpeedMultiplier'] = ['0.5s', '0.625s', '0.75s', '0.875s', '1.0s']
    independentVariables['costActionRatio'] = ['.0c', '.005c', '.01c', '.015c', '.02c', '.025c', '.03c']
    independentVariables['rewardSensitivityToDistance'] = ['0.0r', '0.5r', '1.0r', '1.5r', '2.0r', '2.5r', '3.0r']

    modify = Modify(resultDF)
    levelNames = list(independentVariables.keys())
    levelValues = list(independentVariables.values())
    levelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=levelIndex)
    resultDFEasy = toSplitFrame.groupby(levelNames).apply(modify)
    #
    # killDF = resultDFEasy.reset_index().explode('kill')

    # killDF = loadFromPickle(os.path.join(resultPath, 'killDF.pkl'))
    from statsmodels.formula.api import ols
    import statsmodels.api as sm

    # model = ols("kill ~ C(temperature, Sum) + C(chemical, Sum) + C(method, Sum) + C(temperature, Sum):C(method, Sum)",
    #             data=manufac).fit()
    model = ols("meanKill ~ C(numWolves) + C(sheepSpeedMultiplier) + C(costActionRatio) + C(rewardSensitivityToDistance) + C(numWolves):C(sheepSpeedMultiplier) +  C(numWolves):C(costActionRatio) +  C(numWolves):C(rewardSensitivityToDistance) + C(sheepSpeedMultiplier):C(costActionRatio) +  C(sheepSpeedMultiplier):C(rewardSensitivityToDistance) +  C(costActionRatio):C(rewardSensitivityToDistance)",data=resultDFEasy.reset_index()).fit()

    aov_table = sm.stats.anova_lm(model, typ=3)
    print(aov_table)


    # sensitivityTickLabels = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, '10k']
    sensitivityTickLabels = ['unselfish', '', '', '', '', '', 'selfish']
    costTickLabels = ['.0', '.005', '.01', '.015', '.02', '.025', '.03']
    speedTickLabels = ['0.65x', '0.82x', '1x', '1.15x', '1.3x']
    # levelNames = resultDF.index.names




if __name__ == '__main__':
    main()