import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))

import itertools as it
import matplotlib.pyplot as plt
import pickle
from functionTools.loadSaveModel import saveToPickle, restoreVariables
import seaborn as sns

def loadFromPickle(path):
    pickleIn = open(path, 'rb')
    object = pickle.load(pickleIn)
    pickleIn.close()
    return object


def main():
    resultPath = os.path.join(dirName, '..', 'evalResults')
    resultLoc = os.path.join(resultPath, 'data.pkl')
    resultDF = loadFromPickle(resultLoc)

    independentVariables = dict()
    independentVariables['numWolves'] = [2, 3, 4, 5, 6]
    independentVariables['sheepSpeedMultiplier'] = [0.5, 0.625, 0.75, 0.875, 1.0]
    independentVariables['costActionRatio'] = [.0, .005, .01, .015, .02, .025, .03]
    independentVariables['rewardSensitivityToDistance'] = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    ## linear regression for numPredators vs killNumber
    fig1DF = resultDF.groupby(['numWolves', 'rewardSensitivityToDistance']).mean()['meanKill'].reset_index()

    ### selfish agents: linear increase
    selfishDF = fig1DF[fig1DF['rewardSensitivityToDistance'] == 3.0]
    sns.lmplot(x="numWolves", y="meanKill", data=selfishDF)
    plt.xlabel('number of predators')
    plt.ylabel('mean episode kill')
    plt.savefig(os.path.join(resultPath, 'regression_selfish_numPredator.pdf'), dpi=600)

    ### unselfish agents: decrease
    unselfishDF = fig1DF[fig1DF['rewardSensitivityToDistance'] == 0.0]
    sns.lmplot(x="numWolves", y="meanKill", data=unselfishDF)
    plt.xlabel('number of predators')
    plt.ylabel('mean episode kill')
    plt.savefig(os.path.join(resultPath, 'regression_unselfish_numPredator.pdf'), dpi=600)

    ## the  more  selfish  the  predators,  the  better  their  performance
    wolves6df = fig1DF[fig1DF['numWolves'] == 6]
    sns.lmplot(x="rewardSensitivityToDistance", y="meanKill", data=wolves6df)
    plt.xlabel('selfish index')
    plt.ylabel('mean episode kill')
    plt.savefig(os.path.join(resultPath, 'regression_6Predator.pdf'), dpi=600)

    # The  performance  of  all  agents  decreases  as  the  action  cost increases
    fig2DF = resultDF.groupby(['costActionRatio']).mean()['meanKill'].reset_index()
    sns.lmplot(x="costActionRatio", y="meanKill", data=fig2DF)
    plt.xlabel('cost-action ratio')
    plt.ylabel('mean episode kill')
    plt.savefig(os.path.join(resultPath, 'regression_actionCost.pdf'), dpi=600)



if __name__ == '__main__':
    main()
