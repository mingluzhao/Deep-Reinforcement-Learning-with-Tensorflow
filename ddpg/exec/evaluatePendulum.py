import pandas as pd
import pylab as plt
import numpy as np
import seaborn as sns

def evaluateModel(modelDf):
    reward = np.random.uniform(-1, 1)
    resultSe = pd.Series({'reward':reward})
    return resultSe

def drawHeatmapPlot(plotDf, ax):
    plotDf = plotDf.reset_index().pivot(columns= 'sheepXPosition', index = 'sheepYPosition', values = 'reward')
    sns.heatmap(plotDf, ax = ax)

def drawLinePlot(plotDf, ax):
    for sheepYPosition, subDf in plotDf.groupby('sheepYPosition'):
        subDf = subDf.droplevel('sheepYPosition')
        subDf.plot.line(ax = ax, label = 'sheepYPosition = {}'.format(sheepYPosition), y = 'reward', marker = 'o')

def main():
    wolfXPosition = [-10, -5, 0, 5, 10]
    wolfYPosition = [-10, -5, 0, 5, 10]
    sheepXPosition = [-20, -10, 0, 10, 20]
    sheepYPosition = [-20, -10, 0, 10, 20]

    levelValues = [wolfXPosition, wolfYPosition, sheepXPosition, sheepYPosition]
    levelNames = ["wolfXPosition", "wolfYPosition", "sheepXPosition", "sheepYPosition"]

    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)

    toSplitFrame = pd.DataFrame(index = modelIndex)

    modelResultDf = toSplitFrame.groupby(levelNames).apply(evaluateModel)

    fig = plt.figure()
    plotLevels = ['sheepXPosition', 'sheepYPosition']
    plotRowNum = len(wolfYPosition)
    plotColNum = len(wolfXPosition)
    plotCounter = 1

    for (key, plotDf) in modelResultDf.groupby(['wolfYPosition', 'wolfXPosition']):
        plotDf.index = plotDf.index.droplevel(['wolfYPosition', 'wolfXPosition'])
        ax = fig.add_subplot(plotRowNum, plotColNum, plotCounter)
        #drawHeatmapPlot(plotDf, ax)
        drawLinePlot(plotDf, ax)
        plotCounter+=1

    fig.text(x = 0.5, y = 0.04, s = 'wolfXPosition', ha = 'center', va = 'center')
    fig.text(x = 0.05, y = 0.5, s = 'wolfYPosition', ha = 'center', va = 'center', rotation=90)

    plt.show()









if __name__ == "__main__":
    main()
