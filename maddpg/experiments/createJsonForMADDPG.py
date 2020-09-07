import json

'''
'numWolvesLevels': [2, 3, 4, 5, 6],
'numSheepsLevels': [1],
'numBlocksLevels': [2],
'maxTimeStepLevels': [75],
'sheepSpeedMultiplierLevels': [0.5, 1.5],
'individualRewardWolfLevels': [0, 1],
'costActionRatioList': [0, 0.01, 0.02, 0.03]
'''

'''
left:
        'numWolvesLevels': [2, 3, 4],
        'numSheepsLevels': [1],
        'numBlocksLevels': [2],
        'maxTimeStepLevels': [75],
        'sheepSpeedMultiplierLevels': [0.5, 1.5],
        'individualRewardWolfLevels': [0, 1],
        'costActionRatioList': [0.03]}
'''

def main():
    conditions = dict()
    conditions['Lululucyzs-MBP.lan'] = {
        'numWolvesLevels': [2, 3],
        'numSheepsLevels': [1],
        'numBlocksLevels': [2],
        'maxTimeStepLevels': [75],
        'sheepSpeedMultiplierLevels': [1.0],
        'individualRewardWolfLevels': [1],
        'costActionRatioList': [0.03]}

    conditions['antman'] = {
        'numWolvesLevels': [2, 3, 4],
        'numSheepsLevels': [1],
        'numBlocksLevels': [2],
        'maxTimeStepLevels': [75],
        'sheepSpeedMultiplierLevels': [0.5],
        'individualRewardWolfLevels': [0, 1],
        'costActionRatioList': [0]}

    conditions['blackwidow'] = {
        'numWolvesLevels': [2, 3, 4],
        'numSheepsLevels': [1],
        'numBlocksLevels': [2],
        'maxTimeStepLevels': [75],
        'sheepSpeedMultiplierLevels': [1.5],
        'individualRewardWolfLevels': [0, 1],
        'costActionRatioList': [0]}

    conditions['carter'] = {
        'numWolvesLevels': [2, 3, 4],
        'numSheepsLevels': [1],
        'numBlocksLevels': [2],
        'maxTimeStepLevels': [75],
        'sheepSpeedMultiplierLevels': [0.5],
        'individualRewardWolfLevels': [0, 1],
        'costActionRatioList': [0.01]}

    conditions['deadpool'] = {
        'numWolvesLevels': [2, 3, 4],
        'numSheepsLevels': [1],
        'numBlocksLevels': [2],
        'maxTimeStepLevels': [75],
        'sheepSpeedMultiplierLevels': [1.5],
        'individualRewardWolfLevels': [0, 1],
        'costActionRatioList': [0.01]}

    conditions['echo'] = {
        'numWolvesLevels': [2, 3, 4],
        'numSheepsLevels': [1],
        'numBlocksLevels': [2],
        'maxTimeStepLevels': [75],
        'sheepSpeedMultiplierLevels': [0.5],
        'individualRewardWolfLevels': [0, 1],
        'costActionRatioList': [0.02]}

    conditions['falcon'] = {
        'numWolvesLevels': [2, 3, 4],
        'numSheepsLevels': [1],
        'numBlocksLevels': [2],
        'maxTimeStepLevels': [75],
        'sheepSpeedMultiplierLevels': [1.5],
        'individualRewardWolfLevels': [0, 1],
        'costActionRatioList': [0.02]}

    conditions['algebra'] = {
        'numWolvesLevels': [6],
        'numSheepsLevels': [1],
        'numBlocksLevels': [2],
        'maxTimeStepLevels': [75],
        'sheepSpeedMultiplierLevels': [0.5],
        'individualRewardWolfLevels': [0],
        'costActionRatioList': [0, 0.01, 0.02, 0.03]}

    conditions['bernoulli'] = {
        'numWolvesLevels': [6],
        'numSheepsLevels': [1],
        'numBlocksLevels': [2],
        'maxTimeStepLevels': [75],
        'sheepSpeedMultiplierLevels': [0.5],
        'individualRewardWolfLevels': [1],
        'costActionRatioList': [0, 0.01, 0.02, 0.03]}

    conditions['calculus'] = {
        'numWolvesLevels': [6],
        'numSheepsLevels': [1],
        'numBlocksLevels': [2],
        'maxTimeStepLevels': [75],
        'sheepSpeedMultiplierLevels': [1.5],
        'individualRewardWolfLevels': [0],
        'costActionRatioList': [0, 0.01, 0.02, 0.03]}

    conditions['derivative'] = {
        'numWolvesLevels': [6],
        'numSheepsLevels': [1],
        'numBlocksLevels': [2],
        'maxTimeStepLevels': [75],
        'sheepSpeedMultiplierLevels': [1.5],
        'individualRewardWolfLevels': [1],
        'costActionRatioList': [0, 0.01, 0.02, 0.03]}

    conditions['expectation'] = {
        'numWolvesLevels': [5],
        'numSheepsLevels': [1],
        'numBlocksLevels': [2],
        'maxTimeStepLevels': [75],
        'sheepSpeedMultiplierLevels': [0.5],
        'individualRewardWolfLevels': [0],
        'costActionRatioList': [0, 0.01, 0.02, 0.03]}

    conditions['integration'] = {
        'numWolvesLevels': [5],
        'numSheepsLevels': [1],
        'numBlocksLevels': [2],
        'maxTimeStepLevels': [75],
        'sheepSpeedMultiplierLevels': [0.5],
        'individualRewardWolfLevels': [1],
        'costActionRatioList': [0, 0.01, 0.02, 0.03]}

    conditions['p100b'] = {
        'numWolvesLevels': [5],
        'numSheepsLevels': [1],
        'numBlocksLevels': [2],
        'maxTimeStepLevels': [75],
        'sheepSpeedMultiplierLevels': [1.5],
        'individualRewardWolfLevels': [0, 1],
        'costActionRatioList': [0, 0.01, 0.02, 0.03]}

    conditions['titanxp'] = {
        'numWolvesLevels': [2, 3],
        'numSheepsLevels': [1],
        'numBlocksLevels': [2],
        'maxTimeStepLevels': [75],
        'sheepSpeedMultiplierLevels': [1.0],
        'individualRewardWolfLevels': [1],
        'costActionRatioList': [0.03]}

    outputFile = open('conditions.json', 'w')
    json.dump(conditions, outputFile)
    outputFile.close()
    print(conditions.keys())



if __name__ == '__main__':
    main()