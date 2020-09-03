import json


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
        'numWolvesLevels': [2, 3],
        'numSheepsLevels': [1],
        'numBlocksLevels': [2],
        'maxTimeStepLevels': [75],
        'sheepSpeedMultiplierLevels': [1.0],
        'individualRewardWolfLevels': [1],
        'costActionRatioList': [0.03]}

    conditions['blackwidow'] = {
        'numWolvesLevels': [2, 3],
        'numSheepsLevels': [1],
        'numBlocksLevels': [2],
        'maxTimeStepLevels': [75],
        'sheepSpeedMultiplierLevels': [1.0],
        'individualRewardWolfLevels': [1],
        'costActionRatioList': [0.03]}

    conditions['carter'] = {
        'numWolvesLevels': [2, 3],
        'numSheepsLevels': [1],
        'numBlocksLevels': [2],
        'maxTimeStepLevels': [75],
        'sheepSpeedMultiplierLevels': [1.0],
        'individualRewardWolfLevels': [1],
        'costActionRatioList': [0.03]}

    conditions['dealpool'] = {
        'numWolvesLevels': [2, 3],
        'numSheepsLevels': [1],
        'numBlocksLevels': [2],
        'maxTimeStepLevels': [75],
        'sheepSpeedMultiplierLevels': [1.0],
        'individualRewardWolfLevels': [1],
        'costActionRatioList': [0.03]}

    conditions['echo'] = {
        'numWolvesLevels': [2, 3],
        'numSheepsLevels': [1],
        'numBlocksLevels': [2],
        'maxTimeStepLevels': [75],
        'sheepSpeedMultiplierLevels': [1.0],
        'individualRewardWolfLevels': [1],
        'costActionRatioList': [0.03]}

    conditions['falcon'] = {
        'numWolvesLevels': [2, 3],
        'numSheepsLevels': [1],
        'numBlocksLevels': [2],
        'maxTimeStepLevels': [75],
        'sheepSpeedMultiplierLevels': [1.0],
        'individualRewardWolfLevels': [1],
        'costActionRatioList': [0.03]}

    conditions['algebra'] = {
        'numWolvesLevels': [2, 3],
        'numSheepsLevels': [1],
        'numBlocksLevels': [2],
        'maxTimeStepLevels': [75],
        'sheepSpeedMultiplierLevels': [1.0],
        'individualRewardWolfLevels': [1],
        'costActionRatioList': [0.03]}

    conditions['bernoulli'] = {
        'numWolvesLevels': [2, 3],
        'numSheepsLevels': [1],
        'numBlocksLevels': [2],
        'maxTimeStepLevels': [75],
        'sheepSpeedMultiplierLevels': [1.0],
        'individualRewardWolfLevels': [1],
        'costActionRatioList': [0.03]}

    conditions['calculus'] = {
        'numWolvesLevels': [2, 3],
        'numSheepsLevels': [1],
        'numBlocksLevels': [2],
        'maxTimeStepLevels': [75],
        'sheepSpeedMultiplierLevels': [1.0],
        'individualRewardWolfLevels': [1],
        'costActionRatioList': [0.03]}

    conditions['derivative'] = {
        'numWolvesLevels': [2, 3],
        'numSheepsLevels': [1],
        'numBlocksLevels': [2],
        'maxTimeStepLevels': [75],
        'sheepSpeedMultiplierLevels': [1.0],
        'individualRewardWolfLevels': [1],
        'costActionRatioList': [0.03]}

    conditions['expectation'] = {
        'numWolvesLevels': [2, 3],
        'numSheepsLevels': [1],
        'numBlocksLevels': [2],
        'maxTimeStepLevels': [75],
        'sheepSpeedMultiplierLevels': [1.0],
        'individualRewardWolfLevels': [1],
        'costActionRatioList': [0.03]}

    conditions['integration'] = {
        'numWolvesLevels': [2, 3],
        'numSheepsLevels': [1],
        'numBlocksLevels': [2],
        'maxTimeStepLevels': [75],
        'sheepSpeedMultiplierLevels': [1.0],
        'individualRewardWolfLevels': [1],
        'costActionRatioList': [0.03]}

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