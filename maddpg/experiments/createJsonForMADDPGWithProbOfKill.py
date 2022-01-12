import json

def main():
    conditions = dict()

    conditions['Lululucyzs-MBP.lan'] = {
        'numWolvesLevels': [4],
        'sheepSpeedMultiplierLevels': [0.75],
        'costActionRatioList': [0.02, 0.025],
        'rewardSensitivityToDistanceLevels': [0],
        'biteRewardLevels': [0.05]}

    conditions['antman'] = {
        'numWolvesLevels': [6],
        'sheepSpeedMultiplierLevels': [0.75],
        'costActionRatioList': [0.03],
        'rewardSensitivityToDistanceLevels': [0, 1, 2, 10000],
        'biteRewardLevels': [0]}

    conditions['blackwidow'] = {
        'numWolvesLevels': [6],
        'sheepSpeedMultiplierLevels': [1],
        'costActionRatioList': [0.03],
        'rewardSensitivityToDistanceLevels': [0, 1, 2, 10000],
        'biteRewardLevels': [0]}

    conditions['carter'] = {
        'numWolvesLevels': [6],
        'sheepSpeedMultiplierLevels': [0.5],
        'costActionRatioList': [0],
        'rewardSensitivityToDistanceLevels': [0, 1, 2, 10000],
        'biteRewardLevels': [0]}

    conditions['deadpool'] = {
        'numWolvesLevels': [6],
        'sheepSpeedMultiplierLevels': [0.5],
        'costActionRatioList': [0.01],
        'rewardSensitivityToDistanceLevels': [0, 1, 2, 10000],
        'biteRewardLevels': [0]}

    conditions['echo'] = {
        'numWolvesLevels': [6],
        'sheepSpeedMultiplierLevels': [0.5],
        'costActionRatioList': [0.02],
        'rewardSensitivityToDistanceLevels': [0, 1, 2, 10000],
        'biteRewardLevels': [0]}

    conditions['falcon'] = {
        'numWolvesLevels': [6],
        'sheepSpeedMultiplierLevels': [0.5],
        'costActionRatioList': [0.03],
        'rewardSensitivityToDistanceLevels': [0, 1, 2, 10000],
        'biteRewardLevels': [0]}

    conditions['algebra'] = {
        'numWolvesLevels': [2, 3],
        'sheepSpeedMultiplierLevels': [0.5, 0.75, 1],
        'costActionRatioList': [0.02],
        'rewardSensitivityToDistanceLevels': [0],
        'biteRewardLevels': [0]}

    conditions['bernoulli'] = {#
        'numWolvesLevels': [2],
        'sheepSpeedMultiplierLevels': [0.75],
        'costActionRatioList': [0, 0.01, 0.02, 0.03],
        'rewardSensitivityToDistanceLevels': [2],
        'biteRewardLevels': [0]}

    conditions['calculus'] = {
        'numWolvesLevels': [3],
        'sheepSpeedMultiplierLevels': [0.5, 0.75, 1],
        'costActionRatioList': [0.02, 0.03],
        'rewardSensitivityToDistanceLevels': [2],
        'biteRewardLevels': [0]}

    conditions['derivative'] = {
        'numWolvesLevels': [2],
        'sheepSpeedMultiplierLevels': [0.5],
        'costActionRatioList': [0, 0.01, 0.02, 0.03],
        'rewardSensitivityToDistanceLevels': [2],
        'biteRewardLevels': [0]}

    conditions['expectation'] = {
        'numWolvesLevels': [2],
        'sheepSpeedMultiplierLevels': [1.5],
        'costActionRatioList': [0, 0.01, 0.02, 0.03],
        'rewardSensitivityToDistanceLevels': [2],
        'biteRewardLevels': [0]}

    conditions['integration'] = {
        'numWolvesLevels': [3],
        'sheepSpeedMultiplierLevels': [0.75],
        'costActionRatioList': [0, 0.01, 0.02, 0.03],
        'rewardSensitivityToDistanceLevels': [2],
        'biteRewardLevels': [0]}

    conditions['p100b'] = {
        'numWolvesLevels': [6],
        'sheepSpeedMultiplierLevels': [0.75, 1],
        'costActionRatioList': [0.005],
        'rewardSensitivityToDistanceLevels': [10000],
        'biteRewardLevels': [0, 0.05, 0.1]} # not run

    conditions['titanxp'] = {
        'numWolvesLevels': [4],
        'sheepSpeedMultiplierLevels': [1.25],
        'costActionRatioList': [0.02, 0.025],
        'rewardSensitivityToDistanceLevels': [2, 5],
        'biteRewardLevels': [0.5]}

    outputFile = open('conditionsMADDPGWithProbKill.json', 'w')
    json.dump(conditions, outputFile)
    outputFile.close()
    print(conditions.keys())


if __name__ == '__main__':
    main()