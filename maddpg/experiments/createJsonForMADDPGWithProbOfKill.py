import json

'''
until 9.7 have models for:

'numWolvesLevels':                  [4],
'sheepSpeedMultiplierLevels':       [.75, 1, 1.25],
'costActionRatioList':              [0, 0.005, 0.01, 0.02, 0.025],
'rewardSensitivityToDistance':      [0, 2, 5, 10000],
'biteRewardLevels':                 [0.05, 0.1, 0.5]

'''

'''
9.9 new:

'numWolvesLevels':                  [6],
'sheepSpeedMultiplierLevels':       [.75, 1],
'costActionRatioList':              [0, 0.005, 0.01, 0.02, 0.025],
'rewardSensitivityToDistance':      [0, 1, 2, 10000],
'biteRewardLevels':                 [0, 0.05, 0.1]

'''

'''
antman bcdef

'numWolvesLevels':                  [6],
'sheepSpeedMultiplierLevels':       [.75, 1],
'costActionRatioList':              [0],
'rewardSensitivityToDistance':      [0, 1, 2, 10000],
'biteRewardLevels':                 [0, 0.05, 0.1]

algebra bcdei
'numWolvesLevels':                  [6],
'sheepSpeedMultiplierLevels':       [.75, 1],
'costActionRatioList':              [0.005],
'rewardSensitivityToDistance':      [0, 1, 2, 10000],
'biteRewardLevels':                 [0, 0.05, 0.1]

left:

'''


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
        'costActionRatioList': [0],
        'rewardSensitivityToDistanceLevels': [0, 1, 2, 10000],
        'biteRewardLevels': [0]}

    conditions['blackwidow'] = {
        'numWolvesLevels': [6],
        'sheepSpeedMultiplierLevels': [0.75],
        'costActionRatioList': [0],
        'rewardSensitivityToDistanceLevels': [0, 1, 2, 10000],
        'biteRewardLevels': [0.05]}

    conditions['carter'] = {
        'numWolvesLevels': [6],
        'sheepSpeedMultiplierLevels': [0.75],
        'costActionRatioList': [0],
        'rewardSensitivityToDistanceLevels': [0, 1, 2, 10000],
        'biteRewardLevels': [0.1]}

    conditions['deadpool'] = {
        'numWolvesLevels': [6],
        'sheepSpeedMultiplierLevels': [1],
        'costActionRatioList': [0],
        'rewardSensitivityToDistanceLevels': [0, 1, 2, 10000],
        'biteRewardLevels': [0]}

    conditions['echo'] = {
        'numWolvesLevels': [6],
        'sheepSpeedMultiplierLevels': [1],
        'costActionRatioList': [0],
        'rewardSensitivityToDistanceLevels': [0, 1, 2, 10000],
        'biteRewardLevels': [0.05]}

    conditions['falcon'] = {
        'numWolvesLevels': [6],
        'sheepSpeedMultiplierLevels': [1],
        'costActionRatioList': [0],
        'rewardSensitivityToDistanceLevels': [0, 1, 2, 10000],
        'biteRewardLevels': [0.1]}

    conditions['algebra'] = {
        'numWolvesLevels': [6],
        'sheepSpeedMultiplierLevels': [0.75],
        'costActionRatioList': [0.005],
        'rewardSensitivityToDistanceLevels': [0],
        'biteRewardLevels': [0, 0.05, 0.1]}

    conditions['bernoulli'] = {
        'numWolvesLevels': [6],
        'sheepSpeedMultiplierLevels': [1],
        'costActionRatioList': [0.005],
        'rewardSensitivityToDistanceLevels': [0],
        'biteRewardLevels': [0, 0.05, 0.1]}

    conditions['calculus'] = {
        'numWolvesLevels': [6],
        'sheepSpeedMultiplierLevels': [0.75],
        'costActionRatioList': [0.005],
        'rewardSensitivityToDistanceLevels': [1],
        'biteRewardLevels': [0, 0.05, 0.1]}

    conditions['derivative'] = {
        'numWolvesLevels': [6],
        'sheepSpeedMultiplierLevels': [1],
        'costActionRatioList': [0.005],
        'rewardSensitivityToDistanceLevels': [1],
        'biteRewardLevels': [0, 0.05, 0.1]}

    conditions['expectation'] = {
        'numWolvesLevels': [6],
        'sheepSpeedMultiplierLevels': [0.75],
        'costActionRatioList': [0.005],
        'rewardSensitivityToDistanceLevels': [2],
        'biteRewardLevels': [0, 0.05, 0.1]}

    conditions['integration'] = {
        'numWolvesLevels': [6],
        'sheepSpeedMultiplierLevels': [1],
        'costActionRatioList': [0.005],
        'rewardSensitivityToDistanceLevels': [2],
        'biteRewardLevels': [0, 0.05, 0.1]}

    conditions['p100b'] = {
        'numWolvesLevels': [6],
        'sheepSpeedMultiplierLevels': [0.75, 1],
        'costActionRatioList': [0.005],
        'rewardSensitivityToDistanceLevels': [10000],
        'biteRewardLevels': [0, 0.05, 0.1]}

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