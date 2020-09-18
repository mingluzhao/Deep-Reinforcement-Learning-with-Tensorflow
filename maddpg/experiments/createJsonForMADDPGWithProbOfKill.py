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

'numWolvesLevels':                  [6],
'sheepSpeedMultiplierLevels':       [.75, 1],
'costActionRatioList':              [0.01, 0.02, 0.025],
'rewardSensitivityToDistance':      [0, 1, 2, 10000],
'biteRewardLevels':                 [0, 0.05, 0.1]
--------------------------------------------------------------

antman bcdef

'numWolvesLevels':                  [6],
'sheepSpeedMultiplierLevels':       [.75, 1],
'costActionRatioList':              [0.02],
'rewardSensitivityToDistance':      [0, 1, 2, 10000],
'biteRewardLevels':                 [0, 0.05, 0.1]

9.12 left:
'numWolvesLevels':                  [6],
'sheepSpeedMultiplierLevels':       [.75, 1],
'costActionRatioList':              [0.025],
'rewardSensitivityToDistance':      [0, 1, 2, 10000],
'biteRewardLevels':                 [0, 0.05, 0.1]

Antman Blackwidow Falcon:
'numWolvesLevels':                  [6],
'sheepSpeedMultiplierLevels':       [.75],
'costActionRatioList':              [0.025],
'rewardSensitivityToDistance':      [0, 1, 2, 10000],
'biteRewardLevels':                 [0, 0.05, 0.1]**

algebra calculus Deadpool:
'numWolvesLevels':                  [6],
'sheepSpeedMultiplierLevels':       [1],
'costActionRatioList':              [0.025],
'rewardSensitivityToDistance':      [0, 1, 2, 10000],**10000left
'biteRewardLevels':                 [0, 0.05, 0.1] 
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
        'numWolvesLevels': [4, 5],
        'sheepSpeedMultiplierLevels': [0.5, 0.75, 1],
        'costActionRatioList': [0],
        'rewardSensitivityToDistanceLevels': [10000],
        'biteRewardLevels': [0]}

    conditions['blackwidow'] = {
        'numWolvesLevels': [4, 5],
        'sheepSpeedMultiplierLevels': [0.5, 0.75, 1],
        'costActionRatioList': [0.01],
        'rewardSensitivityToDistanceLevels': [10000],
        'biteRewardLevels': [0]}

    conditions['carter'] = {
        'numWolvesLevels': [4, 5],
        'sheepSpeedMultiplierLevels': [0.5, 0.75, 1],
        'costActionRatioList': [0.02],
        'rewardSensitivityToDistanceLevels': [10000],
        'biteRewardLevels': [0]}

    conditions['deadpool'] = {
        'numWolvesLevels': [4, 5],
        'sheepSpeedMultiplierLevels': [0.5, 0.75, 1],
        'costActionRatioList': [0.03],
        'rewardSensitivityToDistanceLevels': [10000],
        'biteRewardLevels': [0]}

    conditions['echo'] = {
        'numWolvesLevels': [2, 3],
        'sheepSpeedMultiplierLevels': [0.5, 0.75, 1],
        'costActionRatioList': [0],
        'rewardSensitivityToDistanceLevels': [10000],
        'biteRewardLevels': [0]}

    conditions['falcon'] = {
        'numWolvesLevels': [2, 3],
        'sheepSpeedMultiplierLevels': [0.5, 0.75, 1],
        'costActionRatioList': [0.01],
        'rewardSensitivityToDistanceLevels': [10000],
        'biteRewardLevels': [0]}

    conditions['algebra'] = {
        'numWolvesLevels': [2, 3],
        'sheepSpeedMultiplierLevels': [0.5, 0.75, 1],
        'costActionRatioList': [0.02],
        'rewardSensitivityToDistanceLevels': [10000],
        'biteRewardLevels': [0]}

    conditions['bernoulli'] = {#
        'numWolvesLevels': [2],
        'sheepSpeedMultiplierLevels': [0.5],
        'costActionRatioList': [0, 0.01, 0.02, 0.03],
        'rewardSensitivityToDistanceLevels': [2],
        'biteRewardLevels': [0]}

    conditions['calculus'] = {
        'numWolvesLevels': [2, 3],
        'sheepSpeedMultiplierLevels': [0.5, 0.75, 1],
        'costActionRatioList': [0.03],
        'rewardSensitivityToDistanceLevels': [10000],
        'biteRewardLevels': [0]}

    conditions['derivative'] = {
        'numWolvesLevels': [5],
        'sheepSpeedMultiplierLevels': [0.5],
        'costActionRatioList': [0, 0.01, 0.02, 0.03],
        'rewardSensitivityToDistanceLevels': [2],
        'biteRewardLevels': [0]}

    conditions['expectation'] = {
        'numWolvesLevels': [4],
        'sheepSpeedMultiplierLevels': [0.5],
        'costActionRatioList': [0, 0.01, 0.02, 0.03],
        'rewardSensitivityToDistanceLevels': [2],
        'biteRewardLevels': [0]}

    conditions['integration'] = {
        'numWolvesLevels': [3],
        'sheepSpeedMultiplierLevels': [0.5],
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