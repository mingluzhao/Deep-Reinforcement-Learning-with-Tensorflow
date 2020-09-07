import json

'''
start on 9.3
'numWolvesLevels':                  [4],
'sheepSpeedMultiplierLevels':       [.75, 1, 1.25],
'costActionRatioList':              [0.005, 0.01, 0.02, 0.025],
'rewardSensitivityToDistance':      [0, 10000],
'biteRewardLevels':                 [0.05, 0.1, 0.5]

----------------------------------------------------------------
for abcdef antman:
'numWolvesLevels':                  [4],
'sheepSpeedMultiplierLevels':       [.75, 1, 1.25],
'costActionRatioList':              [0.005, 0.01], #
'rewardSensitivityToDistance':      [0, 10000],
'biteRewardLevels':                 [0.05, 0.1, 0.5]

abcdei algebra:
'numWolvesLevels':                  [4],
'sheepSpeedMultiplierLevels':       [.75, 1, 1.25],
'costActionRatioList':              [0.02, 0.025],
'rewardSensitivityToDistance':      [0, 10000],
'biteRewardLevels':                 [0.05, 0.1]#

left:
'numWolvesLevels':                  [4],
'sheepSpeedMultiplierLevels':       [.75, 1, 1.25],
'costActionRatioList':              [0.02, 0.025],
'rewardSensitivityToDistance':      [0, 10000],
'biteRewardLevels':                 [0.5]

'''

# def main():
#     conditions = dict()
#
#     conditions['Lululucyzs-MBP.lan'] = {
#         'numWolvesLevels': [4],
#         'sheepSpeedMultiplierLevels': [0.75],
#         'costActionRatioList': [0.02, 0.025],
#         'rewardSensitivityToDistanceLevels': [0],
#         'biteRewardLevels': [0.05]}
#
#     conditions['antman'] = {
#         'numWolvesLevels': [4],
#         'sheepSpeedMultiplierLevels': [0.75],
#         'costActionRatioList': [0.005],
#         'rewardSensitivityToDistanceLevels': [0, 10000],
#         'biteRewardLevels': [0.05, 0.1, 0.5]}
#
#     conditions['blackwidow'] = {
#         'numWolvesLevels': [4],
#         'sheepSpeedMultiplierLevels': [1],
#         'costActionRatioList': [0.005],
#         'rewardSensitivityToDistanceLevels': [0, 10000],
#         'biteRewardLevels': [0.05, 0.1, 0.5]}
#
#     conditions['carter'] = {
#         'numWolvesLevels': [4],
#         'sheepSpeedMultiplierLevels': [1.25],
#         'costActionRatioList': [0.005],
#         'rewardSensitivityToDistanceLevels': [0, 10000],
#         'biteRewardLevels': [0.05, 0.1, 0.5]}
#
#     conditions['deadpool'] = {
#         'numWolvesLevels': [4],
#         'sheepSpeedMultiplierLevels': [0.75],
#         'costActionRatioList': [0.01],
#         'rewardSensitivityToDistanceLevels': [0, 10000],
#         'biteRewardLevels': [0.05, 0.1, 0.5]}
#
#     conditions['echo'] = {
#         'numWolvesLevels': [4],
#         'sheepSpeedMultiplierLevels': [1],
#         'costActionRatioList': [0.01],
#         'rewardSensitivityToDistanceLevels': [0, 10000],
#         'biteRewardLevels': [0.05, 0.1, 0.5]}
#
#     conditions['falcon'] = {
#         'numWolvesLevels': [4],
#         'sheepSpeedMultiplierLevels': [1.25],
#         'costActionRatioList': [0.01],
#         'rewardSensitivityToDistanceLevels': [0, 10000],
#         'biteRewardLevels': [0.05, 0.1, 0.5]}
#
#     conditions['algebra'] = {
#         'numWolvesLevels': [4],
#         'sheepSpeedMultiplierLevels': [0.75],
#         'costActionRatioList': [0.02, 0.025],
#         'rewardSensitivityToDistanceLevels': [0, 10000],
#         'biteRewardLevels': [0.05]}
#
#     conditions['bernoulli'] = {
#         'numWolvesLevels': [4],
#         'sheepSpeedMultiplierLevels': [1],
#         'costActionRatioList': [0.02, 0.025],
#         'rewardSensitivityToDistanceLevels': [0, 10000],
#         'biteRewardLevels': [0.05]}
#
#     conditions['calculus'] = {
#         'numWolvesLevels': [4],
#         'sheepSpeedMultiplierLevels': [1.25],
#         'costActionRatioList': [0.02, 0.025],
#         'rewardSensitivityToDistanceLevels': [0, 10000],
#         'biteRewardLevels': [0.05]}
#
#     conditions['derivative'] = {
#         'numWolvesLevels': [4],
#         'sheepSpeedMultiplierLevels': [0.75],
#         'costActionRatioList': [0.02, 0.025],
#         'rewardSensitivityToDistanceLevels': [0, 10000],
#         'biteRewardLevels': [0.1]}
#
#     conditions['expectation'] = {
#         'numWolvesLevels': [4],
#         'sheepSpeedMultiplierLevels': [1],
#         'costActionRatioList': [0.02, 0.025],
#         'rewardSensitivityToDistanceLevels': [0, 10000],
#         'biteRewardLevels': [0.1]}
#
#     conditions['integration'] = {
#         'numWolvesLevels': [4],
#         'sheepSpeedMultiplierLevels': [1.25],
#         'costActionRatioList': [0.02, 0.025],
#         'rewardSensitivityToDistanceLevels': [0, 10000],
#         'biteRewardLevels': [0.1]}
#
#     conditions['p100b'] = {
#         'numWolvesLevels': [4],
#         'sheepSpeedMultiplierLevels': [.75, 1],
#         'costActionRatioList': [0.02, 0.025],
#         'rewardSensitivityToDistanceLevels': [0, 10000],
#         'biteRewardLevels': [0.5]}
#
#     conditions['titanxp'] = {
#         'numWolvesLevels': [4],
#         'sheepSpeedMultiplierLevels': [1.25],
#         'costActionRatioList': [0.02, 0.025],
#         'rewardSensitivityToDistanceLevels': [0, 10000],
#         'biteRewardLevels': [0.5]}
#
#     outputFile = open('conditionsMADDPGWithProbKill.json', 'w')
#     json.dump(conditions, outputFile)
#     outputFile.close()
#     print(conditions.keys())

'''
start on 9.4
'numWolvesLevels':                  [4],
'sheepSpeedMultiplierLevels':       [.75, 1, 1.25],
'costActionRatioList':              [0.005, 0.01, 0.02, 0.025],
'rewardSensitivityToDistance':      [2, 5],
'biteRewardLevels':                 [0.05, 0.1, 0.5]

----------------------------------------------------------------
for abcdef antman:
'numWolvesLevels':                  [4],
'sheepSpeedMultiplierLevels':       [.75, 1, 1.25],
'costActionRatioList':              [0.005, 0.01], #
'rewardSensitivityToDistance':      [2, 5],
'biteRewardLevels':                 [0.05, 0.1, 0.5]

abcdei algebra:
'numWolvesLevels':                  [4],
'sheepSpeedMultiplierLevels':       [.75, 1, 1.25],
'costActionRatioList':              [0.02, 0.025],
'rewardSensitivityToDistance':      [2, 5],
'biteRewardLevels':                 [0.05, 0.1]#

left:
'numWolvesLevels':                  [4],
'sheepSpeedMultiplierLevels':       [.75, 1, 1.25],
'costActionRatioList':              [0.02, 0.025],
'rewardSensitivityToDistance':      [2, 5],
'biteRewardLevels':                 [0.5]

'''


# def main():
#     conditions = dict()
#
#     conditions['Lululucyzs-MBP.lan'] = {
#         'numWolvesLevels': [4],
#         'sheepSpeedMultiplierLevels': [0.75],
#         'costActionRatioList': [0.02, 0.025],
#         'rewardSensitivityToDistanceLevels': [0],
#         'biteRewardLevels': [0.05]}
#
#     conditions['antman'] = {
#         'numWolvesLevels': [4],
#         'sheepSpeedMultiplierLevels': [0.75],
#         'costActionRatioList': [0.005],
#         'rewardSensitivityToDistanceLevels': [2, 5],
#         'biteRewardLevels': [0.05, 0.1, 0.5]}
#
#     conditions['blackwidow'] = {
#         'numWolvesLevels': [4],
#         'sheepSpeedMultiplierLevels': [1],
#         'costActionRatioList': [0.005],
#         'rewardSensitivityToDistanceLevels': [2, 5],
#         'biteRewardLevels': [0.05, 0.1, 0.5]}
#
#     conditions['carter'] = {
#         'numWolvesLevels': [4],
#         'sheepSpeedMultiplierLevels': [1.25],
#         'costActionRatioList': [0.005],
#         'rewardSensitivityToDistanceLevels': [2, 5],
#         'biteRewardLevels': [0.05, 0.1, 0.5]}
#
#     conditions['deadpool'] = {
#         'numWolvesLevels': [4],
#         'sheepSpeedMultiplierLevels': [0.75],
#         'costActionRatioList': [0.01],
#         'rewardSensitivityToDistanceLevels': [2, 5],
#         'biteRewardLevels': [0.05, 0.1, 0.5]}
#
#     conditions['echo'] = {
#         'numWolvesLevels': [4],
#         'sheepSpeedMultiplierLevels': [1],
#         'costActionRatioList': [0.01],
#         'rewardSensitivityToDistanceLevels': [2, 5],
#         'biteRewardLevels': [0.05, 0.1, 0.5]}
#
#     conditions['falcon'] = {
#         'numWolvesLevels': [4],
#         'sheepSpeedMultiplierLevels': [1.25],
#         'costActionRatioList': [0.01],
#         'rewardSensitivityToDistanceLevels': [2, 5],
#         'biteRewardLevels': [0.05, 0.1, 0.5]}
#
#     conditions['algebra'] = {
#         'numWolvesLevels': [4],
#         'sheepSpeedMultiplierLevels': [0.75],
#         'costActionRatioList': [0.02, 0.025],
#         'rewardSensitivityToDistanceLevels': [2, 5],
#         'biteRewardLevels': [0.05]}
#
#     conditions['bernoulli'] = {
#         'numWolvesLevels': [4],
#         'sheepSpeedMultiplierLevels': [1],
#         'costActionRatioList': [0.02, 0.025],
#         'rewardSensitivityToDistanceLevels': [2, 5],
#         'biteRewardLevels': [0.05]}
#
#     conditions['calculus'] = {
#         'numWolvesLevels': [4],
#         'sheepSpeedMultiplierLevels': [1.25],
#         'costActionRatioList': [0.02, 0.025],
#         'rewardSensitivityToDistanceLevels': [2, 5],
#         'biteRewardLevels': [0.05]}
#
#     conditions['derivative'] = {
#         'numWolvesLevels': [4],
#         'sheepSpeedMultiplierLevels': [0.75],
#         'costActionRatioList': [0.02, 0.025],
#         'rewardSensitivityToDistanceLevels': [2, 5],
#         'biteRewardLevels': [0.1]}
#
#     conditions['expectation'] = {
#         'numWolvesLevels': [4],
#         'sheepSpeedMultiplierLevels': [1],
#         'costActionRatioList': [0.02, 0.025],
#         'rewardSensitivityToDistanceLevels': [2, 5],
#         'biteRewardLevels': [0.1]}
#
#     conditions['integration'] = {
#         'numWolvesLevels': [4],
#         'sheepSpeedMultiplierLevels': [1.25],
#         'costActionRatioList': [0.02, 0.025],
#         'rewardSensitivityToDistanceLevels': [2, 5],
#         'biteRewardLevels': [0.1]}
#
#     conditions['p100b'] = {
#         'numWolvesLevels': [4],
#         'sheepSpeedMultiplierLevels': [.75, 1],
#         'costActionRatioList': [0.02, 0.025],
#         'rewardSensitivityToDistanceLevels': [2, 5],
#         'biteRewardLevels': [0.5]}
#
#     conditions['titanxp'] = {
#         'numWolvesLevels': [4],
#         'sheepSpeedMultiplierLevels': [1.25],
#         'costActionRatioList': [0.02, 0.025],
#         'rewardSensitivityToDistanceLevels': [2, 5],
#         'biteRewardLevels': [0.5]}
#
#     outputFile = open('conditionsMADDPGWithProbKill.json', 'w')
#     json.dump(conditions, outputFile)
#     outputFile.close()
#     print(conditions.keys())

'''
until 9.6 have models for:

'numWolvesLevels':                  [4],
'sheepSpeedMultiplierLevels':       [.75, 1, 1.25],
'costActionRatioList':              [0.005, 0.01, 0.02, 0.025],
'rewardSensitivityToDistance':      [0, 2, 5, 10000],
'biteRewardLevels':                 [0.05, 0.1, 0.5]

'''


'''
9.7 add:

'numWolvesLevels':                  [4],
'sheepSpeedMultiplierLevels':       [.75, 1, 1.25],
'costActionRatioList':              [0],
'rewardSensitivityToDistance':      [0, 2, 5, 10000],
'biteRewardLevels':                 [0.05, 0.1, 0.5]

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
        'numWolvesLevels': [4],
        'sheepSpeedMultiplierLevels': [0.75],
        'costActionRatioList': [0],
        'rewardSensitivityToDistanceLevels': [2, 5],
        'biteRewardLevels': [0.05, 0.1, 0.5]}

    conditions['blackwidow'] = {
        'numWolvesLevels': [4],
        'sheepSpeedMultiplierLevels': [1],
        'costActionRatioList': [0],
        'rewardSensitivityToDistanceLevels': [2, 5],
        'biteRewardLevels': [0.05, 0.1, 0.5]}

    conditions['carter'] = {
        'numWolvesLevels': [4],
        'sheepSpeedMultiplierLevels': [1.25],
        'costActionRatioList': [0],
        'rewardSensitivityToDistanceLevels': [2, 5],
        'biteRewardLevels': [0.05, 0.1, 0.5]}

    conditions['deadpool'] = {
        'numWolvesLevels': [4],
        'sheepSpeedMultiplierLevels': [0.75],
        'costActionRatioList': [0],
        'rewardSensitivityToDistanceLevels': [0, 10000],
        'biteRewardLevels': [0.05, 0.1, 0.5]}

    conditions['echo'] = {
        'numWolvesLevels': [4],
        'sheepSpeedMultiplierLevels': [1],
        'costActionRatioList': [0],
        'rewardSensitivityToDistanceLevels': [0, 10000],
        'biteRewardLevels': [0.05, 0.1, 0.5]}

    conditions['falcon'] = {
        'numWolvesLevels': [4],
        'sheepSpeedMultiplierLevels': [1.25],
        'costActionRatioList': [0],
        'rewardSensitivityToDistanceLevels': [0, 10000],
        'biteRewardLevels': [0.05, 0.1, 0.5]}

    conditions['algebra'] = {
        'numWolvesLevels': [4],
        'sheepSpeedMultiplierLevels': [0.75],
        'costActionRatioList': [0.02, 0.025],
        'rewardSensitivityToDistanceLevels': [2, 5],
        'biteRewardLevels': [0.05]}

    conditions['bernoulli'] = {
        'numWolvesLevels': [4],
        'sheepSpeedMultiplierLevels': [1],
        'costActionRatioList': [0.02, 0.025],
        'rewardSensitivityToDistanceLevels': [2, 5],
        'biteRewardLevels': [0.05]}

    conditions['calculus'] = {
        'numWolvesLevels': [4],
        'sheepSpeedMultiplierLevels': [1.25],
        'costActionRatioList': [0.02, 0.025],
        'rewardSensitivityToDistanceLevels': [2, 5],
        'biteRewardLevels': [0.05]}

    conditions['derivative'] = {
        'numWolvesLevels': [4],
        'sheepSpeedMultiplierLevels': [0.75],
        'costActionRatioList': [0.02, 0.025],
        'rewardSensitivityToDistanceLevels': [2, 5],
        'biteRewardLevels': [0.1]}

    conditions['expectation'] = {
        'numWolvesLevels': [4],
        'sheepSpeedMultiplierLevels': [1],
        'costActionRatioList': [0.02, 0.025],
        'rewardSensitivityToDistanceLevels': [2, 5],
        'biteRewardLevels': [0.1]}

    conditions['integration'] = {
        'numWolvesLevels': [4],
        'sheepSpeedMultiplierLevels': [1.25],
        'costActionRatioList': [0.02, 0.025],
        'rewardSensitivityToDistanceLevels': [2, 5],
        'biteRewardLevels': [0.1]}

    conditions['p100b'] = {
        'numWolvesLevels': [4],
        'sheepSpeedMultiplierLevels': [.75, 1],
        'costActionRatioList': [0.02, 0.025],
        'rewardSensitivityToDistanceLevels': [2, 5],
        'biteRewardLevels': [0.5]}

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