import json

def main():
    conditions = dict()

    conditions['Lululucyzs-MBP.lan'] = {
        'numWolvesLevels': [4],
        'sheepSpeedMultiplierLevels': [0.75],
        'costActionRatioList': [0.02, 0.025],
        'rewardSensitivityToDistanceLevels': [0],
        'biteRewardLevels': [0.05]}

    conditions['vi385064core2-PowerEdge-R7515'] = {
        'numWolvesLevels': [2, 4],
        'sheepSpeedMultiplierLevels': [0.75],
        'costActionRatioList': [0, 0.02],
        'rewardSensitivityToDistanceLevels': [0, 10000],
        'biteRewardLevels': [0],
        'fileID': list(range(10))}

    outputFile = open('conditionsMADDPGWithProbKill_testEqui.json', 'w')
    json.dump(conditions, outputFile)
    outputFile.close()
    print(conditions.keys())


if __name__ == '__main__':
    main()