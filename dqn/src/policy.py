import os
import numpy as np 
import random 
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class ActGreedyByModel:
    def __init__(self, getTrainQValue, model):
        self.getTrainQValue = getTrainQValue
        self.model = model
        
    def __call__(self, states):
        stateBatch = np.asarray(states).reshape(1, -1)
        trainQVal = self.getTrainQValue(self.model, stateBatch)
        action = np.argmax(trainQVal)
        
        return action 

class ActRandom:
    def __init__(self, actionDim):
        self.actionDim = actionDim
        
    def __call__(self):
        action = random.randrange(self.actionDim)
        return action


class GetEpsilon:
    def __init__(self, epsilonMax, epsilonMin, epsilonIncrease, decayStartStep):
        self.epsilonMax = epsilonMax
        self.epsilonMin = epsilonMin
        self.epsilonIncrease = epsilonIncrease
        self.decayStartStep = decayStartStep
        
    def __call__(self, runTime):
        epsilon = self.epsilonMin
        if runTime > self.decayStartStep:
            epsilonResult = self.epsilonMin + self.epsilonIncrease * (runTime- self.decayStartStep)
            epsilon = np.clip(epsilonResult, self.epsilonMin, self.epsilonMax)
        
        if runTime % 2000 == 0:
            print('epsilon: ', epsilon)
            
        return epsilon


class ActByTrainNetEpsilonGreedy:
    def __init__(self, getEpsilon, actGreedyByModel, actRandom):
        self.getEpsilon = getEpsilon
        self.actGreedyByModel = actGreedyByModel
        self.actRandom = actRandom
        
    def __call__(self, states, runTime):
        epsilon = self.getEpsilon(runTime)
        action = self.actGreedyByModel(states) if np.random.uniform() < epsilon else self.actRandom()

        return [action]
