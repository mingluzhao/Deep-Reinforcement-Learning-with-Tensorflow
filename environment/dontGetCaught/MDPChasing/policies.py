import numpy as np
import random
import copy 


class RandomPolicy:
    def __init__(self, actionSpace):
        self.actionSpace = actionSpace

    def __call__(self,state):
        
        actionDist = {tuple(action): 1 / len(self.actionSpace) for action in self.actionSpace}
        return actionDist


if __name__ == '__main__':
    main()
