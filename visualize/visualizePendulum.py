import os
from environment.gymEnv.pendulumEnv import VisualizeGymPendulum
from functionTools.loadSaveModel import loadFromPickle

def main():
    dirName = os.path.dirname(__file__)
    path = os.path.join(dirName, '..', 'trajectory', 'pendulumTrajectory.pickle')
    traj = loadFromPickle(path)
    trj = traj[:5000]
    visualize = VisualizeGymPendulum()
    visualize(trj)

if __name__ == '__main__':
    main()
