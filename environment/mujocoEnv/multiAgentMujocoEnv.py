import numpy as np
import math
import mujoco_py as mujoco


class IsOverlap:
    def __init__(self,minDistance):
        self.minDistance=minDistance
    def __call__(self,blocksState,proposalState):
        for blockState in blocksState:
            distance=np.linalg.norm(np.array(proposalState[:2])-np.array(blockState[:2]))
            if distance< self.minDistance:
                return True
        return False

class SampleBlockState:
    def __init__(self,numBlocks,getBlockPos,getBlockSpeed,isOverlap):
        self.numBlocks=numBlocks
        self.getBlockPos=getBlockPos
        self.getBlockSpeed=getBlockSpeed
        self.isOverlap=isOverlap

    def __call__(self):
        blocksState=[]
        for blockID in range(self.numBlocks):
            proposalState=list(self.getBlockPos())+list(self.getBlockSpeed())
            while self.isOverlap(blocksState,proposalState):
                proposalState=list(self.getBlockPos())+list(self.getBlockSpeed())
            blocksState.append(proposalState)

        return blocksState

class ResetUniformWithoutXPos:
    def __init__(self, simulation, numAgent,numBlock, sampleAgentsQPos,sampleAgentsQVel, sampleBlockState):
        self.simulation = simulation
        self.numAgent = self.simulation.model.nsite
        self.numBlock = numBlock
        self.numJointEachSite = int(self.simulation.model.njnt / self.simulation.model.nsite)
        self.sampleAgentsQPos = sampleAgentsQPos
        self.sampleAgentsQVel = sampleAgentsQVel
        self.sampleBlockState = sampleBlockState

    def __call__(self):
        numQPos = len(self.simulation.data.qpos)
        numQVel = len(self.simulation.data.qvel)

        qPos = self.sampleAgentsQPos()
        qVel = self.sampleAgentsQVel()

        blocksState = self.sampleBlockState()

        for block in range(self.numBlock):#change blocks pos in mujoco simulation,see xml for more details,[floor+agents+blocks] [x,y,z]
            self.simulation.model.body_pos[2+self.numAgent+block][:2]=blocksState[block][:2]
        self.simulation.data.qpos[:] = qPos
        self.simulation.data.qvel[:] = qVel
        self.simulation.forward()

        agentQPos = lambda agentIndex: qPos[self.numJointEachSite * agentIndex: self.numJointEachSite * (agentIndex + 1)]
        agentQVel = lambda agentIndex: qVel[self.numJointEachSite * agentIndex: self.numJointEachSite * (agentIndex + 1)]
        getAgentState = lambda agentIndex: np.concatenate([agentQPos(agentIndex), agentQVel(agentIndex)])
        agentState = [getAgentState(agentIndex) for agentIndex in range(self.numAgent)]
        startState=np.asarray(agentState+blocksState)

        return startState



class TransitionFunction:
    def __init__(self, simulation, numAgents, numSimulationFrames,visualize,isTerminal,reshapeAction):
        self.simulation = simulation
        self.numAgents = numAgents
        self.numSimulationFrames = numSimulationFrames
        self.numJointEachSite = int(self.simulation.model.njnt/self.simulation.model.nsite)
        self.visualize=visualize
        self.isTerminal = isTerminal
        self.reshapeAction=reshapeAction
        if visualize:
            self.physicsViewer = mujoco.MjViewer(simulation)
    def __call__(self, state, actions):
        actions = [self.reshapeAction(action) for action in actions]
        state = np.asarray(state)
        actions = np.asarray(actions)
        oldQPos =np.array([QPos for agent in state[:self.numAgents] for QPos in agent[:self.numJointEachSite]]).flatten()
        oldQVel =np.array([QVel for agent in state[:self.numAgents] for QVel in agent[-self.numJointEachSite:]]).flatten()
        blocksState=[np.asarray(block) for block in state[self.numAgents:]]

        self.simulation.data.qpos[:] = oldQPos
        self.simulation.data.qvel[:] = oldQVel
        self.simulation.data.ctrl[:] = actions.flatten()

        agentNewQPos = lambda agentIndex: newQPos[self.numJointEachSite * agentIndex: self.numJointEachSite * (agentIndex + 1)]
        agentNewQVel = lambda agentIndex: newQVel[self.numJointEachSite * agentIndex: self.numJointEachSite * (agentIndex + 1)]
        getSpeed=lambda Vel : np.linalg.norm(Vel)
        agentNewState = lambda agentIndex: np.concatenate([agentNewQPos(agentIndex), agentNewQVel(agentIndex)])

        for simulationFrame in range(self.numSimulationFrames):
            self.simulation.step()
            self.simulation.forward()
            if self.visualize:
                self.physicsViewer.render()
            newQPos, newQVel = self.simulation.data.qpos, self.simulation.data.qvel

            newState = [agentNewState(agentIndex) for agentIndex in range(self.numAgents)]
            if self.isTerminal(newState):
                break

        newQPos, newQVel = self.simulation.data.qpos, self.simulation.data.qvel
        newState = [agentNewState(agentIndex) for agentIndex in range(self.numAgents)]
        newState = np.asarray(newState+blocksState)
        return newState


