import numpy as np
import pinocchio as pin
from enum import Enum

class Side(Enum):
    """Side
    Describes which foot to use
    """
    LEFT=0
    RIGHT=1

def other_foot_id(id):
    if id == Side.LEFT:
        return Side.RIGHT
    else:
        return Side.LEFT
        
class FootStep:
    """FootStep
    Holds all information describing a single footstep
    """
    def __init__(self, pose, footprint, side=Side.LEFT):
        """inti FootStep

        Args:
            pose (pin.SE3): the pose of the footstep
            footprint (np.array): 3 by n matrix of foot vertices
            side (_type_, optional): Foot identifier. Defaults to Side.LEFT.
        """
        self.pose = pose
        self.footprint = footprint
        self.side = side
        
    def poseInWorld(self):
        return self.pose
        
    def plot(self, simulation):
        
        #>>>>TODO: plot in pybullet footprint, addGlobalDebugRectancle(...) 
        
        #>>>>TODO: display the side of the step, addUserDebugText(...)
        
        #>>>>TODO: plot step target position addSphereMarker(...)
        
        return None

class FootStepPlanner:
    """FootStepPlanner
    Creates footstep plans (list of right and left steps)
    """
    
    def __init__(self, conf):
        self.conf = conf
        self.steps = []
        
    def planLine(self, T_0_w, side, no_steps):
        """plan a sequence of steps in a strait line

        Args:
            T_0_w (pin.SE3): The inital starting position of the plan
            side (Side): The intial foot for starting the plan
            no_steps (_type_): The number of steps to take

        Returns:
            list: sequence of steps
        """
        
        # the displacement between steps in x and y direction
        dx = self.conf.step_size_x
        dy = 2*self.conf.step_size_y
        
        # the footprint of the robot
        lfxp, lfxn = self.conf.lfxp, self.conf.lfxn
        lfyp, lfyn = self.conf.lfyp, self.conf.lfyn
        
        #>>>>TODO: Plan a sequence of steps with T_0_w being the first step pose.
        #>>>>Note: Plan the second step parallel to the first step (robot starts standing on both feet)
        #>>>>Note: Plan the final step parallel to the last-1 step (robot stops standing on both feet)
        steps=[]
                                
        self.steps = steps
        return steps

    
    def plot(self, simulation):
        for step in self.steps:
            step.plot(simulation)

            
if __name__=='__main__':
    """ Test footstep planner
    """
    
    #>>>>TODO: Generate a plan and plot it in pybullet.
    #>>>>TODO: Check that the plan looks as expected
