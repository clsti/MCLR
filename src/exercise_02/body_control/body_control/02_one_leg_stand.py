import numpy as np
from numpy import nan
from numpy.linalg import norm as norm
import matplotlib.pyplot as plt

# pinocchio
import pinocchio as pin

# simulator
import pybullet as pb
from simulator.pybullet_wrapper import PybulletWrapper
from simulator.robot import Robot

# robot and controller
from tutorial_4.tsid_wrapper import TSIDWrapper
import tutorial_4.config as conf

# ROS
import rclpy
from rclpy.node import Node
import tf2_ros
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped

################################################################################
# settings
################################################################################

DO_PLOT = True

################################################################################
# Robot
################################################################################

class Talos(Robot):
    def __init__(self, simulator, urdf, model, q=None, verbose=True, useFixedBase=True):
        # TODO call base class constructor
        # TODO add publisher
        # TODO add tf broadcaster
        pass

    def update(self):
        # TODO update base class
        pass
    
    def publish(self, T_b_w):
        # TODO publish jointstate
        # TODO broadcast transformation T_b_w
        pass

################################################################################
# main
################################################################################

def main(): 
    node = rclpy.create_node('tutorial_4_one_leg_stand_node')
    
    # TODO init TSIDWrapper
    # TODO init Simulator
    # TODO init ROBOT
    
    t_publish = 0.0

    while rclpy.ok():

        # elaped time
        t = simulator.simTime()

        # TODO: update the simulator and the robot
        
        # TODO: update TSID controller

        # command to the robot
        robot.setActuatedJointTorques(tau_sol)

        # publish to ros
        if t - t_publish > 1./30.:
            t_publish = t
            # get current BASE Pose
            robot.publish(T_b_w)
    
if __name__ == '__main__': 
    rclpy.init()
    main()
