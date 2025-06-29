"""
talos walking simulation
"""

import numpy as np
import pinocchio as pin

import rclpy
from rclpy.node import Node

# simulator
#>>>>TODO: import simulator

# robot configs
#>>>>TODO: Import talos walking config file

# modules
#>>>>TODO: Import all previously coded modules
        
################################################################################
# main
################################################################################  
    
def main(): 
    
    ############################################################################
    # setup
    ############################################################################
    
    #>>>>TODO: setup ros
    
    # setup the simulator
    #>>>>TODO: simulation
    
    # setup the robot
    #>>>> TODO: robot
    
    # inital footsteps
    T_swing_w = #>>>>TODO: set intial swing foot pose to left foot
    T_support_w = #>>>>TODO: set intial support foot pose to right foot                            
    
    # setup the plan with 20 steps
    no_steps = 20
    planner = #>>>>TODO: Create the planner
    plan = #>>>>TODO: Create the plan
    #>>>>TODO: Append the two last steps once more to the plan so our mpc horizon will never run out 

    # generate reference
    ZMP_ref = #>>>>TODO: Generate the mpc reference
    #>>>>TODO: plot the plan (make sure this workes first)
    
    # setup the lip models
    mpc = #>>>>TODO: setup mpc
    
    # Assume the com is over the first support foot
    x0 = #>>>>TODO: Build the intial mpc state vector
    interpolator = #>>>>TODO: Create the interpolator and set the inital state
    
    # set the com task reference to the inital support foot
    c, c_dot, c_ddot = interpolator.comState()
    #>>>>TODO: Set the COM reference to be over supporting foot 
    
    ############################################################################
    # logging
    ############################################################################

    pre_dur = 3.0   # Time to wait befor walking should start
    
    # Compute number of iterations:
    N_pre = #>>>>TODO: number of sim steps before walking starts 
    N_sim = #>>>>TODO: total number of sim steps during walking
    N_mpc = #>>>>TODO: total number of mpc steps during walking
    
    #>>>>TODO: Create vectors to log all the data of the simulation
    # - COM_POS, COM_VEL, COM_ACC (from the planned reference, pinocchio and pybullet)
    # - Angular momentum (from pinocchio)
    # - Left and right foot POS, VEL, ACC (from planned reference, pinocchio) 
    # - ZMP (from planned reference, from estimator )
    # - DCM (from estimtor)
    # - Normal forces in right and left foot (from pybullet ft sensors, from pinocchio)
    TIME = np.nan*np.empty(N_sim)
    
    ############################################################################
    # logging
    ############################################################################
    
    k = 0                                               # current MPC index                          
    plan_idx = 1                                        # current index of the step within foot step plan
    t_step_elapsed = 0.0                                # elapsed time within current step (use to evaluate spline)
    t_publish = 0.0                                     # last publish time (last time we published something)
    
    for i in range(-N_pre, N_sim):
        t = #>>>>TODO: simulator time
        dt = #>>>>TODO: simulator dt
        
        ########################################################################
        # update the mpc very no_sim_per_mpc steps
        ########################################################################
        
        if i >= 0 and #>>>>TODO: when to update mpc
            #>>>>TODO: Implement MPC update
            k += 1        

        ########################################################################
        # update the foot spline 
        ########################################################################

        if i >= 0 and #>>>>TODO: when to update spline
            #>>>>TODO: Start next step
            t_step_elapsed = 0.0
            plan_idx += 1
            
        ########################################################################
        # in every iteration when walking
        ########################################################################
        
        if i >= 0:
            t_step_elapsed += dt

        ########################################################################
        # update the simulation
        ########################################################################

        #>>>>TODO: update the simulator and the robot

        # publish to ros
        if t - t_publish > 1./30.:
            t_publish = t
            #>>>>TODO: publish
            
        # store for visualizations
        if i >= 0:
            TIME[i] = t
            #>>>>TODO: log information
            

    ########################################################################
    # enough with the simulation, lets plot
    ########################################################################
    
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-dark')
    
    #>>>>TODO: plot everything

if __name__ == '__main__': 
    rclpy.init()
    main()
