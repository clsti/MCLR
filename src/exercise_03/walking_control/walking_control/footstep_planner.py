import numpy as np
import pinocchio as pin
from enum import Enum


class Side(Enum):
    """Side
    Describes which foot to use
    """
    LEFT = 0
    RIGHT = 1


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
        # Plot in pybullet footprint, addGlobalDebugRectancle(...)
        # Compute rectangle size
        front = np.max(self.footprint[0, :])
        rear = -np.min(self.footprint[0, :])
        left = np.max(self.footprint[1, :])
        right = -np.min(self.footprint[1, :])

        length = front + rear
        width = right + left

        # Compute center of footprint in foot frame
        center_offset = np.mean(self.footprint, axis=1)

        # Transform to world frame
        R = self.pose.rotation
        # center_p = (self.pose.translation + R @ center_offset).tolist()
        center_p = (self.pose.translation + center_offset).tolist()
        quat = pin.Quaternion(R).coeffs()
        quat_xyzw = [quat[0], quat[1], quat[2], quat[3]]

        simulation.addGlobalDebugRectancle(
            center_p, Q=quat_xyzw, length=length, width=width)

        # Display the side of the step, addUserDebugText(...)
        text = "Left" if self.side == Side.LEFT else "Right"
        simulation.addUserDebugText(-1, -1, text,
                                    self.pose.translation + np.array([0.0, 0.0, 0.1]))

        # Plot step target position addSphereMarker(...)
        simulation.addSphereMarker(self.pose.translation)

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

        # Plan a sequence of steps with T_0_w being the first step pose.
        # Note: Plan the second step parallel to the first step (robot starts standing on both feet)
        # Note: Plan the final step parallel to the last-1 step (robot stops standing on both feet)
        steps = []

        # Define footprint depending on side
        footprint_l = np.array([[lfxp, lfxp, -lfxn, -lfxn],
                                [lfyp, -lfyn, -lfyn, lfyp],
                                [0, 0, 0, 0]])
        footprint_r = np.array([[lfxp, lfxp, -lfxn, -lfxn],
                                [-lfyn, lfyp, lfyp, -lfyn],
                                [0, 0, 0, 0]])

        curr_side = side
        curr_pose = T_0_w

        # Get rotation of first foot
        R = curr_pose.rotation

        for i in range(no_steps):
            # Distinguish footprint
            footprint = footprint_l if curr_side == Side.LEFT else footprint_r

            # Add footstep
            step = FootStep(curr_pose, footprint, curr_side)
            steps.append(step)

            # Alternate foot side
            curr_side = other_foot_id(curr_side)

            # Get displacement
            dy_pos = dy if curr_side == Side.LEFT else -dy

            if i == 0 or i == no_steps-2:
                # No forward step for first or last step
                t = np.array([0.0, dy_pos, 0.0])
                # translation = R @ t
                translation = t
            else:
                # Compute translation for step
                t = np.array([dx, dy_pos, 0.0])
                # translation = R @ t
                translation = t

            # Create new foot pose
            new_translation = curr_pose.translation + translation
            curr_pose = pin.SE3(R, new_translation)

        self.steps = steps
        return steps

    def plot(self, simulation):
        for step in self.steps:
            step.plot(simulation)


if __name__ == '__main__':
    """ Test footstep planner
    """
    import sys
    import os
    cwd = os.getcwd()
    PYBW_path = os.path.join(cwd, "src/exercise_01/simulator/simulator")
    sys.path.insert(0, PYBW_path)
    from pybullet_wrapper import PybulletWrapper

    # Generate a plan and plot it in pybullet.
    # Check that the plan looks as expected

    class conf:
        # step dimensions
        step_size_x = 0.25              # step size in x direction
        step_size_y = 0.096             # step size in y direction

        foot_scaling = 1.
        lfxp = foot_scaling*0.08                # foot length in positive x direction
        lfxn = foot_scaling*0.12                # foot length in negative x direction
        lfyp = foot_scaling*0.065               # foot length in positive y direction
        lfyn = foot_scaling*0.065               # foot length in negative y direction

    planner = FootStepPlanner(conf)

    random = False
    if random:
        # Random first footstep initialization
        yaw = np.random.uniform(-np.pi, np.pi)
        R = pin.utils.rpyToMatrix(np.array([0.0, 0.0, yaw]))
        t = np.random.uniform(low=[-1.0, -1.0, 0.0], high=[1.0, 1.0, 0.0])
        start_pose = pin.SE3(R, t)
    else:
        # Identitiy first footstep initialization
        start_pose = pin.SE3.Identity()

    steps = planner.planLine(start_pose, Side.LEFT, 10)

    sim = PybulletWrapper()
    planner.plot(sim)

    while True:
        sim.step()
