import pybullet as pb
import numpy as np
from simulator.pybullet_wrapper import PybulletWrapper
from simulator.robot import Robot
import pinocchio as pin

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState


class RobotSimulator(Node):
    def __init__(self):
        super().__init__('robot_simulator')

        # For REEM-C robot
        # urdf = "src/exercise_01/reemc_description/robots/reemc.urdf"
        # path_meshes = "src/exercise_01/reemc_description/meshes/../.."

        # For Talos robot
        urdf = "src/exercise_01/talos_description/robots/talos_reduced.urdf"
        path_meshes = "src/exercise_01/talos_description/meshes/../.."

        '''
        Talos
        0, 1, 2, 3, 4, 5, 			    # left leg
        6, 7, 8, 9, 10, 11, 			# right leg
        12, 13,                         # torso
        14, 15, 16, 17, 18, 19, 20, 21  # left arm
        22, 23, 24, 25, 26, 27, 28, 29  # right arm
        30, 31                          # head

        REEMC
        0, 1, 2, 3, 4, 5, 			    # left leg
        6, 7, 8, 9, 10, 11, 			# right leg
        12, 13,                         # torso
        14, 15, 16, 17, 18, 19, 20,     # left arm
        21, 22, 23, 24, 25, 26, 27,     # right arm
        28, 29                          # head
        '''

        # Initial condition for the simulator an model
        z_init = 1.15
        q_actuated_home = np.zeros(32)
        q_actuated_home[:6] = np.array([0, 0, 0, 0, 0, 0])
        q_actuated_home[6:12] = np.array([0, 0, 0, 0, 0, 0])
        q_actuated_home[14:22] = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        q_actuated_home[22:30] = np.array([0, 0, 0, 0, 0, 0, 0, 0])

        # Initialization position including floating base
        q_home = np.hstack(
            [np.array([0, 0, z_init, 0, 0, 0, 1]), q_actuated_home])

        # setup the task stack
        modelWrap = pin.RobotWrapper.BuildFromURDF(urdf,                        # Model description
                                                   path_meshes,                 # Model geometry descriptors
                                                   pin.JointModelFreeFlyer(),   # Floating base model. Use "None" if fixed
                                                   True,                        # Printout model details
                                                   None)                        # Load meshes different from the descripor
        # Get model from wrapper
        model = modelWrap.model

        # setup the simulator
        self.simulator = PybulletWrapper(sim_rate=1000)

        # Create Pybullet-Pinocchio map
        self.robot = Robot(self.simulator,       # The Pybullet wrapper
                           urdf,                 # Robot descriptor
                           model,                # Pinocchio model
                           [0, 0, z_init],       # Floating base initial position
                           # Floating base initial orientation [x,y,z,w]
                           [0, 0, 0, 1],
                           q=q_home,             # Initial state
                           useFixedBase=False,   # Fixed base or not
                           verbose=True)         # Printout details

        # get data
        data = self.robot._model.createData()

        # Compute inertia matrix
        M = pin.crba(self.robot._model, data, self.robot.q())
        # Compute non-linear effects
        h = pin.nonLinearEffects(
            self.robot._model, data, self.robot.q(), self.robot.v())

        # print complete matrix
        np.set_printoptions(
            threshold=np.inf,
            linewidth=200,
            suppress=True,
            formatter={'float_kind': lambda x: f"{x:.1f}"}
        )

        # Printout Inertia matrix & Non linear effects
        print("Inertia matrix:")
        print(M)
        print("Non linear effects:")
        print(h)

        # Needed for compatibility
        self.simulator.addLinkDebugFrame(-1, -1)

        # Setup pybullet camera
        pb.resetDebugVisualizerCamera(
            cameraDistance=1.2,
            cameraYaw=90,
            cameraPitch=-20,
            cameraTargetPosition=[0.0, 0.0, 0.8])

        # Joint command vector
        self.tau = q_actuated_home*0

        # gain matrices
        Kx_I = np.diag(np.array([3]*12 + [1]*20))
        self.K_p = 400.0 * Kx_I  # stable: 400 / falling: 300
        self.K_d = 0.4 * Kx_I

        # desired joint state
        self.q_d = np.zeros_like(q_actuated_home)

        # spline parameters
        time = 1
        self.t_step_size = 0.001 / time
        self.t_step = 0.0

        q_floating_base = np.array([0, 0, z_init, 0, 0, 0, 1])
        q_home_act = q_actuated_home
        q_home_act[:6] = np.array([0, 0, -0.44, 0.9, -0.45, 0])  # left leg
        q_home_act[6:12] = np.array([0, 0, -0.44, 0.9, -0.45, 0])  # right leg
        q_home_act[14:22] = np.array([0, -0.24, 0, -1, 0, 0, 0, 0])  # left arm
        q_home_act[22:30] = np.array(
            [0, -0.24, 0, -1, 0, 0, 0, 0])  # right arm

        self.q_init = q_home
        self.q_home_pos = np.hstack([q_floating_base, q_home_act])

        # initialize publisher
        self.pub = self.create_publisher(JointState, "/joint_states", 10)

        # initialize timer (1kHz)
        self.timer = self.create_timer(0.001, self.sim_run)
        # joint state publisher step parameter
        self.t_30Hz = 0.0

    def spline_init_home(self, q_0, q_1, step):
        # spline from q_init to q_home
        if step >= 1.0:
            step = 1.0
        elif step <= 0.0:
            step = 0.0
        q = pin.interpolate(self.robot._model, q_0, q_1, step)
        return q[7:]

    def pub_joint_state(self, names, q, q_dot, tau):
        # publish the joint state
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = names
        joint_msg.position = q.tolist()
        joint_msg.velocity = q_dot.tolist()
        joint_msg.effort = tau.tolist()

        self.pub.publish(joint_msg)

    def sim_pub(self):
        self.pub_joint_state(self.robot.actuatedJointNames(),
                             self.robot.actuatedJointPosition(),
                             self.robot.actuatedJointVelocity(),
                             self.tau)

    def sim_run(self):
        # update the simulator and the robot
        self.simulator.step()
        self.simulator.debug()
        self.robot.update()

        # mask floating base
        q = self.robot.q()[7:]
        v = self.robot.v()[6:]

        q_d = self.spline_init_home(self.q_init, self.q_home_pos, self.t_step)
        self.t_step += self.t_step_size

        # PD controller
        self.tau = self.K_p @ (q_d - q) - self.K_d @ v

        # command to the robot
        self.robot.setActuatedJointTorques(self.tau)

        self.t_30Hz += 0.001
        if self.t_30Hz >= 1.0/30.0:
            self.sim_pub()
            self.t_30Hz = 0.0


def main(args=None):
    rclpy.init(args=args)
    node = RobotSimulator()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
