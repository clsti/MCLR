import time
import sys
sys.path.insert(0, "..")  # noqa
sys.path.insert(0, "..")  # noqa
from crocoddyl.utils.quadruped import SimpleQuadrupedalGaitProblem
import crocoddyl
import pinocchio as pin
import numpy as np
import walking.conf_go2 as conf
from simulator.robot import Robot
import example_robot_data
package_dirs = [".."]


class Go2():

    def __init__(self, sim):
        self.sim = sim
        self.conf = conf

        self.go2 = example_robot_data.load("go2")
        self.model = self.go2.model
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nu = self.nv - 6

        self.x_0 = self.model.referenceConfigurations["standing"].copy()

        # Spawn robot in simulation
        self.robot = Robot(
            self.sim,
            conf.urdf,
            self.model,
            q=self.x_0,
            basePosition=self.x_0[:3],
            baseQuationerion=self.x_0[3:7],
            useFixedBase=False
        )

        self.solve_one_step()

    def solve_one_step(self):
        lfFoot, rfFoot, lhFoot, rhFoot = "FL_foot", "FR_foot", "RL_foot", "RR_foot"
        self.problem = SimpleQuadrupedalGaitProblem(
            self.model, lfFoot, rfFoot, lhFoot, rhFoot)

        # Walking gait parameters
        self.walking_gait = {
            "stepLength": 0.25,
            "stepHeight": 0.15,
            "timeStep": 1e-2,
            "stepKnots": 100,
            "supportKnots": 2,
        }

        # Initialize with standing configuration
        q0 = self.model.referenceConfigurations["standing"].copy()
        v0 = pin.utils.zero(self.model.nv)

        self.current_state = np.concatenate([q0, v0])

        self.solver = crocoddyl.SolverBoxDDP(
            self.problem.createWalkingProblem(
                self.current_state,
                self.walking_gait["stepLength"],
                self.walking_gait["stepHeight"],
                self.walking_gait["timeStep"],
                self.walking_gait["stepKnots"],
                self.walking_gait["supportKnots"],
            )
        )

        # Initial guess
        xs = [self.current_state] * (self.solver.problem.T + 1)
        us = self.solver.problem.quasiStatic(
            [self.current_state] * self.solver.problem.T)

        # Solve
        self.solver.solve(xs, us, 100, False, 0.1)

        xs = self.solver.xs
        xs_array = [np.array(x) for x in xs]

        n_one_step = int(len(xs_array)/1)
        xs_array_trunc = xs_array[:int(n_one_step)]

        # Visualize
        joints_to_print = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']

        model_visu = self.model.copy()
        data_visu = model_visu.createData()

        for q in xs_array_trunc:
            pin.forwardKinematics(model_visu, data_visu, q[:self.nq])
            pin.updateFramePlacements(model_visu, data_visu)
            for name in joints_to_print:
                frame_id = self.model.getFrameId(name)
                pos = data_visu.oMf[frame_id].translation
                # self.sim.addSphereMarker(pos, radius=0.01)
            self.set_position(q[:self.nq])
            self.sim.step()
            time.sleep(0.01)

    def update(self):
        self.robot.update()

    def get_state(self, nqx=False):
        # Get base position and orientation
        base_pos = self.robot.baseWorldPosition()
        base_orn = self.robot.baseWorldOrientation()  # quaternion [x,y,z,w]
        if nqx:
            R = pin.Quaternion(base_orn).toRotationMatrix()
            base_orn = pin.log3(R)

        # Get joint positions and velocities
        joint_pos = self.robot.actuatedJointPosition()
        joint_vel = self.robot.actuatedJointVelocity()

        # Get base velocity
        base_vel_linear = self.robot.baseWorldLinearVeloctiy()
        base_vel_angular = self.robot.baseWorldAngulaVeloctiy()

        # Construct state vector [q, v] for crocoddyl
        q = np.concatenate([base_pos, base_orn, joint_pos])
        v = np.concatenate([base_vel_linear, base_vel_angular, joint_vel])

        return np.concatenate([q, v])

    def compute_inverse_dynamics(self, q_d):
        zero_vel = np.zeros(self.nv)
        zero_acc = np.zeros(self.nv)
        return pin.rnea(self.model, self.go2.data, q_d, zero_vel, zero_acc)[6:]

    def set_torque(self, tau):
        self.robot.setActuatedJointTorques(tau)

    def set_position(self, pos):
        self.robot.setActuatedJointPositions(pos)

    def publish(self):
        joint_positions = self.robot.actuatedJointPosition()
        joint_velocities = self.robot.actuatedJointVelocity()
        print("Joint positions:", joint_positions)
        print("Joint velocities:", joint_velocities)
