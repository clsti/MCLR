import sys
import time
sys.path.insert(0, "..")  # noqa
sys.path.insert(0, "..")  # noqa
from simulator.pybullet_wrapper import PybulletWrapper
from walking.go2 import Go2
import pybullet as pb
import numpy as np
import pinocchio as pin


def main():
    sim = PybulletWrapper()
    robot = Go2(sim)

    flag = True

    while True:

        # robot.update()
        '''
        if flag:
            FL_pos = np.array(pb.getLinkState(robot.robot.id(), 3)[0])

            FL_pos[2] += 0.1

            flag = False
            q = robot.robot.q()[7:]
            v = robot.robot.v()[6:]

        sim.addSphereMarker(FL_pos)
        p = robot.robot.computeInverseKinematic(3, FL_pos)
        p_goal = np.concatenate([q[:7], p])
        tau = robot.compute_inverse_dynamics(p_goal)
        # robot.set_position(p_goal)
        robot.set_torque(tau)

        data = pin.Data(robot.model)
        tau = pin.rnea(robot.model, data, q,
                       v, np.zeros_like(v))
        tau_actuated = np.zeros(robot.robot.numActuatedJoints())
        tau_actuated[robot.robot._tau_map_pin_pb] = tau
        # robot.set_torque(tau_actuated)
        '''

        # Step the simulation
        sim.step()
        sim.debug()

        time.sleep(0.01)


if __name__ == '__main__':
    main()
