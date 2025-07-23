import sys
import time
sys.path.insert(0, "..")  # noqa
from simulator.pybullet_wrapper import PybulletWrapper
from walking.go2 import Go2
from walking.controller import Go2Controller

import pinocchio as pin
import numpy as np


def main():
    sim = PybulletWrapper()
    robot = Go2(sim)
    controller = Go2Controller(robot.model)

    x0 = robot.get_state()
    q_d = x0[7:robot.nq]
    v_d = x0[robot.nq+6:robot.nq+robot.nv]

    while True:
        robot.update()
        x0 = robot.get_state()
        q = x0[7:robot.nq]
        v = x0[robot.nq+6:robot.nq+robot.nv]
        u0, x0 = controller.solve(x0)
        # ------------ TEST ------------
        # q = x0[:robot.nq]
        # v = x0[robot.nq:robot.nq + robot.nv]
        # tau = pin.rnea(controller.model, controller.data,
        #               q, v, np.zeros_like(v))
        # tau = tau[6:]
        # robot.set_torque(tau)
        # ------------ TEST ------------
        # robot.set_torque(u0, q_d, q, v_d, v)
        robot.set_torque(u0)
        # robot.set_position(x0[:robot.nq])

        # Step the simulation
        sim.step()
        sim.debug()

        time.sleep(0.01)


if __name__ == '__main__':
    main()
