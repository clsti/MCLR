import sys
import time
sys.path.insert(0, "..")  # noqa
sys.path.insert(0, "..")  # noqa
from simulator.pybullet_wrapper import PybulletWrapper
from walking.go2_test import Go2


def main():
    sim = PybulletWrapper()
    robot = Go2(sim)

    for i in range(1000):
        # Step the walking controller
        robot.step_walking_controller()

        robot.robot.update()

        # Step the simulation
        sim.step()
        sim.debug()

        time.sleep(0.01)


if __name__ == '__main__':
    main()
