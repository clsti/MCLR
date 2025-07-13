import rclpy
from simulator.pybullet_wrapper import PybulletWrapper

from walking.go2 import Go2


def main(args=None):

    rclpy.init(args=args)

    sim = PybulletWrapper()

    robot = Go2(sim)

    while rclpy.ok():
        sim.step()
        sim.debug()


if __name__ == '__main__':
    main()
