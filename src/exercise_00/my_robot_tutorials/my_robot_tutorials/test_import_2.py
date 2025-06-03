#!/usr/bin/env python

import rclpy

from my_python_libs.python_lib_1 import say_it_works
from my_python_libs.python_lib_2 import libpy2

from my_other_python_libs.other_python_lib_1 import say_it_again
from my_other_python_libs.other_python_lib_2 import libpy3


def main(args=None):
    rclpy.init()
    node = rclpy.create_node('test_node')
    say_it_works()
    lib2instance = libpy2(2, 3)
    lib2instance.say_it_too(node)
    lib2instance.calculate(node)

    say_it_again()
    lib3instance = libpy3(2, 3)
    lib3instance.say_it_too(node)
    lib3instance.calculate(node)

    print("End of script")


if __name__ == '__main__':
    main()
