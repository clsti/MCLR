#!/usr/bin/env py -3
class libpy2():

    '''
    Test library class
    '''

    def __init__(self, a=1, b=2):
        self._a = a
        self._b = b

    def say_it_too(self, node):
        node.get_logger().info("Imported from Library 2 !!")

    def calculate(self, node):
        c = self._a + self._b
        node.get_logger().info("a + b = %f" % (c,))
