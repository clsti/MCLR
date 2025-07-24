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
    # com reference height
    com_h = robot.robot.baseCoMPosition()[2]
    controller = Go2Controller(robot.model, com_h)

    x0 = robot.get_state()
    q_d = x0[7:robot.nq]
    v_d = x0[robot.nq+6:robot.nq+robot.nv]

    while True:
        robot.update()
        x0 = robot.get_state()
        q = x0[7:robot.nq]
        v = x0[robot.nq+6:robot.nq+robot.nv]
        u0, xs0 = controller.solve(x0, controller.standing_problem)
        # ------------ TEST ------------
        # q = x0[:robot.nq]
        # v = x0[robot.nq:robot.nq + robot.nv]
        # tau = pin.rnea(controller.model, controller.data,
        #               q, v, np.zeros_like(v))
        # tau = tau[6:]
        # robot.set_torque(tau)
        # ------------ TEST ------------
        
        try:
            k, K = controller.get_feedback_gains(node_index=0)
            
            state_error = xs0 - x0
            alpha = 1.0 
            
            if state_error.shape[0] == 37 and K.shape[1] == 36:
                # Convert state error to tangent space representation
                q_current = x0[:robot.nq]  
                q_ref = xs0[:robot.nq]     
                v_current = x0[robot.nq:]  
                v_ref = xs0[robot.nq:]     
                
                # configuration error in tangent space
                q_error_tangent = pin.difference(controller.model, q_ref, q_current)
                v_error = v_current - v_ref
                state_error_tangent = np.concatenate([q_error_tangent, v_error])   
                
                             
                uk = u0 + alpha * k + K @ state_error_tangent
                #print(f"uo: {u0}\nuk: {uk}")
            else:
                uk = u0 + alpha * k + K @ state_error

        except (AttributeError, RuntimeError) as e:
            print(f"Warning: Could not get feedback gains: {e}")
+            uk = u0
        
        #robot.set_torque(uk, q_d, q, v_d, v)
        #robot.set_torque(u0)
        #robot.set_position(x0[:robot.nq])

        # Step the simulation
        sim.step()
        sim.debug()

        time.sleep(0.01)


if __name__ == '__main__':
    main()
