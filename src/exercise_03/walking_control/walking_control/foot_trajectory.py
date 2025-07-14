import numpy as np
import pinocchio as pin

import numpy as np
import matplotlib.pyplot as plt


class SexticSpline3D:
    def __init__(self, t0, t1, t2, p0, p1, p2, v0, v2, a0, a2):
        """
        Create a single 6th degree polynomial spline over [t0, t2]
        Enforces:
        - position at t0, t1, t2
        - velocity at t0, t2
        - acceleration at t0, t2

        Args:
            t0, t1, t2: time points (t0 < t1 < t2)
            p0, p1, p2: positions at those times (3D vectors)
            v0, v2: velocities at t0 and t2 (3D vectors)
            a0, a2: accelerations at t0 and t2 (3D vectors)
        """
        self.t0 = t0
        self.t1 = t1
        self.t2 = t2

        # Matrix A for constraints (7 constraints)
        # Polynomial basis: 1, t, t^2, ..., t^6
        def row(t, deriv_order=0):
            if deriv_order == 0:  # position
                return np.array([t**i for i in range(7)])
            elif deriv_order == 1:  # velocity (first derivative)
                return np.array([0] + [i * t**(i-1) for i in range(1, 7)])
            elif deriv_order == 2:  # acceleration (second derivative)
                return np.array([0, 0] + [i*(i-1)*t**(i-2) for i in range(2, 7)])
            else:
                raise ValueError("Derivative order must be 0,1,2")

        A = np.vstack([
            row(t0, 0),  # pos t0
            row(t1, 0),  # pos t1
            row(t2, 0),  # pos t2
            row(t0, 1),  # vel t0
            row(t2, 1),  # vel t2
            row(t0, 2),  # acc t0
            row(t2, 2),  # acc t2
        ])

        # For each dimension solve linear system
        self.coeffs = np.zeros((3, 7))  # 3 dims, 7 coeffs each

        for dim in range(3):
            b = np.array([
                p0[dim],
                p1[dim],
                p2[dim],
                v0[dim],
                v2[dim],
                a0[dim],
                a2[dim],
            ])
            self.coeffs[dim] = np.linalg.solve(A, b)

    def _eval_poly(self, coeffs, t):
        powers = np.array([t**i for i in range(7)])
        return coeffs @ powers

    def _eval_poly_deriv(self, coeffs, t, order):
        if order == 1:
            powers = np.array([0] + [i * t**(i-1) for i in range(1, 7)])
        elif order == 2:
            powers = np.array([0, 0] + [i*(i-1)*t**(i-2) for i in range(2, 7)])
        else:
            raise ValueError("Supports only 1st and 2nd derivatives")
        return coeffs @ powers

    def pos(self, t):
        return np.array([self._eval_poly(self.coeffs[dim], t) for dim in range(3)])

    def vel(self, t):
        return np.array([self._eval_poly_deriv(self.coeffs[dim], t, 1) for dim in range(3)])

    def acc(self, t):
        return np.array([self._eval_poly_deriv(self.coeffs[dim], t, 2) for dim in range(3)])


class CubicSpline3D:
    """
    • x, y : single cubic from p0 → p2 over [t0, t2]  (ẋ = 0 at both ends)
    • z    : two cubics
             – segment‑1  p0.z → p1.z  over [t0, t1]
             – segment‑2  p1.z → p2.z  over [t1, t2]
             (ẋ = 0 at t0, t1, t2)
    """

    def __init__(self, t0, t1, t2, p0, p1, p2, *unused):
        self.t0, self.t1, self.t2 = float(t0), float(t1), float(t2)
        self.T = self.t2 - self.t0
        self.T_seg1 = self.t1 - self.t0
        self.T_seg2 = self.t2 - self.t1

        # --- helper: cubic with zero start/end velocity -----------------
        def coeffs(x0, x1, T):
            a = x0
            b = np.zeros_like(x0)
            c = 3.0 * (x1 - x0) / T**2
            d = 2.0 * (x0 - x1) / T**3     # note sign
            return np.vstack([a, b, c, d]).T   # (n,4)

        # x,y coefficients (shared for both segments)
        self.coeffs_xy = coeffs(p0[:2], p2[:2], self.T)   # (2,4)

        # z coefficients
        self.coeffs_z1 = coeffs(np.array([p0[2]]), np.array(
            [p1[2]]), self.T_seg1)  # (1,4)
        self.coeffs_z2 = coeffs(np.array([p1[2]]), np.array(
            [p2[2]]), self.T_seg2)  # (1,4)

    # ------------------------------------------------------------------
    @staticmethod
    def _poly(c, τ): return c @ np.array([1, τ, τ**2,  τ**3])
    @staticmethod
    def _dpoly(c, τ): return c @ np.array([0, 1, 2*τ,  3*τ**2])
    @staticmethod
    def _ddpoly(c, τ): return c @ np.array([0, 0, 2,    6*τ])

    # ------------------------------------------------------------------
    def _z_coeff_tau(self, t):
        """Return (coeff_row, τ_local) for the z axis."""
        if t < self.t1:
            τ = np.clip(t - self.t0, 0.0, self.T_seg1)
            return self.coeffs_z1[0], τ
        τ = np.clip(t - self.t1, 0.0, self.T_seg2)
        return self.coeffs_z2[0], τ

    # ------------------------------------------------------------------
    def pos(self, t):
        τ_xy = np.clip(t - self.t0, 0.0, self.T)        # full‑range τ
        x = self._poly(self.coeffs_xy[0], τ_xy)
        y = self._poly(self.coeffs_xy[1], τ_xy)

        c_z, τz = self._z_coeff_tau(t)
        z = self._poly(c_z, τz)
        return np.array([x, y, z])

    def vel(self, t):
        τ_xy = np.clip(t - self.t0, 0.0, self.T)
        vx = self._dpoly(self.coeffs_xy[0], τ_xy)
        vy = self._dpoly(self.coeffs_xy[1], τ_xy)

        c_z, τz = self._z_coeff_tau(t)
        vz = self._dpoly(c_z, τz)
        return np.array([vx, vy, vz])

    def acc(self, t):
        τ_xy = np.clip(t - self.t0, 0.0, self.T)
        ax = self._ddpoly(self.coeffs_xy[0], τ_xy)
        ay = self._ddpoly(self.coeffs_xy[1], τ_xy)

        c_z, τz = self._z_coeff_tau(t)
        az = self._ddpoly(c_z, τz)
        return np.array([ax, ay, az])


class SwingFootTrajectory:
    """SwingFootTrajectory
    Interpolate Foot trajectory between SE3 T0 and T1
    """

    def __init__(self, T0, T1, duration, height=0.05):
        """initialize SwingFootTrajectory

        Args:
            T0(pin.SE3): Inital foot pose
            T1(pin.SE3): Final foot pose
            duration(float): step duration
            height(float, optional): setp height. Defaults to 0.05.
        """
        self._height = height
        self._t_elapsed = 0.0
        self._duration = duration
        self._t_end = self._t_elapsed + duration
        self.reset(T0, T1)

    def reset(self, T0, T1):
        '''reset back to zero, update poses
        '''
        # plan the spline
        self.T0 = T0
        self.T1 = T1

        p0 = T0.translation
        p1 = T1.translation

        # Midpoint with weight added
        pmid = 0.5 * (p0 + p1)
        pmid[2] += self._height

        tmid = self._t_elapsed + 0.5 * self._duration
        zeros = np.zeros(3)

        # TODO Cubic spline to prevent high velocity that result in infeasible qp solver!!!!
        Sextic = False
        if Sextic:
            self.curve = SexticSpline3D(
                self._t_elapsed, tmid, self._t_end,
                p0, pmid, p1,
                zeros, zeros, zeros, zeros
            )
        else:
            self.curve = CubicSpline3D(
                self._t_elapsed, tmid, self._t_end,
                p0, pmid, p1,
                zeros, zeros, zeros, zeros
            )

        # Interpolate rotations linearly in SO3
        # self.R0 = T0.rotation
        # self.R1 = T1.rotation

    def isDone(self):
        return self._t_elapsed >= self._duration

    def evaluate(self, t):
        """evaluate at time t
        """
        # evaluate the spline at time t, return pose, velocity, acceleration
        t = np.clip(t, self._t_elapsed, self._t_end)

        pos = self.curve.pos(t)
        vel = self.curve.vel(t)
        acc = self.curve.acc(t)

        # alpha = t / self._duration
        # R = self.R0 @ pin.exp3(pin.log3(self.R0.T @ self.R1) * alpha)

        R = np.eye(3)

        pose = pin.SE3(R, pos)
        return pose, vel, acc


if __name__ == "__main__":
    T0 = pin.SE3(np.eye(3), np.array([0, 0, 0]))
    T1 = pin.SE3(np.eye(3), np.array([0.2, 0, 0]))

    # plot to make sure everything is correct
    duration = 5.0
    traj = SwingFootTrajectory(T0, T1, duration, height=0.05)

    N = 100
    times = np.linspace(0, duration, N)

    # Pre-allocate arrays
    positions = np.zeros((N, 3))
    velocities = np.zeros((N, 3))
    accelerations = np.zeros((N, 3))

    # Fill arrays
    for i, t in enumerate(times):
        pose, vel, acc = traj.evaluate(t)
        positions[i] = pose.translation
        velocities[i] = vel
        accelerations[i] = acc

    # Unpack for plotting
    xs, ys, zs = positions[:, 0], positions[:, 1], positions[:, 2]
    vxs, vys, vzs = velocities[:, 0], velocities[:, 1], velocities[:, 2]
    axs, ays, azs = accelerations[:,
                                  0], accelerations[:, 1], accelerations[:, 2]

    # Plotting
    fig = plt.figure(figsize=(15, 5))

    ax = fig.add_subplot(131, projection='3d')
    ax.plot(xs, ys, zs, label='Position')
    ax.scatter([T0.translation[0], T1.translation[0]],
               [T0.translation[1], T1.translation[1]],
               [T0.translation[2], T1.translation[2]],
               c='r', label='Start/End')
    ax.set_title('Foot Trajectory')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    ax2 = fig.add_subplot(132)
    ax2.plot(times, vxs, label='vx')
    ax2.plot(times, vys, label='vy')
    ax2.plot(times, vzs, label='vz')
    ax2.set_title('Velocity')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.legend()

    ax3 = fig.add_subplot(133)
    ax3.plot(times, axs, label='ax')
    ax3.plot(times, ays, label='ay')
    ax3.plot(times, azs, label='az')
    ax3.set_title('Acceleration')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Acceleration (m/s²)')
    ax3.legend()

    plt.tight_layout()
    plt.show()
