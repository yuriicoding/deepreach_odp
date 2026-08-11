import numpy as np

try:
    import heterocl as hcl
    HCL_AVAILABLE = True
except ImportError:
    hcl = None
    HCL_AVAILABLE = False

_G_DEFAULT = 9.81       # gravitational acceleration [m/s^2]
_U_DEFAULT = 2.0        # |u_theta| bound [rad/s^2] — shared_l2_control U_MAX

"""
4D X-SUBSYSTEM OF QuadrotorHover10D
(near-hover longitudinal channel: position / velocity / pitch / pitch rate)

State  : [px, vx, theta, q]      units: m, m/s, rad, rad/s
Control: [u_theta]               units: rad/s^2

px_dot    = vx
vx_dot    = g * theta
theta_dot = q
q_dot     = u_theta

The full 10D state is x = (px, vx, theta, q, py, vy, phi, p, pz, vz) and splits
into three subsystems that share NO state:

    X: [px, vx, theta, q]  (this file, control u_theta)
    Y: [py, vy, phi,   p]  (QuadrotorHover10DY, control u_phi)
    Z: [pz, vz]            (QuadrotorHover10DZ, control u_z)

Control coupling depends on the configuration (see
examples/quadrotor10d_decomposed_example.py):
  * independent_control (|u_theta|<=2, |u_phi|<=2, |u_z|<=3): the three
    subsystems share nothing at all, so the decomposition is EXACT.
  * shared_l2_control (u_theta^2 + u_phi^2 + u_z^2 <= U_MAX^2): each subsystem
    is solved over its projected interval [-U_MAX, U_MAX], which is an
    OVER-approximation of the coupled admissible set (the "leaking corner":
    the box [-U,U]^3 strictly contains the L2 ball). The reconstruction
    max(Vx, Vy, Vz) is then a LOWER bound on the true coupled value function
    (an over-approximation of the BRT). This is the configuration DeepReach's
    QuadrotorHover10D solves jointly, without that relaxation.

Hamiltonian (reach, uMode="min"):
  H = p_px*vx + p_vx*(g*theta) + p_theta*q + p_q*u_theta

  u_theta term: p_q * u_theta  ->  minimise over u_theta using sign(p_q),
  i.e. u* = uMin when p_q > 0, uMax otherwise (bang-bang; the interval
  constraint per subsystem is what makes this valid — the joint L2 problem
  has the non-bang-bang solution u* = -U_MAX * c/||c||).
"""


class QuadrotorHover10DX:
    def __init__(
        self,
        x=[0, 0, 0, 0],
        uMin=[-_U_DEFAULT],
        uMax=[_U_DEFAULT],
        dMin=[0.0],
        dMax=[0.0],
        uMode="min",
        dMode="max",
        gravity=_G_DEFAULT,
    ):
        """
        Args:
            x       : initial state [px (m), vx (m/s), theta (rad), q (rad/s)]
            uMin    : [u_theta_min (rad/s^2)]
            uMax    : [u_theta_max (rad/s^2)]
            uMode   : "min" for reach; "max" for avoid
            gravity : g [m/s^2]; must match DeepReach QuadrotorHover10D.gravity
        """
        self.x = x
        self.uMin = uMin
        self.uMax = uMax
        self.dMin = dMin
        self.dMax = dMax
        self.gravity = gravity

        assert uMode in ["min", "max"]
        self.uMode = uMode
        if uMode == "min":
            assert dMode == "max"
        else:
            assert dMode == "min"
        self.dMode = dMode

    # ------------------------------------------------------------------
    # HeteroCL interface
    # ------------------------------------------------------------------

    def opt_ctrl(self, t, state, spat_deriv):
        """Optimal bang-bang control in HeteroCL graph.

        state indices: 0=px, 1=vx, 2=theta, 3=q
        spat_deriv[3] = p_q  (the only costate the control enters through)
        """
        opt_u = hcl.scalar(self.uMax[0], "opt_u")
        in2   = hcl.scalar(0, "in2")
        in3   = hcl.scalar(0, "in3")
        in4   = hcl.scalar(0, "in4")

        if self.uMode == "min":
            with hcl.if_(spat_deriv[3] > 0):
                opt_u[0] = self.uMin[0]
        else:
            with hcl.if_(spat_deriv[3] < 0):
                opt_u[0] = self.uMin[0]

        return (opt_u[0], in2[0], in3[0], in4[0])

    def opt_dstb(self, t, state, spat_deriv):
        """No disturbance."""
        d1 = hcl.scalar(0, "d1")
        d2 = hcl.scalar(0, "d2")
        d3 = hcl.scalar(0, "d3")
        d4 = hcl.scalar(0, "d4")
        return (d1[0], d2[0], d3[0], d4[0])

    def dynamics(self, t, state, uOpt, dOpt):
        """Near-hover longitudinal dynamics in HeteroCL graph."""
        px_dot    = hcl.scalar(0, "px_dot")
        vx_dot    = hcl.scalar(0, "vx_dot")
        theta_dot = hcl.scalar(0, "theta_dot")
        q_dot     = hcl.scalar(0, "q_dot")

        px_dot[0]    = state[1]
        vx_dot[0]    = self.gravity * state[2]
        theta_dot[0] = state[3]
        q_dot[0]     = uOpt[0]

        return (px_dot[0], vx_dot[0], theta_dot[0], q_dot[0])

    # ------------------------------------------------------------------
    # Pure-Python helpers (trajectory rollout / post-processing)
    # ------------------------------------------------------------------

    def optCtrl_inPython(self, spat_deriv):
        """Bang-bang optimal control in plain Python."""
        if self.uMode == "min":
            opt_u = self.uMin[0] if spat_deriv[3] > 0 else self.uMax[0]
        else:
            opt_u = self.uMax[0] if spat_deriv[3] > 0 else self.uMin[0]
        return (opt_u,)

    def dynamics_inPython(self, state, action):
        """Near-hover longitudinal dynamics in plain Python.

        Args:
            state  : [px (m), vx (m/s), theta (rad), q (rad/s)]
            action : [u_theta (rad/s^2)]
        Returns:
            tuple (px_dot, vx_dot, theta_dot, q_dot)
        """
        px, vx, theta, q = state
        (u_theta,) = action

        px_dot    = vx
        vx_dot    = self.gravity * theta
        theta_dot = q
        q_dot     = u_theta

        return (px_dot, vx_dot, theta_dot, q_dot)
