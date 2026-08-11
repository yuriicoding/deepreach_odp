import numpy as np

try:
    import heterocl as hcl
    HCL_AVAILABLE = True
except ImportError:
    hcl = None
    HCL_AVAILABLE = False

_G_DEFAULT = 9.81       # gravitational acceleration [m/s^2]
_U_DEFAULT = 2.0        # |u_phi| bound [rad/s^2] — shared_l2_control U_MAX

"""
4D Y-SUBSYSTEM OF QuadrotorHover10D
(near-hover lateral channel: position / velocity / roll / roll rate)

State  : [py, vy, phi, p]        units: m, m/s, rad, rad/s
Control: [u_phi]                 units: rad/s^2

py_dot   = vy
vy_dot   = -g * phi              <-- NOTE the sign; the X channel has +g*theta
phi_dot  = p
p_dot    = u_phi

Independent of the X subsystem [px, vx, theta, q] and the Z subsystem
[pz, vz] — no shared state. See QuadrotorHover10DX for the control-coupling
discussion (exact for independent_control, over-approximating for
shared_l2_control).

Mirror symmetry (useful as a solver sanity check): substituting
(phi, p) -> (-phi, -p) maps these dynamics onto the X subsystem's, and both
the target box and the control interval are symmetric, so with identical
grids and bounds
    V_Y(py, vy, phi, p, t) == V_X(py, vy, -phi, -p, t)
to within grid/numerical error. The example script solves both anyway rather
than mirroring one into the other.

Hamiltonian (reach, uMode="min"):
  H = p_py*vy + p_vy*(-g*phi) + p_phi*p + p_p*u_phi

  u_phi term: p_p * u_phi  ->  minimise over u_phi using sign(p_p).
"""


class QuadrotorHover10DY:
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
            x       : initial state [py (m), vy (m/s), phi (rad), p (rad/s)]
            uMin    : [u_phi_min (rad/s^2)]
            uMax    : [u_phi_max (rad/s^2)]
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

        state indices: 0=py, 1=vy, 2=phi, 3=p
        spat_deriv[3] = p_p  (the only costate the control enters through)
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
        """Near-hover lateral dynamics in HeteroCL graph."""
        py_dot  = hcl.scalar(0, "py_dot")
        vy_dot  = hcl.scalar(0, "vy_dot")
        phi_dot = hcl.scalar(0, "phi_dot")
        p_dot   = hcl.scalar(0, "p_dot")

        py_dot[0]  = state[1]
        vy_dot[0]  = -self.gravity * state[2]
        phi_dot[0] = state[3]
        p_dot[0]   = uOpt[0]

        return (py_dot[0], vy_dot[0], phi_dot[0], p_dot[0])

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
        """Near-hover lateral dynamics in plain Python.

        Args:
            state  : [py (m), vy (m/s), phi (rad), p (rad/s)]
            action : [u_phi (rad/s^2)]
        Returns:
            tuple (py_dot, vy_dot, phi_dot, p_dot)
        """
        py, vy, phi, p = state
        (u_phi,) = action

        py_dot  = vy
        vy_dot  = -self.gravity * phi
        phi_dot = p
        p_dot   = u_phi

        return (py_dot, vy_dot, phi_dot, p_dot)
