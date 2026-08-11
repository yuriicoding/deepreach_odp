import numpy as np

try:
    import heterocl as hcl
    HCL_AVAILABLE = True
except ImportError:
    hcl = None
    HCL_AVAILABLE = False

_U_DEFAULT = 2.0        # |u_z| bound [m/s^2] — shared_l2_control U_MAX
                        # (independent_control uses 3.0; pass it explicitly)

"""
2D Z-SUBSYSTEM OF QuadrotorHover10D
(near-hover vertical channel: double integrator)

State  : [pz, vz]                units: m, m/s
Control: [u_z]                   units: m/s^2

pz_dot = vz
vz_dot = u_z

Independent of the X subsystem [px, vx, theta, q] and the Y subsystem
[py, vy, phi, p] — no shared state. See QuadrotorHover10DX for the
control-coupling discussion (exact for independent_control, over-approximating
for shared_l2_control).

Hamiltonian (reach, uMode="min"):
  H = p_pz*vz + p_vz*u_z

  u_z term: p_vz * u_z  ->  minimise over u_z using sign(p_vz).
"""


class QuadrotorHover10DZ:
    def __init__(
        self,
        x=[0, 0],
        uMin=[-_U_DEFAULT],
        uMax=[_U_DEFAULT],
        dMin=[0.0],
        dMax=[0.0],
        uMode="min",
        dMode="max",
    ):
        """
        Args:
            x     : initial state [pz (m), vz (m/s)]
            uMin  : [u_z_min (m/s^2)]
            uMax  : [u_z_max (m/s^2)]
            uMode : "min" for reach; "max" for avoid
        """
        self.x = x
        self.uMin = uMin
        self.uMax = uMax
        self.dMin = dMin
        self.dMax = dMax

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

        state indices: 0=pz, 1=vz
        spat_deriv[1] = p_vz
        """
        opt_u = hcl.scalar(self.uMax[0], "opt_u")
        in2   = hcl.scalar(0, "in2")

        if self.uMode == "min":
            with hcl.if_(spat_deriv[1] > 0):
                opt_u[0] = self.uMin[0]
        else:
            with hcl.if_(spat_deriv[1] < 0):
                opt_u[0] = self.uMin[0]

        return (opt_u[0], in2[0])

    def opt_dstb(self, t, state, spat_deriv):
        """No disturbance."""
        d1 = hcl.scalar(0, "d1")
        d2 = hcl.scalar(0, "d2")
        return (d1[0], d2[0])

    def dynamics(self, t, state, uOpt, dOpt):
        """Vertical double-integrator dynamics in HeteroCL graph."""
        pz_dot = hcl.scalar(0, "pz_dot")
        vz_dot = hcl.scalar(0, "vz_dot")

        pz_dot[0] = state[1]
        vz_dot[0] = uOpt[0]

        return (pz_dot[0], vz_dot[0])

    # ------------------------------------------------------------------
    # Pure-Python helpers (trajectory rollout / post-processing)
    # ------------------------------------------------------------------

    def optCtrl_inPython(self, spat_deriv):
        """Bang-bang optimal control in plain Python."""
        if self.uMode == "min":
            opt_u = self.uMin[0] if spat_deriv[1] > 0 else self.uMax[0]
        else:
            opt_u = self.uMax[0] if spat_deriv[1] > 0 else self.uMin[0]
        return (opt_u,)

    def dynamics_inPython(self, state, action):
        """Vertical double-integrator dynamics in plain Python.

        Args:
            state  : [pz (m), vz (m/s)]
            action : [u_z (m/s^2)]
        Returns:
            tuple (pz_dot, vz_dot)
        """
        pz, vz = state
        (u_z,) = action

        pz_dot = vz
        vz_dot = u_z

        return (pz_dot, vz_dot)
