import math
import numpy as np

try:
    import heterocl as hcl
    HCL_AVAILABLE = True
except ImportError:
    hcl = None
    HCL_AVAILABLE = False

# Default orbital parameters — match Docking4D.py in paper code
_MU        = 3.986004418e14
_R_EARTH   = 6371e3
_ORBIT_ALT = 400e3
_N_DEFAULT = math.sqrt(_MU / (_R_EARTH + _ORBIT_ALT) ** 3)  # ≈ 0.001133 rad/s
_M_DEFAULT = 200.0  # chaser mass [kg]

"""
4D TRANSLATIONAL SUBSYSTEM OF SpacecraftDocking6D
(Clohessy-Wiltshire dynamics, rotation-free)

State  : [px, py, vx, vy]
Control: [Fx, Fy]

px_dot = vx
py_dot = vy
vx_dot = 3*n^2*px + 2*n*vy + Fx/m
vy_dot = -2*n*vx          + Fy/m

The rotation subsystem [theta, omega] is completely independent — no shared
state or control — so this subsystem's BRT is computed independently and the
full 6D BRT is reconstructed exactly as max(V_trans, V_rot).

Hamiltonian (reach, uMode="min"):
  H = p_px*vx + p_py*vy
    + p_vx*(3n²px + 2n*vy + Fx/m)
    + p_vy*(-2n*vx + Fy/m)

  Fx term: p_vx * Fx / m  →  minimise over Fx using sign(p_vx)
  Fy term: p_vy * Fy / m  →  minimise over Fy using sign(p_vy)
"""


class SpacecraftDocking6DTrans:
    def __init__(
        self,
        x=[0, 0, 0, 0],
        uMin=[-20.0, -20.0],
        uMax=[20.0, 20.0],
        dMin=[0.0, 0.0],
        dMax=[0.0, 0.0],
        uMode="min",
        dMode="max",
        n=_N_DEFAULT,
        m=_M_DEFAULT,
    ):
        """
        Args:
            x     : initial state [px, py, vx, vy]
            uMin  : [Fx_min (N), Fy_min (N)]
            uMax  : [Fx_max (N), Fy_max (N)]
            uMode : "min" for reach; "max" for avoid
            n     : orbital mean motion [rad/s]
            m     : chaser mass [kg]
        """
        self.x = x
        self.uMin = uMin
        self.uMax = uMax
        self.dMin = dMin
        self.dMax = dMax
        self.n = n
        self.m = m

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

        state indices: 0=px, 1=py, 2=vx, 3=vy
        spat_deriv[2] = p_vx, spat_deriv[3] = p_vy
        """
        opt_Fx = hcl.scalar(self.uMax[0], "opt_Fx")
        opt_Fy = hcl.scalar(self.uMax[1], "opt_Fy")
        in3    = hcl.scalar(0, "in3")
        in4    = hcl.scalar(0, "in4")

        if self.uMode == "min":
            with hcl.if_(spat_deriv[2] > 0):
                opt_Fx[0] = self.uMin[0]
            with hcl.if_(spat_deriv[3] > 0):
                opt_Fy[0] = self.uMin[1]
        else:
            with hcl.if_(spat_deriv[2] < 0):
                opt_Fx[0] = self.uMin[0]
            with hcl.if_(spat_deriv[3] < 0):
                opt_Fy[0] = self.uMin[1]

        return (opt_Fx[0], opt_Fy[0], in3[0], in4[0])

    def opt_dstb(self, t, state, spat_deriv):
        """No disturbance."""
        d1 = hcl.scalar(0, "d1")
        d2 = hcl.scalar(0, "d2")
        d3 = hcl.scalar(0, "d3")
        d4 = hcl.scalar(0, "d4")
        return (d1[0], d2[0], d3[0], d4[0])

    def dynamics(self, t, state, uOpt, dOpt):
        """CW translational dynamics in HeteroCL graph."""
        n     = self.n
        inv_m = 1.0 / self.m

        px_dot = hcl.scalar(0, "px_dot")
        py_dot = hcl.scalar(0, "py_dot")
        vx_dot = hcl.scalar(0, "vx_dot")
        vy_dot = hcl.scalar(0, "vy_dot")

        px_dot[0] = state[2]
        py_dot[0] = state[3]
        vx_dot[0] = 3.0 * n * n * state[0] + 2.0 * n * state[3] + uOpt[0] * inv_m
        vy_dot[0] = -2.0 * n * state[2]                           + uOpt[1] * inv_m

        return (px_dot[0], py_dot[0], vx_dot[0], vy_dot[0])

    # ------------------------------------------------------------------
    # Pure-Python helpers (trajectory rollout / post-processing)
    # ------------------------------------------------------------------

    def optCtrl_inPython(self, spat_deriv):
        """Bang-bang optimal control in plain Python."""
        if self.uMode == "min":
            opt_Fx = self.uMin[0] if spat_deriv[2] > 0 else self.uMax[0]
            opt_Fy = self.uMin[1] if spat_deriv[3] > 0 else self.uMax[1]
        else:
            opt_Fx = self.uMax[0] if spat_deriv[2] > 0 else self.uMin[0]
            opt_Fy = self.uMax[1] if spat_deriv[3] > 0 else self.uMin[1]
        return opt_Fx, opt_Fy

    def dynamics_inPython(self, state, action):
        """CW translational dynamics in plain Python.

        Args:
            state  : [px, py, vx, vy]
            action : [Fx (N), Fy (N)]
        Returns:
            tuple (px_dot, py_dot, vx_dot, vy_dot)
        """
        px, py, vx, vy = state
        Fx, Fy = action
        n = self.n

        px_dot = vx
        py_dot = vy
        vx_dot = 3.0 * n**2 * px + 2.0 * n * vy + Fx / self.m
        vy_dot = -2.0 * n * vx                    + Fy / self.m

        return (px_dot, py_dot, vx_dot, vy_dot)
