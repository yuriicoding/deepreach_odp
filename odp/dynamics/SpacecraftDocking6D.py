import math
import numpy as np

try:
    import heterocl as hcl
    HCL_AVAILABLE = True
except ImportError:
    hcl = None
    HCL_AVAILABLE = False

# Default parameters — match paper code (Docking4D.py + Docking2D.py)
_MU        = 3.986004418e14
_R_EARTH   = 6371e3
_ORBIT_ALT = 400e3
_N_DEFAULT = math.sqrt(_MU / (_R_EARTH + _ORBIT_ALT) ** 3)  # ≈ 0.001133 rad/s
_M_DEFAULT = 200.0           # chaser mass [kg]
_I_DEFAULT = 200.0 / 6.0    # moment of inertia [kg*m^2] = 1/12 * m * (1^2 + 1^2)

"""
6D Planar Spacecraft Docking Dynamics
(Clohessy-Wiltshire translational + single-axis rotational)

State: x = [px, py, vx, vy, theta, omega]
  px, py   : chaser position relative to target in LVLH frame [m]
  vx, vy   : chaser velocity relative to target [m/s]
  theta    : chaser attitude (single-axis, rotation about z) [rad]
  omega    : angular rate [rad/s]

Dynamics:
  px_dot    = vx
  py_dot    = vy
  vx_dot    = 3*n^2*px + 2*n*vy + Fx/m
  vy_dot    = -2*n*vx + Fy/m
  theta_dot = omega
  omega_dot = tau/I

Controls: u = [Fx, Fy, tau]
  Fx, Fy : body-frame forces projected onto LVLH xy-plane [N]
  tau    : torque about z-axis [N*m]

Reference: Thorup et al., "Neural Backward Reach-Avoid Tubes with MPC Supervision
for High-Dimensional Systems," arXiv:2605.02021, 2026.
"""


class SpacecraftDocking6D:
    def __init__(
        self,
        x=[0, 0, 0, 0, 0, 0],
        uMin=[-20.0, -20.0, -1.5],
        uMax=[20.0, 20.0, 1.5],
        dMin=[0.0, 0.0, 0.0],
        dMax=[0.0, 0.0, 0.0],
        uMode="min",
        dMode="max",
        n=_N_DEFAULT,
        m=_M_DEFAULT,
        I=_I_DEFAULT,
    ):
        """
        Args:
            x      : initial state [px, py, vx, vy, theta, omega]
            uMin   : lower control bounds [Fx_min (N), Fy_min (N), tau_min (N*m)]
            uMax   : upper control bounds [Fx_max (N), Fy_max (N), tau_max (N*m)]
            dMin   : lower disturbance bounds (unused here, set to zero)
            dMax   : upper disturbance bounds (unused here, set to zero)
            uMode  : "min" for reach (controller minimises H), "max" for avoid
            dMode  : opposite of uMode
            n      : orbital mean motion [rad/s]; default ~400 km LEO
            m      : chaser mass [kg]
            I      : chaser moment of inertia about z-axis [kg*m^2]
        """
        self.x = x
        self.uMin = uMin
        self.uMax = uMax
        self.dMin = dMin
        self.dMax = dMax
        self.n = n
        self.m = m
        self.I = I

        assert uMode in ["min", "max"]
        self.uMode = uMode
        if uMode == "min":
            assert dMode == "max"
        else:
            assert dMode == "min"
        self.dMode = dMode

    def opt_ctrl(self, t, state, spat_deriv):
        """
        Bang-bang optimal control via sign of costate (B^T p).

        Control influence on dynamics:
          Fx  -> vx_dot  : coefficient p[2] / m
          Fy  -> vy_dot  : coefficient p[3] / m
          tau -> omega_dot: coefficient p[5] / I

        Since m, I > 0, sign(p[i] / m) == sign(p[i]).
        For uMode="min": choose u_i = uMin if p[i] > 0, else uMax.
        """
        opt_Fx  = hcl.scalar(self.uMax[0], "opt_Fx")
        opt_Fy  = hcl.scalar(self.uMax[1], "opt_Fy")
        opt_tau = hcl.scalar(self.uMax[2], "opt_tau")
        # ODP graph expects 6 control inputs; pad unused slots with zeros
        in4 = hcl.scalar(0, "in4")
        in5 = hcl.scalar(0, "in5")
        in6 = hcl.scalar(0, "in6")

        if self.uMode == "min":
            with hcl.if_(spat_deriv[2] > 0):
                opt_Fx[0] = self.uMin[0]
            with hcl.if_(spat_deriv[3] > 0):
                opt_Fy[0] = self.uMin[1]
            with hcl.if_(spat_deriv[5] > 0):
                opt_tau[0] = self.uMin[2]
        else:
            with hcl.if_(spat_deriv[2] < 0):
                opt_Fx[0] = self.uMin[0]
            with hcl.if_(spat_deriv[3] < 0):
                opt_Fy[0] = self.uMin[1]
            with hcl.if_(spat_deriv[5] < 0):
                opt_tau[0] = self.uMin[2]

        return (opt_Fx[0], opt_Fy[0], opt_tau[0], in4[0], in5[0], in6[0])

    def opt_dstb(self, t, state, spat_deriv):
        """No disturbance modelled (disturbance-free formulation)."""
        d1 = hcl.scalar(0, "d1")
        d2 = hcl.scalar(0, "d2")
        d3 = hcl.scalar(0, "d3")
        d4 = hcl.scalar(0, "d4")
        d5 = hcl.scalar(0, "d5")
        d6 = hcl.scalar(0, "d6")
        return (d1[0], d2[0], d3[0], d4[0], d5[0], d6[0])

    def dynamics(self, t, state, uOpt, dOpt):
        """
        HeteroCL dynamics for backward reachability.

        state[0]=px, state[1]=py, state[2]=vx, state[3]=vy,
        state[4]=theta, state[5]=omega
        uOpt[0]=Fx, uOpt[1]=Fy, uOpt[2]=tau
        """
        n     = self.n
        inv_m = 1.0 / self.m
        inv_I = 1.0 / self.I

        px_dot    = hcl.scalar(0, "px_dot")
        py_dot    = hcl.scalar(0, "py_dot")
        vx_dot    = hcl.scalar(0, "vx_dot")
        vy_dot    = hcl.scalar(0, "vy_dot")
        theta_dot = hcl.scalar(0, "theta_dot")
        omega_dot = hcl.scalar(0, "omega_dot")

        px_dot[0]    = state[2]
        py_dot[0]    = state[3]
        vx_dot[0]    = 3.0 * n * n * state[0] + 2.0 * n * state[3] + uOpt[0] * inv_m
        vy_dot[0]    = -2.0 * n * state[2] + uOpt[1] * inv_m
        theta_dot[0] = state[5]
        omega_dot[0] = uOpt[2] * inv_I

        return (px_dot[0], py_dot[0], vx_dot[0], vy_dot[0], theta_dot[0], omega_dot[0])

    # ------------------------------------------------------------------
    # Python-native helpers (for trajectory simulation, not HCL graph)
    # ------------------------------------------------------------------

    def optCtrl_inPython(self, spat_deriv):
        """Pure-Python bang-bang control (for trajectory rollout)."""
        if self.uMode == "min":
            opt_Fx  = self.uMin[0] if spat_deriv[2] > 0 else self.uMax[0]
            opt_Fy  = self.uMin[1] if spat_deriv[3] > 0 else self.uMax[1]
            opt_tau = self.uMin[2] if spat_deriv[5] > 0 else self.uMax[2]
        else:
            opt_Fx  = self.uMax[0] if spat_deriv[2] > 0 else self.uMin[0]
            opt_Fy  = self.uMax[1] if spat_deriv[3] > 0 else self.uMin[1]
            opt_tau = self.uMax[2] if spat_deriv[5] > 0 else self.uMin[2]
        return opt_Fx, opt_Fy, opt_tau

    def dynamics_inPython(self, state, action):
        """Pure-Python CW + rotation dynamics (no disturbance).

        Args:
            state  : array-like [px, py, vx, vy, theta, omega]
            action : array-like [Fx (N), Fy (N), tau (N*m)]

        Returns:
            tuple of six state derivatives
        """
        px, py, vx, vy, theta, omega = state
        Fx, Fy, tau = action
        n = self.n

        px_dot    = vx
        py_dot    = vy
        vx_dot    = 3.0 * n**2 * px + 2.0 * n * vy + Fx / self.m
        vy_dot    = -2.0 * n * vx + Fy / self.m
        theta_dot = omega
        omega_dot = tau / self.I

        return (px_dot, py_dot, vx_dot, vy_dot, theta_dot, omega_dot)

    def forward(self, ctrl_freq, current_state, action):
        """Forward-Euler step.

        Args:
            ctrl_freq     : control frequency [Hz]
            current_state : array-like [px, py, vx, vy, theta, omega]
            action        : array-like [Fx, Fy, tau]

        Returns:
            next_state (tuple of 6 floats)
        """
        dt = 1.0 / ctrl_freq
        derivs = self.dynamics_inPython(current_state, action)
        return tuple(s + d * dt for s, d in zip(current_state, derivs))
