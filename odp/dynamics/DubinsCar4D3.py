import numpy as np
import math

try:
    import heterocl as hcl
    HCL_AVAILABLE = True
except ImportError:
    hcl = None
    HCL_AVAILABLE = False

"""
4D DUBINS CAR - BICYCLE MODEL, NO DISTURBANCE

State : [x, y, v, theta]
Control: [a, delta]  (acceleration, steering angle)

x_dot     = v * cos(theta)
y_dot     = v * sin(theta)
v_dot     = a
theta_dot = v * tan(delta) / L

L := wheelbase (default 0.3, Tamiya TT02)

Mirrors Dubins4D_new in dynamics.py.  Optimal steering is based on
sign(p_theta * v) — the correct condition for the bicycle-model
Hamiltonian — rather than sign(p_theta) alone.

(6.2) https://arxiv.org/pdf/1711.03449.pdf
"""


class DubinsCar4D3:
    def __init__(
        self,
        x=[0, 0, 0, 0],
        uMin=[-1.5, -math.pi / 18],
        uMax=[1.5, math.pi / 18],
        uMode="max",
        wheelbase=0.3,
    ):
        """
        Args:
            x        : initial state [x, y, v, theta]
            uMin     : [a_min, delta_min]
            uMax     : [a_max, delta_max]
            uMode    : "max" → avoid (maximise V); "min" → reach (minimise V)
            wheelbase: vehicle wheelbase L
        """
        self.x = x
        self.uMin = uMin
        self.uMax = uMax
        assert uMode in ["min", "max"]
        self.uMode = uMode
        self.wheelbase = wheelbase

    # ------------------------------------------------------------------
    # HeteroCL interface
    # ------------------------------------------------------------------

    def opt_ctrl(self, t, state, spat_deriv):
        """Optimal control in HeteroCL graph.

        Hamiltonian contributions:
          p_v * a                        (acceleration)
          p_theta * v * tan(delta) / L   (steering)

        avoid (uMode="max"): choose control that maximises each term.
        reach (uMode="min"): choose control that minimises each term.
        """
        opt_a     = hcl.scalar(self.uMax[0], "opt_a")
        opt_delta = hcl.scalar(self.uMax[1], "opt_delta")
        in3       = hcl.scalar(0, "in3")
        in4       = hcl.scalar(0, "in4")

        if self.uMode == "min":
            # reach: minimise Hamiltonian
            with hcl.if_(spat_deriv[2] > 0):
                opt_a[0] = self.uMin[0]
            with hcl.if_(state[2] * spat_deriv[3] > 0):
                opt_delta[0] = self.uMin[1]
        else:
            # avoid: maximise Hamiltonian
            with hcl.if_(spat_deriv[2] < 0):
                opt_a[0] = self.uMin[0]
            with hcl.if_(state[2] * spat_deriv[3] < 0):
                opt_delta[0] = self.uMin[1]

        return (opt_a[0], opt_delta[0], in3[0], in4[0])

    def opt_dstb(self, t, state, spat_deriv):
        """No disturbance — returns zeros."""
        d1 = hcl.scalar(0, "d1")
        d2 = hcl.scalar(0, "d2")
        d3 = hcl.scalar(0, "d3")
        d4 = hcl.scalar(0, "d4")
        return (d1[0], d2[0], d3[0], d4[0])

    def dynamics(self, t, state, uOpt, dOpt):
        L         = hcl.scalar(self.wheelbase, "L")
        x_dot     = hcl.scalar(0, "x_dot")
        y_dot     = hcl.scalar(0, "y_dot")
        v_dot     = hcl.scalar(0, "v_dot")
        theta_dot = hcl.scalar(0, "theta_dot")

        x_dot[0]     = state[2] * hcl.cos(state[3])
        y_dot[0]     = state[2] * hcl.sin(state[3])
        v_dot[0]     = uOpt[0]
        theta_dot[0] = state[2] * (hcl.sin(uOpt[1]) / hcl.cos(uOpt[1])) / L[0]

        return (x_dot[0], y_dot[0], v_dot[0], theta_dot[0])

    # ------------------------------------------------------------------
    # Pure-Python helpers (post-solve trajectory simulation / analysis)
    # ------------------------------------------------------------------

    def optCtrl_inPython(self, state, spat_deriv):
        """Optimal control in plain Python (for post-processing).

        Args:
            state      : array-like of length 4, [x, y, v, theta]
            spat_deriv : array-like of length 4, spatial derivative of V
        Returns:
            (opt_a, opt_delta)
        """
        opt_a     = self.uMax[0]
        opt_delta = self.uMax[1]

        if self.uMode == "min":
            if spat_deriv[2] > 0:
                opt_a = self.uMin[0]
            if state[2] * spat_deriv[3] > 0:
                opt_delta = self.uMin[1]
        else:
            if spat_deriv[2] < 0:
                opt_a = self.uMin[0]
            if state[2] * spat_deriv[3] < 0:
                opt_delta = self.uMin[1]

        return opt_a, opt_delta

    def dynamics_inPython(self, state, action):
        """First-order state derivative in plain Python.

        Args:
            state  : array-like [x, y, v, theta]
            action : array-like [a, delta]
        Returns:
            tuple (x_dot, y_dot, v_dot, theta_dot)
        """
        x_dot     = state[2] * np.cos(state[3])
        y_dot     = state[2] * np.sin(state[3])
        v_dot     = action[0]
        theta_dot = state[2] * np.tan(action[1]) / self.wheelbase
        return (x_dot, y_dot, v_dot, theta_dot)
