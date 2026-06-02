import numpy as np
import math

try:
    import heterocl as hcl
    HCL_AVAILABLE = True
except ImportError:
    hcl = None
    HCL_AVAILABLE = False

"""
3D Y-SUBSYSTEM OF DubinsCar4D3 (bicycle model, no disturbance)

Decomposed from DubinsCar4D3 following Chen et al. (2018), Sec. III-IV.
Full state (x, y, v, theta) splits as z1=x, z2=y, zc=(v, theta).

State  : [y, v, theta]   (x dropped)
Control: [a, delta]       (shared with Vx subsystem)

y_dot     = v * sin(theta)
v_dot     = a
theta_dot = v * tan(delta) / L

Hamiltonian control terms:
  p_v     * a                      -> optimise on sign(spat_deriv[1])
  p_theta * v * tan(delta) / L     -> optimise on sign(state[1] * spat_deriv[2])
"""


class DubinsCar4D3Vy:
    def __init__(
        self,
        x=[0, 0, 0],
        uMin=[-1.5, -math.pi / 18],
        uMax=[1.5, math.pi / 18],
        uMode="max",
        wheelbase=0.3,
    ):
        """
        Args:
            x        : initial state [y, v, theta]
            uMin     : [a_min, delta_min]
            uMax     : [a_max, delta_max]
            uMode    : "max" -> avoid; "min" -> reach
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

        state[0]=y, state[1]=v, state[2]=theta
        spat_deriv[1] -> p_v,  spat_deriv[2] -> p_theta
        """
        opt_a     = hcl.scalar(self.uMax[0], "opt_a")
        opt_delta = hcl.scalar(self.uMax[1], "opt_delta")
        in3       = hcl.scalar(0, "in3")

        if self.uMode == "min":
            with hcl.if_(spat_deriv[1] > 0):
                opt_a[0] = self.uMin[0]
            with hcl.if_(state[1] * spat_deriv[2] > 0):
                opt_delta[0] = self.uMin[1]
        else:
            with hcl.if_(spat_deriv[1] < 0):
                opt_a[0] = self.uMin[0]
            with hcl.if_(state[1] * spat_deriv[2] < 0):
                opt_delta[0] = self.uMin[1]

        return (opt_a[0], opt_delta[0], in3[0])

    def opt_dstb(self, t, state, spat_deriv):
        """No disturbance — returns zeros."""
        d1 = hcl.scalar(0, "d1")
        d2 = hcl.scalar(0, "d2")
        d3 = hcl.scalar(0, "d3")
        return (d1[0], d2[0], d3[0])

    def dynamics(self, t, state, uOpt, dOpt):
        L         = hcl.scalar(self.wheelbase, "L")
        y_dot     = hcl.scalar(0, "y_dot")
        v_dot     = hcl.scalar(0, "v_dot")
        theta_dot = hcl.scalar(0, "theta_dot")

        y_dot[0]     = state[1] * hcl.sin(state[2])
        v_dot[0]     = uOpt[0]
        theta_dot[0] = state[1] * (hcl.sin(uOpt[1]) / hcl.cos(uOpt[1])) / L[0]

        return (y_dot[0], v_dot[0], theta_dot[0])

    # ------------------------------------------------------------------
    # Pure-Python helpers
    # ------------------------------------------------------------------

    def optCtrl_inPython(self, state, spat_deriv):
        """Optimal control in plain Python.

        Args:
            state      : array-like [y, v, theta]
            spat_deriv : array-like length 3
        Returns:
            (opt_a, opt_delta)
        """
        opt_a     = self.uMax[0]
        opt_delta = self.uMax[1]

        if self.uMode == "min":
            if spat_deriv[1] > 0:
                opt_a = self.uMin[0]
            if state[1] * spat_deriv[2] > 0:
                opt_delta = self.uMin[1]
        else:
            if spat_deriv[1] < 0:
                opt_a = self.uMin[0]
            if state[1] * spat_deriv[2] < 0:
                opt_delta = self.uMin[1]

        return opt_a, opt_delta

    def dynamics_inPython(self, state, action):
        """First-order state derivative in plain Python.

        Args:
            state  : array-like [y, v, theta]
            action : array-like [a, delta]
        Returns:
            tuple (y_dot, v_dot, theta_dot)
        """
        y_dot     = state[1] * np.sin(state[2])
        v_dot     = action[0]
        theta_dot = state[1] * np.tan(action[1]) / self.wheelbase
        return (y_dot, v_dot, theta_dot)
