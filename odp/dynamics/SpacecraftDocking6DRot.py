import numpy as np

try:
    import heterocl as hcl
    HCL_AVAILABLE = True
except ImportError:
    hcl = None
    HCL_AVAILABLE = False

"""
2D ROTATIONAL SUBSYSTEM OF SpacecraftDocking6D
(single-axis attitude dynamics)

State  : [theta, omega]
Control: [tau]

theta_dot = omega
omega_dot = tau / I

The translation subsystem [px, py, vx, vy] is completely independent — no
shared state or control — so this BRT is solved independently and the full
6D BRT is reconstructed exactly as max(V_trans, V_rot).

Hamiltonian (reach, uMode="min"):
  H = p_theta * omega + p_omega * (tau / I)

  tau term: p_omega * tau / I  →  minimise over tau using sign(p_omega)
  (I > 0 so sign(p_omega / I) == sign(p_omega))
"""


class SpacecraftDocking6DRot:
    def __init__(
        self,
        x=[0, 0],
        uMin=[-1.5],
        uMax=[1.5],
        dMin=[0.0],
        dMax=[0.0],
        uMode="min",
        dMode="max",
        I=50.0,
    ):
        """
        Args:
            x     : initial state [theta (rad), omega (rad/s)]
            uMin  : [tau_min (N*m)]
            uMax  : [tau_max (N*m)]
            uMode : "min" for reach; "max" for avoid
            I     : moment of inertia about z-axis [kg*m^2]
        """
        self.x = x
        self.uMin = uMin
        self.uMax = uMax
        self.dMin = dMin
        self.dMax = dMax
        self.I = I

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

        state indices: 0=theta, 1=omega
        spat_deriv[1] = p_omega
        """
        opt_tau = hcl.scalar(self.uMax[0], "opt_tau")
        in2     = hcl.scalar(0, "in2")

        if self.uMode == "min":
            with hcl.if_(spat_deriv[1] > 0):
                opt_tau[0] = self.uMin[0]
        else:
            with hcl.if_(spat_deriv[1] < 0):
                opt_tau[0] = self.uMin[0]

        return (opt_tau[0], in2[0])

    def opt_dstb(self, t, state, spat_deriv):
        """No disturbance."""
        d1 = hcl.scalar(0, "d1")
        d2 = hcl.scalar(0, "d2")
        return (d1[0], d2[0])

    def dynamics(self, t, state, uOpt, dOpt):
        """Rotational dynamics in HeteroCL graph."""
        inv_I     = 1.0 / self.I

        theta_dot = hcl.scalar(0, "theta_dot")
        omega_dot = hcl.scalar(0, "omega_dot")

        theta_dot[0] = state[1]
        omega_dot[0] = uOpt[0] * inv_I

        return (theta_dot[0], omega_dot[0])

    # ------------------------------------------------------------------
    # Pure-Python helpers (trajectory rollout / post-processing)
    # ------------------------------------------------------------------

    def optCtrl_inPython(self, spat_deriv):
        """Bang-bang optimal control in plain Python."""
        if self.uMode == "min":
            opt_tau = self.uMin[0] if spat_deriv[1] > 0 else self.uMax[0]
        else:
            opt_tau = self.uMax[0] if spat_deriv[1] > 0 else self.uMin[0]
        return (opt_tau,)

    def dynamics_inPython(self, state, action):
        """Rotational dynamics in plain Python.

        Args:
            state  : [theta (rad), omega (rad/s)]
            action : [tau (N*m)]
        Returns:
            tuple (theta_dot, omega_dot)
        """
        theta, omega = state
        (tau,) = action

        theta_dot = omega
        omega_dot = tau / self.I

        return (theta_dot, omega_dot)
