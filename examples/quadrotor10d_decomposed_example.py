"""
Example: Decomposed BRS/BRT for QuadrotorHover10D (near-hover quadrotor)

The 10D state
    x = (px, vx, theta, q,   py, vy, phi, p,   pz, vz)
splits into 3 subsystems that share NO state:
  - QuadrotorHover10DX : [px, vx, theta, q]   (4D, control u_theta)
  - QuadrotorHover10DY : [py, vy, phi,   p]   (4D, control u_phi)
  - QuadrotorHover10DZ : [pz, vz]             (2D, control u_z)

Target: axis-aligned box |x_i| <= TARGET_RADIUS in every one of the 10
components (ShapeRectangle, i.e. an L-infinity box, NOT a ball). It is exactly
the intersection of the 3 subsystem boxes, since all 3 use the same radius —
which is what makes the terminal cost max-decomposable in the first place.

NO FULL 10D GRID IS PRODUCED
----------------------------
Unlike the 6D docking example (which reconstructs v_brat_all on the full grid),
this script writes ONLY the three subsystem arrays. The full 10D grid would be
41^4 * 41^4 * 81^2 ~= 5.2e16 cells (~2e8 GB per time step) — not
representable at any resolution worth having. Downstream consumers evaluate

    BRS at one t :  V_10D(x, t) = max_i V_i(x_i, t)
    BRT over [0,t]: V_10D(x, t) = min_{s <= t} max_i V_i(x_i, s)

on demand by interpolating each subsystem array at the projected state — see
reconstruct_brt_pointwise() below.

PER-STEP BRS — the ordering the decomposition theory requires
-------------------------------------------------------------
Each subsystem is solved with TargetSetMode="none", so NO min(V, l) clamp is
applied and the saved arrays are the PER-TIME-STEP BRS. That is what the
max-decomposition needs: Chen et al. 2018 (Prop. 1 + 4) holds per time step,
and the union-over-time does not commute with the intersection-over-subsystems:

    EXACT   (what this does)  :  V_BRT(x, t) = min_{s<=t} max_i V_i^BRS(x_i, s)
    WRONG   (what v2 did)     :  V_BRT(x, t) = max_i min_{s<=t} V_i(x_i, s)

Since max_i min_s (...) <= min_s max_i (...), the v2 form under-estimated the
value and so OVER-estimated the BRT. The two agree only if each subsystem can
wait inside its own target box until the others arrive, and that box is not
control-invariant here: from the vx = 0.2 corner, px overshoots to ~0.28 before
theta can decelerate it.

Consequence for consumers: the min over time is now THEIRS to take. A v2
consumer that takes only the max gets the BRS at a single t, not the BRT. The
manifest renames the keys (v_sub_*_brt -> v_sub_*_brs) so that such a consumer
fails loudly instead of silently changing meaning.

What the v2 clamp was hiding, and what replaced it
--------------------------------------------------
v2 justified the clamp with: "the true range is [min l, max l] = [-0.2, 5.8],
but 'none' reaches max |V| = 328." That reference value is WRONG for a BRS.

V_BRS(x, t) = min_u l(x(0)) is the terminal cost at a state the trajectory has
actually REACHED, and over t = 3 s these trajectories leave the grid by a wide
margin — so V is not bounded above by max l over the grid. The bound is valid
for a BRT, where V(x,t) <= l(x) by staying put, which is precisely why the
clamped v2 solve satisfied it BY CONSTRUCTION and the check proved nothing.

The correct a-priori bounds for a BRS are:

    lower   V(z,t) >= min l = -r        globally valid, since
                                        l = max_i|z_i| - r >= -r everywhere
    upper   V(z,t) <= l(Phi_t^u(z))     for ANY admissible control u, because
                                        the true V minimises over all of them

Every subsystem here is a linear chain of integrators, so the flow under a
CONSTANT control is a closed-form polynomial and that upper envelope costs
nothing to evaluate over the whole grid. free_flow_upper_bound() sweeps a set
of constant controls and keeps the pointwise min; value_bounds_report() then
scores the solve against that envelope instead of against 5.8, separately
inside and outside the DeepReach reporting window.

Treat V magnitudes near a domain face as unreliable regardless: odp closes
non-periodic boundaries with ghost-cell extrapolation, and switching the clamp
off exposes that layer rather than removing it. The envelope is what tells you
how far it has intruded.

Time-axis convention (same as HJSolver / the 6D example):
    index  0 (first) = t = tmax
    index -1 (last)  = t = 0  (the initial target function)

Reference arrays circulated as Vx_time.npy / Vy_time.npy use the OPPOSITE
order (index 0 = t = 0). They are otherwise bit-for-bit identical to this
script's output; reverse the last axis before comparing.

CONTROL CONFIGURATION: shared_l2_control only
---------------------------------------------
DeepReach's QuadrotorHover10D solves the joint system under the coupled bound

    u_theta^2 + u_phi^2 + u_z^2 <= U_MAX^2

Each subsystem here is solved over its PROJECTED interval [-U_MAX, U_MAX],
whose Cartesian product [-U_MAX, U_MAX]^3 strictly contains that L2 ball (the
"leaking corner"). So the decomposed reconstruction is an OVER-approximation of
the true coupled BRT — a lower bound on the value function, not an equality.
That gap is exactly what the joint DeepReach model is meant to close, and this
grid solve is its ground-truth-side reference. Do not report the reconstruction
as the exact coupled value function.

Output (written to --out_dir, default <repo root>/output_QuadrotorHover10D_decomposed/,
alongside the other output_* dirs and covered by .gitignore's /output_* rule):
  v_sub_x_brs.npy  — X BRS at all time steps, (npx, nvx, nth, nq, T)   ~1.4 GB
  v_sub_y_brs.npy  — Y BRS at all time steps, (npy, nvy, nph, np_, T)  ~1.4 GB
  v_sub_z_brs.npy  — Z BRS at all time steps, (npz, nvz, T)            ~3.2 MB
(float64 on disk, though odp computes in float32 — HJSolver's all-time-steps
 buffer is a plain np.zeros(...). Halve the sizes by saving as float32.)
  artifact_manifest.json / metrics.json
"""

import argparse
import json
import os
import time

import numpy as np

from odp.Grid import Grid
from odp.Shapes import ShapeRectangle
from odp.dynamics import QuadrotorHover10DX, QuadrotorHover10DY, QuadrotorHover10DZ
from odp.solver import HJSolver


# Anchor the default output to the repo root (the parent of examples/), so the
# results land next to output_DubinsCar4D3 / output_SpacecraftDocking6D_decomposed
# no matter which directory the script is launched from. A relative default
# would follow the CWD into examples/, where .gitignore's root-anchored
# /output_* rule does not reach.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT_DIR = os.path.join(_REPO_ROOT, "output_QuadrotorHover10D_decomposed")


# ---------------------------------------------------------------------------
# Configuration — must stay in sync with DeepReach's QuadrotorHover10D
# (state_var / state_test_range / u_max / target_radius / gravity)
# ---------------------------------------------------------------------------
CFG = {
    # time — same horizon for all 3 subsystems (max(Vx,Vy,Vz) is only the
    # joint BRS at a single t if every subsystem was solved over that same t),
    # and must equal DeepReach's tMax for the two to be comparable.
    "tmax": 3.0,
    "dt": 0.05,
    "small_number": 1e-5,

    # grids — units are SI: m, m/s, rad, rad/s
    # v_sub_x_brs / v_sub_y_brs: 41^4 x T x 8 bytes ~= 1.4 GB each (T = 61)
    # v_sub_z_brs:               81^2 x T x 8 bytes ~= 3.2 MB
    "px_min": -6.0,  "px_max": 6.0,
    "vx_min": -2.5,  "vx_max": 2.5,
    "th_min": -0.3, "th_max": 0.3,     # theta [rad]
    "q_min":  -1.5,  "q_max":  1.5,      # pitch rate [rad/s]
    "npx": 41, "nvx": 41, "nth": 41, "nq": 41,

    "py_min": -6.0,  "py_max": 6.0,
    "vy_min": -2.5,  "vy_max": 2.5,
    "ph_min": -0.3, "ph_max": 0.3,     # phi [rad]
    "p_min":  -1.5,  "p_max":  1.5,      # roll rate [rad/s]
    "npy": 41, "nvy": 41, "nph": 41, "np": 41,

    "pz_min": -6.0, "pz_max": 6.0,
    "vz_min": -2.5, "vz_max": 2.5,
    "npz": 81, "nvz": 81,

    # physics — gravity is NOT in DeepReach's state config; it is a constructor
    # arg there (default 9.81). Both sides must use this value.
    "gravity": 9.81,

    # control — shared_l2_control: u_theta^2 + u_phi^2 + u_z^2 <= u_max^2,
    # projected to the interval [-u_max, u_max] for each subsystem.
    "u_max": 2.0,

    # target box half-width, applied to all 10 components
    "target_radius": 0.2,

    # solver — accuracy: ONLY "low" and "medium" exist. odp/computeGraphs/graph_4D.py
    # branches on exactly those two strings; anything else (e.g. "high") leaves
    # the spatial derivatives uncomputed, so V never evolves and the solve
    # silently returns the target function at every time step. Guarded below.
    "accuracy": "medium",

    # "none" -> no min(V, l) clamp -> the PER-STEP BRS that the decomposition
    # theory requires. Do not switch this back to "minVWithV0" without also
    # reverting the reconstruction: that mode bakes the min-over-time into each
    # subsystem, which is the wrong operand order (max_i min_s instead of
    # min_s max_i) and over-estimates the BRT. See the "PER-STEP BRS" section
    # of the module docstring. The clamp is also LOSSY — the BRS cannot be
    # recovered from clamped arrays, so the choice is made here or not at all.
    "target_set_mode": "none",
    "u_mode": "min",     # reach

    # DeepReach's state_test_range, i.e. the window these arrays are actually
    # REPORTED on. The grid above deliberately overhangs it (3x in position,
    # 1.7x in velocity, 1.2x in theta/phi) so the ghost-cell boundary layer has
    # somewhere to live outside the reported region. Note q and p are NOT
    # padded — they are the dims to watch in the bounds report below.
    "report_window": {
        "x": [[-2.0, 2.0], [-1.5, 1.5], [-0.25, 0.25], [-1.5, 1.5]],
        "y": [[-2.0, 2.0], [-1.5, 1.5], [-0.25, 0.25], [-1.5, 1.5]],
        "z": [[-2.0, 2.0], [-1.5, 1.5]],
    },

    # NOTE — working precision is float32 and is NOT configurable from here.
    # odp hardcodes hcl.config.init_dtype = hcl.Float(32) in HJSolver, so the
    # whole compute graph runs in single precision; there is no float_bits
    # argument to pass.
    #
    # A local float64 patch was tried and reverted, because it does NOT fix the
    # X/Y mirror mismatch that motivated it. That mismatch is not round-off:
    #
    #   * Starting from a bitwise-symmetric target, ONE substep already breaks
    #     the symmetry by ~1e-3 even at float64. The source is the ENO
    #     tie-break in
    #     odp/spatialDerivatives/secondOrderENO/second_orderENO4D.py:
    #     `if |D2_a| <= |D2_b|: use D2_a else D2_b`. Under the mirror map the
    #     two candidates swap places, so wherever they tie with D2_a = -D2_b
    #     the mirrored solve picks the opposite stencil and the derivative
    #     differs by O(1). A box target is piecewise linear, so those exact
    #     ties are common (~1300 cells per axis on a 15^4 test grid), and they
    #     are exact ties in real arithmetic — more mantissa bits does not make
    #     them go away.
    #   * That O(1e-3) seed is then amplified by the outflow boundary:
    #     |q| <= 1.5 rad/s crosses the +-0.25 rad theta window in 0.33 s, so
    #     nearly every characteristic exits the domain and meets the ghost-cell
    #     extrapolation BC, which is ill-conditioned. On a 21^4 X-only solve
    #     peak |V| grows from 1.80 (= max of the target) to 21.9 over 1.5 s,
    #     with the maximum sitting in the (vx, theta, q) = max corner, and the
    #     X subsystem's OWN symmetry V(-x) = V(x) degrades in lockstep (15.0).
    #
    # So the mirror number measures the boundary layer, not an X-vs-Y dynamics
    # error — X alone fails its own symmetry by the same order. float64 bought
    # accuracy in the bulk, not symmetry, and paid for it in runtime.
}


# ---------------------------------------------------------------------------
# Subsystem solvers
# ---------------------------------------------------------------------------

def solve_x(c, tau):
    """Solve X subsystem BRT at every time step.

    State = [px, vx, theta, q], control u_theta in [-u_max, u_max].
    Target = box |px|,|vx|,|theta|,|q| <= target_radius.

    Returns array of shape (npx, nvx, nth, nq, T).
    """
    g = Grid(
        np.array([c["px_min"], c["vx_min"], c["th_min"], c["q_min"]]),
        np.array([c["px_max"], c["vx_max"], c["th_max"], c["q_max"]]),
        4,
        np.array([c["npx"], c["nvx"], c["nth"], c["nq"]]),
        [],  # no periodic dimensions (theta stays small-angle, never wrapped)
    )

    r = c["target_radius"]
    target = ShapeRectangle(g, -r * np.ones(4), r * np.ones(4))

    dyn = QuadrotorHover10DX(
        uMin=[-c["u_max"]],
        uMax=[c["u_max"]],
        uMode=c["u_mode"],
        gravity=c["gravity"],
    )

    return HJSolver(
        dyn, g,
        target,
        tau,
        {"TargetSetMode": c["target_set_mode"]},
        saveAllTimeSteps=True,
        accuracy=c["accuracy"],
    )   # shape (npx, nvx, nth, nq, T)


def solve_y(c, tau):
    """Solve Y subsystem BRT at every time step.

    State = [py, vy, phi, p], control u_phi in [-u_max, u_max].
    Same target box as X; only the sign of the gravity term differs
    (vy_dot = -g*phi vs vx_dot = +g*theta).

    Returns array of shape (npy, nvy, nph, np, T).
    """
    g = Grid(
        np.array([c["py_min"], c["vy_min"], c["ph_min"], c["p_min"]]),
        np.array([c["py_max"], c["vy_max"], c["ph_max"], c["p_max"]]),
        4,
        np.array([c["npy"], c["nvy"], c["nph"], c["np"]]),
        [],
    )

    r = c["target_radius"]
    target = ShapeRectangle(g, -r * np.ones(4), r * np.ones(4))

    dyn = QuadrotorHover10DY(
        uMin=[-c["u_max"]],
        uMax=[c["u_max"]],
        uMode=c["u_mode"],
        gravity=c["gravity"],
    )

    return HJSolver(
        dyn, g,
        target,
        tau,
        {"TargetSetMode": c["target_set_mode"]},
        saveAllTimeSteps=True,
        accuracy=c["accuracy"],
    )   # shape (npy, nvy, nph, np, T)


def solve_z(c, tau):
    """Solve Z subsystem BRT at every time step.

    State = [pz, vz] (double integrator), control u_z in [-u_max, u_max].
    Target = box |pz|,|vz| <= target_radius.

    Returns array of shape (npz, nvz, T).
    """
    g = Grid(
        np.array([c["pz_min"], c["vz_min"]]),
        np.array([c["pz_max"], c["vz_max"]]),
        2,
        np.array([c["npz"], c["nvz"]]),
        [],
    )

    r = c["target_radius"]
    target = ShapeRectangle(g, -r * np.ones(2), r * np.ones(2))

    dyn = QuadrotorHover10DZ(
        uMin=[-c["u_max"]],
        uMax=[c["u_max"]],
        uMode=c["u_mode"],
    )

    return HJSolver(
        dyn, g,
        target,
        tau,
        {"TargetSetMode": c["target_set_mode"]},
        saveAllTimeSteps=True,
        accuracy=c["accuracy"],
    )   # shape (npz, nvz, T)


# ---------------------------------------------------------------------------
# Subsystem descriptors — grid axes, reporting window, closed-form flow
# ---------------------------------------------------------------------------
#
# Every subsystem is a chain of integrators driven by the control, so the flow
# under a CONSTANT control u held for a duration t is a polynomial in t. The
# expressions below are exact solutions of the ODE, not a discretisation:
#
#   X : px' = vx,  vx' = +g*theta,  theta' = q,  q' = u
#   Y : py' = vy,  vy' = -g*phi,    phi'   = p,  p' = u    (X with g -> -g)
#   Z : pz' = vz,  vz' = u
#
# They take the grid axes in broadcast shape and return the same, so one call
# evaluates the flow over an entire subsystem grid.

def _flow_pos_vel_ang(axes, t, u, g):
    """Closed-form flow for the X/Y subsystems: (pos, vel, angle, rate)."""
    p0, v0, a0, w0 = axes
    w = w0 + u * t
    a = a0 + w0 * t + u * t**2 / 2.0
    v = v0 + g * (a0 * t + w0 * t**2 / 2.0 + u * t**3 / 6.0)
    p = p0 + v0 * t + g * (a0 * t**2 / 2.0 + w0 * t**3 / 6.0 + u * t**4 / 24.0)
    return (p, v, a, w)


def _flow_double_integrator(axes, t, u, g):
    """Closed-form flow for the Z subsystem: (pos, vel). g is unused."""
    p0, v0 = axes
    return (p0 + v0 * t + u * t**2 / 2.0, v0 + u * t)


def make_spec(grid_min, grid_max, npts, window, flow, g):
    """Bundle the grid axes, the reporting-window mask and the flow for one
    subsystem, so the bound machinery below does not need to know which one it
    is looking at.

    `axes` are 1-D node coordinates; `axes_b` are the same reshaped to
    broadcast against a full subsystem grid. `in_window` marks the cells that
    lie inside DeepReach's state_test_range — everything outside it is padding
    that exists to give the ghost-cell boundary layer somewhere to sit.
    """
    k = len(npts)
    axes, axes_b, mask = [], [], None
    for i in range(k):
        a = np.linspace(grid_min[i], grid_max[i], npts[i])
        shape = [1] * k
        shape[i] = npts[i]
        ab = a.reshape(shape)
        axes.append(a)
        axes_b.append(ab)
        m = (ab >= window[i][0] - 1e-12) & (ab <= window[i][1] + 1e-12)
        mask = m if mask is None else (mask & m)
    return {
        "axes": axes,
        "axes_b": axes_b,
        "in_window": np.broadcast_to(mask, tuple(npts)),
        "flow": flow,
        "g": g,
    }


# ---------------------------------------------------------------------------
# A-priori value envelope
# ---------------------------------------------------------------------------

def free_flow_upper_bound(spec, t, r, u_max, n_candidates=9):
    """Pointwise UPPER bound on the per-step BRS value at horizon t.

    V_BRS(z, t) = min over ALL admissible controls of l(z(0)). Any particular
    admissible control therefore gives an upper bound, and the min over a
    family of them gives a tighter one. Here the family is constant controls
    sampled across [-u_max, u_max], for which the flow is closed-form.

    This is a genuine certificate, not a heuristic: whatever the solver
    reports above this envelope is numerical error, with no appeal.

    It is only an upper bound — a solve sitting comfortably below it is not
    thereby correct. Its job is to replace the v2 reference value of max l over
    the grid (5.8), which is a valid bound for a BRT but NOT for a BRS, since a
    BRS trajectory is free to leave the grid before the terminal cost is read.
    """
    best = None
    for u in np.linspace(-u_max, u_max, n_candidates):
        z = spec["flow"](spec["axes_b"], t, u, spec["g"])
        l = np.abs(z[0])
        for zi in z[1:]:
            l = np.maximum(l, np.abs(zi))
        l = l - r
        best = l if best is None else np.minimum(best, l)
    return np.broadcast_to(best, spec["in_window"].shape)


def value_bounds_report(v_all, spec, tau, r, u_max, tol=1e-5, rel_tol=0.10):
    """Score a per-step BRS solve against bounds that are actually valid for it.

        lower   V(z,t) >= -r                 l = max_i|z_i| - r >= -r globally
        upper   V(z,t) <= free-flow envelope  see free_flow_upper_bound()

    Reported separately over the full grid and over the DeepReach reporting
    window, because the two answer different questions. Exceedance on the full
    grid says the ghost-cell layer is active somewhere; exceedance INSIDE the
    window says it has reached the region anyone will actually read, which is
    the number that decides whether the padding is sufficient.

    Also traces the violation against horizon and derives `trusted_horizon`:
    the longest t up to which the in-window violation stays under `rel_tol` of
    the envelope. The boundary layer grows with horizon, so short horizons stay
    clean long after the full one has gone — which makes this far more useful
    than a single pass/fail over the whole run.

    Time-axis convention: index j holds horizon tau[T-1-j], so index 0 is tmax
    and index -1 is t = 0.
    """
    T = v_all.shape[-1]
    win = spec["in_window"]
    out = {
        "v_min": np.inf, "v_max": -np.inf,
        "v_min_window": np.inf, "v_max_window": -np.inf,
        "envelope_max": -np.inf,
        "over_envelope": 0.0, "over_envelope_window": 0.0,
        "under_lower": 0.0, "under_lower_window": 0.0,
        "worst_over_window_t": None,
        "profile": [],
    }
    for j in range(T):
        t = float(tau[T - 1 - j])
        env = free_flow_upper_bound(spec, t, r, u_max)
        v = v_all[..., j]
        vw = v[win]

        out["v_min"] = min(out["v_min"], float(v.min()))
        out["v_max"] = max(out["v_max"], float(v.max()))
        out["v_min_window"] = min(out["v_min_window"], float(vw.min()))
        out["v_max_window"] = max(out["v_max_window"], float(vw.max()))
        out["envelope_max"] = max(out["envelope_max"], float(env.max()))

        over = v - env
        out["over_envelope"] = max(out["over_envelope"], float(over.max()))
        ow = float(over[win].max())
        if ow > out["over_envelope_window"]:
            out["over_envelope_window"] = ow
            out["worst_over_window_t"] = t

        out["under_lower"] = max(out["under_lower"], float(-r - v.min()))
        out["under_lower_window"] = max(out["under_lower_window"], float(-r - vw.min()))

        env_w = max(float(env[win].max()), 1.0)   # floor: t=0 envelope is small
        out["profile"].append({
            "t": t,
            "over_window": max(0.0, ow),
            "over_window_rel": max(0.0, ow) / env_w,
        })

    # Longest clean prefix of horizons, in increasing t.
    trusted = 0.0
    for p in sorted(out["profile"], key=lambda d: d["t"]):
        if p["over_window_rel"] > rel_tol:
            break
        trusted = p["t"]
    out["trusted_horizon"] = trusted
    out["trusted_horizon_rel_tol"] = rel_tol

    out["lower_bound"] = float(-r)
    out["within_bounds"] = bool(
        out["over_envelope"] <= tol and out["under_lower"] <= tol)
    out["within_bounds_window"] = bool(
        out["over_envelope_window"] <= tol and out["under_lower_window"] <= tol)
    for k in ("over_envelope", "over_envelope_window", "under_lower",
              "under_lower_window"):
        out[k] = max(0.0, out[k])
    return out


# ---------------------------------------------------------------------------
# Reconstruction  (Chen et al. 2018, Prop. 1 + 4 — correct operand order)
# ---------------------------------------------------------------------------

def reconstruct_brt_pointwise(states, v_x_all, v_y_all, v_z_all, c):
    """Evaluate the full 10D BRT at arbitrary states, without a 10D grid.

        BRS at index j :  max_i V_i(x_i, j)
        BRT at index j :  min_{s <= t_j} max_i V_i(x_i, s)  =  min over k >= j

    The max is taken FIRST, per time step, and only then the min over time —
    that operand order is the whole point of this revision, and it is why the
    subsystems must be stored as per-step BRS rather than per-subsystem BRTs.

    Index j = 0 is t = tmax and j = -1 is t = 0, so the horizons at or below
    t_j are exactly the indices k >= j, and a reverse cumulative min realises
    the union over time.

    Args:
        states : (N, 10) array in the DeepReach state order
                 (px, vx, theta, q, py, vy, phi, p, pz, vz)
        v_*_all: the per-step BRS arrays as saved by main()
        c      : CFG

    Returns:
        (N, T) array — the BRT at every time index. Slice [:, 0] for the full
        horizon. Query points outside a subsystem grid are extrapolated
        linearly by the interpolator; they are outside the reporting window by
        construction, so treat them as unreliable.
    """
    try:
        from scipy.interpolate import RegularGridInterpolator
    except ImportError as exc:                                # pragma: no cover
        raise ImportError(
            "reconstruct_brt_pointwise needs scipy (it is in environment.yml). "
            "The solve itself does not."
        ) from exc

    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 10:
        raise ValueError(f"expected states of shape (N, 10), got {states.shape}")

    subsystems = [
        (v_x_all, [0, 1, 2, 3],
         [c["px_min"], c["vx_min"], c["th_min"], c["q_min"]],
         [c["px_max"], c["vx_max"], c["th_max"], c["q_max"]],
         [c["npx"], c["nvx"], c["nth"], c["nq"]]),
        (v_y_all, [4, 5, 6, 7],
         [c["py_min"], c["vy_min"], c["ph_min"], c["p_min"]],
         [c["py_max"], c["vy_max"], c["ph_max"], c["p_max"]],
         [c["npy"], c["nvy"], c["nph"], c["np"]]),
        (v_z_all, [8, 9],
         [c["pz_min"], c["vz_min"]],
         [c["pz_max"], c["vz_max"]],
         [c["npz"], c["nvz"]]),
    ]

    v_max = None
    for arr, idx, lo, hi, npts in subsystems:
        axes = tuple(np.linspace(lo[i], hi[i], npts[i]) for i in range(len(npts)))
        # Trailing time axis is carried through as a value vector, so one call
        # returns every time step at once: (N, T).
        interp = RegularGridInterpolator(
            axes, arr, method="linear", bounds_error=False, fill_value=None)
        vi = interp(states[:, idx])
        v_max = vi if v_max is None else np.maximum(v_max, vi)

    # min over k >= j  ==  reverse cumulative min along the time axis
    return np.minimum.accumulate(v_max[:, ::-1], axis=1)[:, ::-1]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def mirror_symmetry_report(v_x_all, v_y_all, peel=3):
    """Compare Vy(py,vy,phi,p) against Vx(py,vy,-phi,-p) — they must be equal.

    The Y dynamics map onto the X dynamics under (phi, p) -> (-phi, -p), and
    both the target box and the control interval are symmetric, so with
    identical grids and bounds the two value functions are exact mirrors. The
    grids ARE symmetric about 0 with an odd point count, so reversing the last
    two spatial axes realises that reflection exactly on grid nodes.

    In exact arithmetic. The DISCRETISATION does not reproduce that mirror,
    for two compounding reasons (both verified, see the precision note in CFG):

      1. odp's ENO tie-break `if |D2_a| <= |D2_b|` resolves exact ties toward
         whichever candidate is written first. Mirroring swaps the two, so on
         ties with D2_a = -D2_b the two solves take different stencils. A box
         target is piecewise linear, so this fires on many cells and injects
         an O(1e-3) asymmetry in the very first substep — at any precision.
      2. odp closes the non-periodic boundaries with ghost-cell extrapolation
         (V_ghost = V_edge + |dV|*sign(V), see second_orderENO4D.py). Here the
         domain is crossed very fast — |q| <= 1.5 rad/s traverses the whole
         +-0.25 rad theta window in 0.33 s — so nearly every characteristic
         exits through a boundary, and that extrapolation compounds over the
         CFL substeps into a growing outflow-corner layer that amplifies the
         seed from (1).

    So a large number here is a statement about the boundary layer, not about
    the X/Y dynamics: the X subsystem alone violates its own symmetry
    V(-px,-vx,-theta,-q) = V(px,vx,theta,q) by the same order of magnitude.

    So this returns several numbers instead:
      max            — global, dominated by the boundary layer
      interior_max   — after peeling `peel` cells off all 4 spatial dims
      mean           — bulk agreement
      sign_mismatch  — fraction of cells where the two disagree on sign(V).
                       This is the one that matters: the BRT only depends on
                       the zero level set, so a small value here means the
                       boundary-layer noise does not move the reachable set.

    A LARGE interior_max together with a large sign_mismatch would mean one of
    the two dynamics classes or grids is genuinely wrong. To tell that apart
    from the boundary-layer artefact, check the X-only self-symmetry
    |Vx - Vx[::-1,::-1,::-1,::-1]| first: if it is the same order as this
    number, the mirror check is measuring the scheme, not the dynamics.

    Computed one time step at a time to avoid materialising a second array the
    size of the inputs.
    """
    if v_x_all.shape != v_y_all.shape:
        return None

    interior = (slice(peel, -peel),) * 4
    max_err = 0.0
    interior_max = 0.0
    sum_err = np.float64(0.0)
    n_cells = np.int64(0)
    n_sign = np.int64(0)

    for i in range(v_x_all.shape[-1]):
        vy_t = v_y_all[..., i]
        vx_t = v_x_all[:, :, ::-1, ::-1, i]     # the mirrored X value
        err = np.abs(vy_t - vx_t)
        max_err = max(max_err, float(err.max()))
        interior_max = max(interior_max, float(err[interior].max()))
        sum_err += err.sum(dtype=np.float64)
        n_cells += err.size
        n_sign += int(((vy_t < 0) != (vx_t < 0)).sum())

    return {
        "max": max_err,
        "interior_max": interior_max,
        "interior_peel_cells": peel,
        "mean": float(sum_err / n_cells),
        "sign_mismatch": float(n_sign / n_cells),
    }


def self_symmetry_report(v_all):
    """Control for mirror_symmetry_report: the X subsystem against ITSELF.

    The X dynamics are odd — px'=vx, vx'=g*theta, theta'=q, q'=u with a
    symmetric control interval — and the target box is symmetric, so
    Vx(-px,-vx,-theta,-q) = Vx(px,vx,theta,q) exactly, in one single solve.

    This shares the grids, the dynamics object and the compiled kernel with
    that solve, so anything it reports is purely the numerical scheme. Compare
    it against the X/Y mirror number: if they are the same order, the mirror
    check is not telling you anything about the X-vs-Y dynamics.

    Streamed per time step, like mirror_symmetry_report.
    """
    flip = (slice(None, None, -1),) * 4
    max_err = 0.0
    peak_v = 0.0
    for i in range(v_all.shape[-1]):
        v_t = v_all[..., i]
        max_err = max(max_err, float(np.abs(v_t - v_t[flip]).max()))
        peak_v = max(peak_v, float(np.abs(v_t).max()))
    return {"max": max_err, "peak_abs_v": peak_v}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    c = CFG

    # graph_4D/graph_2D branch on these two strings only; any other value
    # leaves the spatial derivatives uncomputed and the "solve" returns the
    # target function unchanged, with no error raised.
    if c["accuracy"] not in ("low", "medium"):
        raise ValueError(
            f"accuracy={c['accuracy']!r} is not implemented by odp's compute "
            f"graphs (only 'low' and 'medium' are); it would silently produce "
            f"a frozen value function."
        )

    tau = np.arange(
        start=0,
        stop=c["tmax"] + c["small_number"],
        step=c["dt"],
    )
    T = len(tau)
    print(f"Time steps T = {T}  (tmax={c['tmax']} s, dt={c['dt']} s)")

    # -- X subsystem ---------------------------------------------------------
    print("\nSolving X subsystem  [px, vx, theta, q]  (per-step BRS) ...")
    t0 = time.time()
    v_x_all = solve_x(c, tau)
    t_x = time.time() - t0
    print(f"  shape : {v_x_all.shape}   time : {t_x:.1f}s")
    print(f"  BRS volume at tmax (V<0): {(v_x_all[..., 0] < 0).mean():.4f}")

    # -- Y subsystem ---------------------------------------------------------
    print("\nSolving Y subsystem  [py, vy, phi, p]  (per-step BRS) ...")
    t0 = time.time()
    v_y_all = solve_y(c, tau)
    t_y = time.time() - t0
    print(f"  shape : {v_y_all.shape}   time : {t_y:.1f}s")
    print(f"  BRS volume at tmax (V<0): {(v_y_all[..., 0] < 0).mean():.4f}")

    # -- Z subsystem ---------------------------------------------------------
    print("\nSolving Z subsystem  [pz, vz]  (per-step BRS) ...")
    t0 = time.time()
    v_z_all = solve_z(c, tau)
    t_z = time.time() - t0
    print(f"  shape : {v_z_all.shape}   time : {t_z:.1f}s")
    print(f"  BRS volume at tmax (V<0): {(v_z_all[..., 0] < 0).mean():.4f}")

    # -- Sanity check: a-priori value bounds ---------------------------------
    # No clamp is active any more, so this is a real check rather than a
    # tautology: the lower bound is the global min of l, and the upper bound is
    # the closed-form free-flow envelope. See free_flow_upper_bound() for why
    # the v2 reference value (max l over the grid) does not apply to a BRS.
    r = c["target_radius"]
    w = c["report_window"]
    specs = {
        "x": make_spec(
            [c["px_min"], c["vx_min"], c["th_min"], c["q_min"]],
            [c["px_max"], c["vx_max"], c["th_max"], c["q_max"]],
            [c["npx"], c["nvx"], c["nth"], c["nq"]],
            w["x"], _flow_pos_vel_ang, +c["gravity"]),
        "y": make_spec(
            [c["py_min"], c["vy_min"], c["ph_min"], c["p_min"]],
            [c["py_max"], c["vy_max"], c["ph_max"], c["p_max"]],
            [c["npy"], c["nvy"], c["nph"], c["np"]],
            w["y"], _flow_pos_vel_ang, -c["gravity"]),
        "z": make_spec(
            [c["pz_min"], c["vz_min"]],
            [c["pz_max"], c["vz_max"]],
            [c["npz"], c["nvz"]],
            w["z"], _flow_double_integrator, 0.0),
    }
    bounds = {
        "x": value_bounds_report(v_x_all, specs["x"], tau, r, c["u_max"]),
        "y": value_bounds_report(v_y_all, specs["y"], tau, r, c["u_max"]),
        "z": value_bounds_report(v_z_all, specs["z"], tau, r, c["u_max"]),
    }
    print("\nValue bounds — V must lie in [-r, free-flow envelope]; anything "
          "outside is numerical error:")
    for k, b in bounds.items():
        print(f"  {k}: full grid   V in [{b['v_min']:+.4f}, {b['v_max']:+.4f}]"
              f"   envelope max {b['envelope_max']:+.4f}"
              f"   over by {b['over_envelope']:.4f}"
              f"   {'ok' if b['within_bounds'] else 'OUT OF BOUNDS'}")
        print(f"     report window V in [{b['v_min_window']:+.4f}, "
              f"{b['v_max_window']:+.4f}]"
              f"   over by {b['over_envelope_window']:.4f}"
              f"   {'ok' if b['within_bounds_window'] else 'OUT OF BOUNDS'}"
              + (f"   (worst at t = {b['worst_over_window_t']:.2f}s)"
                 if b["worst_over_window_t"] is not None else ""))
    print("  The window row is the one that decides whether the grid padding is "
          "enough: it asks whether the boundary layer reached the region "
          "DeepReach actually reads.")

    # -- Verdict: how far out in horizon is this solve actually usable? -------
    # Removing the v2 clamp buys the correct operand order and costs the bound
    # the clamp was providing. The boundary layer is now visible, it grows with
    # horizon, and padding the grid does not cure it (measured: doubling the
    # theta/q extents AND the px/vx extents, 14.5x the cells, only moves the
    # worst in-window violation from ~12 to ~4.7). So the honest output is a
    # horizon up to which the values are trustworthy, not a global pass/fail.
    trusted = min(b["trusted_horizon"] for b in bounds.values())
    worst = min(bounds, key=lambda k: bounds[k]["trusted_horizon"])
    print(f"\nTrusted horizon: t <= {trusted:.2f}s  (of tmax = {c['tmax']}s; "
          f"limited by the {worst.upper()} subsystem, at "
          f"{bounds[worst]['trusted_horizon_rel_tol']*100:.0f}% of the envelope)")
    if trusted < c["tmax"] - 1e-9:
        print(f"  WARNING: beyond t = {trusted:.2f}s the in-window value "
              f"MAGNITUDES are unreliable. The zero level set degrades more "
              f"slowly than the magnitudes, so the BRT may still be usable — "
              f"but do not report V itself past this horizon without checking.")
        print(f"  This is the cost of the correct operand order: v2's "
              f"min(V, l) clamp bounded the magnitudes at the price of "
              f"computing max_i min_s instead of min_s max_i.")

    # -- Sanity check: X/Y mirror symmetry -----------------------------------
    # Read this together with the X self-symmetry below: the mirror number is
    # only evidence about the dynamics to the extent that it EXCEEDS what one
    # subsystem already does to itself.
    self_sym = self_symmetry_report(v_x_all)
    print("\nX self-symmetry  |Vx(x) - Vx(-x)|   (one solve, exact in theory):")
    print(f"  max           = {self_sym['max']:.3e}")
    print(f"  peak |Vx|     = {self_sym['peak_abs_v']:.3e}   "
          f"(free-flow envelope maxes out at {bounds['x']['envelope_max']:.2f})")

    mirror = mirror_symmetry_report(v_x_all, v_y_all)
    if mirror is None:
        print("\nX/Y mirror check skipped (grids differ)")
    else:
        print("\nX/Y mirror check  |Vy(phi,p) - Vx(-phi,-p)|:")
        print(f"  max           = {mirror['max']:.3e}   "
              f"(global; when large it is set by one outflow-corner cell)")
        print(f"  interior max  = {mirror['interior_max']:.3e}   "
              f"({mirror['interior_peel_cells']} cells peeled off each spatial dim)")
        print(f"  mean          = {mirror['mean']:.3e}")
        print(f"  sign mismatch = {mirror['sign_mismatch']*100:.4f}% of cells   "
              f"<- the one that matters; the BRS/BRT only sees sign(V)")
        if mirror["max"] <= 4.0 * self_sym["max"]:
            print("  => same order as the X self-symmetry: this is the ENO/boundary "
                  "scheme, not an X-vs-Y dynamics error.")

    # -- Save subsystem arrays (no full 10D grid — see module docstring) -----
    # HJSolver computes in float32 but hands back float64: its saveAllTimeSteps
    # buffer is a plain np.zeros(...). So the arrays below are float64-typed
    # single-precision values, and the on-disk dtype is decided here and nowhere
    # else — adding .astype(np.float32) would halve the files to ~690 MB without
    # losing anything the solver actually computed.
    for fname, arr in [("v_sub_x_brs.npy", v_x_all),
                       ("v_sub_y_brs.npy", v_y_all),
                       ("v_sub_z_brs.npy", v_z_all)]:
        p = os.path.join(out_dir, fname)
        np.save(p, arr)
        print(f"Saved {fname:18s} shape={str(arr.shape):28s} "
              f"{str(arr.dtype):8s} {arr.nbytes/1e6:.1f} MB  -> {p}")

    # -- Manifest ------------------------------------------------------------
    # Paths are absolute: the manifest is consumed from a different directory
    # than the one holding the arrays (see odp_10d.sh).
    root = os.path.abspath(out_dir)
    manifest = {
        # version 3: arrays switched BACK from per-subsystem BRT
        # ("minVWithV0", v2) to per-step BRS (TargetSetMode="none"), because v2
        # reconstructed in the wrong operand order. Keys/filenames go from
        # v_sub_*_brt to v_sub_*_brs. A v2 consumer will not find its keys —
        # that is intentional: it would otherwise keep taking the max alone and
        # silently return a BRS where it used to get a BRT.
        "version": 3,
        "root": root,
        "system": "QuadrotorHover10D",
        "control_config": "shared_l2_control",
        "value_semantics": "per_step_brs",
        "target_set_mode": c["target_set_mode"],
        "report_window": c["report_window"],
        "report_window_note": "DeepReach state_test_range. The grid overhangs "
                              "it so the ghost-cell boundary layer has somewhere "
                              "to sit outside the reported region; q and p are "
                              "NOT padded.",
        "values": {
            "v_sub_x_brs": {
                "path": os.path.join(root, "v_sub_x_brs.npy"),
                "shape": list(v_x_all.shape),
                "dtype": str(v_x_all.dtype),
                "axes": ["px", "vx", "theta", "q", "time"],
                "state_idx": [0, 1, 2, 3],
                "grid_min": [c["px_min"], c["vx_min"], c["th_min"], c["q_min"]],
                "grid_max": [c["px_max"], c["vx_max"], c["th_max"], c["q_max"]],
                "note": "X subsystem BRS at each time step (no clamping)",
            },
            "v_sub_y_brs": {
                "path": os.path.join(root, "v_sub_y_brs.npy"),
                "shape": list(v_y_all.shape),
                "dtype": str(v_y_all.dtype),
                "axes": ["py", "vy", "phi", "p", "time"],
                "state_idx": [4, 5, 6, 7],
                "grid_min": [c["py_min"], c["vy_min"], c["ph_min"], c["p_min"]],
                "grid_max": [c["py_max"], c["vy_max"], c["ph_max"], c["p_max"]],
                "note": "Y subsystem BRS at each time step (no clamping)",
            },
            "v_sub_z_brs": {
                "path": os.path.join(root, "v_sub_z_brs.npy"),
                "shape": list(v_z_all.shape),
                "dtype": str(v_z_all.dtype),
                "axes": ["pz", "vz", "time"],
                "state_idx": [8, 9],
                "grid_min": [c["pz_min"], c["vz_min"]],
                "grid_max": [c["pz_max"], c["vz_max"]],
                "note": "Z subsystem BRS at each time step (no clamping)",
            },
        },
        "time": {
            "tmax": c["tmax"],
            "dt": c["dt"],
            "num_steps": T,
            "index_0": "t = tmax",
            "index_-1": "t = 0 (target function)",
            "note": "This is odp's native order (HJSolver writes the target to "
                    "valfuncs[..., -1] and fills backwards). Reference arrays "
                    "circulated as Vx_time.npy / Vy_time.npy are REVERSED "
                    "(index 0 = t = 0); reverse the last axis before comparing.",
        },
        "reconstruction": {
            "full_grid_materialized": False,
            "reason": "full 10D grid is 41^4 * 41^4 * 81^2 ~= 5.2e16 cells per "
                      "time step; evaluate on demand instead",
            "method": "Proposition 1 + 4, Chen et al. 2018 (max over subsystems per "
                      "time step, THEN min over time)",
            "formula_brs": "V10D_BRS(x, t_j) = max(interp(v_sub_x_brs, x[0:4], j), "
                           "interp(v_sub_y_brs, x[4:8], j), "
                           "interp(v_sub_z_brs, x[8:10], j))",
            "formula_brt": "V10D_BRT(x, t_j) = min over k >= j of V10D_BRS(x, t_k)",
            "formula_note": "The max comes FIRST, per time step; the min over time is "
                            "applied to the reconstructed 10D value, NOT inside each "
                            "subsystem. Taking the max alone (the v2 recipe) now yields "
                            "the BRS at a single t, not the BRT. Index 0 = tmax and "
                            "index -1 = t = 0, so the horizons at or below t_j are the "
                            "indices k >= j; a reverse cumulative min realises the union "
                            "over time. Reference implementation: "
                            "reconstruct_brt_pointwise() in the generating script.",
            "exact": False,
            "exactness_note": "ONE over-approximation remains, down from two in v2. "
                              "(1) shared_l2_control: each subsystem is solved over the "
                              "projected interval [-u_max, u_max], whose product box "
                              "strictly contains the L2 ball, so the reconstruction "
                              "over-estimates the BRT of the true coupled system and "
                              "lower-bounds its value. This is irreducible by "
                              "decomposition and is the gap the joint DeepReach model "
                              "exists to close. "
                              "(2) ordering: FIXED in v3. The arrays are per-step BRS and "
                              "the formula above computes min_s max_i, which is the exact "
                              "Prop 1 + 4 reconstruction for the relaxed (box-control) "
                              "system. v2 stored per-subsystem BRTs and computed "
                              "max_i min_s <= min_s max_i, an extra over-approximation.",
            "brs_available": True,
            "brs_note": "These arrays ARE the per-step BRS. Because no min(V, l) clamp is "
                        "applied, the ghost-cell boundary layer is visible rather than "
                        "masked -- see validation.value_bounds, which scores the solve "
                        "against a closed-form free-flow envelope separately inside and "
                        "outside the reporting window.",
        },
        "validation": {
            "value_bounds": bounds,
            "trusted_horizon": trusted,
            "trusted_horizon_note": "Longest horizon for which the in-window "
                                    "violation of the free-flow envelope stays under "
                                    "10% of that envelope, minimised over subsystems. "
                                    "Past it, treat in-window V MAGNITUDES as "
                                    "unreliable; the zero level set degrades more "
                                    "slowly. Grid padding does not cure this -- "
                                    "measured, 14.5x the cells moved the worst "
                                    "in-window violation only from ~12 to ~4.7 -- "
                                    "because over this horizon the characteristics "
                                    "cross the whole domain, so the truncated-domain "
                                    "BRS genuinely depends on data off the grid.",
            "value_bounds_identity": "-r <= V(z, t) <= min over constant u of "
                                     "l(Phi_t^u(z)) for every t",
            "value_bounds_note": "With no clamp active this is a real check, not a "
                                 "tautology. Lower bound: l = max_i|z_i| - r >= -r "
                                 "globally. Upper bound: the true BRS minimises over all "
                                 "controls, so ANY admissible control gives a valid "
                                 "upper bound; the envelope sweeps constant controls, for "
                                 "which these integrator chains have closed-form flow. "
                                 "NOTE the v2 bound (max l over the grid = 5.8) is valid "
                                 "for a BRT, where V(x,t) <= l(x) by staying put, but NOT "
                                 "for a BRS, whose trajectory leaves the grid before the "
                                 "terminal cost is read -- so '328 vs 5.8' overstated the "
                                 "divergence. The *_window fields are the ones that "
                                 "matter: they say whether the boundary layer reached the "
                                 "region DeepReach reads.",
            "xy_mirror": mirror,
            "xy_mirror_identity": "Vy(py,vy,phi,p,t) == Vx(py,vy,-phi,-p,t)",
            "x_self_symmetry": self_sym,
            "x_self_symmetry_identity": "Vx(px,vx,theta,q,t) == Vx(-px,-vx,-theta,-q,t)",
            "interpretation_note": "x_self_symmetry is the control: it comes from a "
                                   "SINGLE solve, so it measures the scheme alone. "
                                   "xy_mirror is only evidence about the X-vs-Y dynamics "
                                   "to the extent that it exceeds x_self_symmetry.",
            "scheme_note": "Two discretisation artefacts, neither fixable with more "
                           "float precision. (1) odp's ENO tie-break "
                           "'if |D2_a| <= |D2_b|' resolves exact ties toward the first "
                           "candidate; mirroring swaps the candidates, so ties with "
                           "D2_a = -D2_b take different stencils. A box target is "
                           "piecewise linear, so these are common and inject ~1e-3 "
                           "asymmetry in the first substep. (2) odp closes non-periodic "
                           "boundaries by ghost-cell extrapolation; with |q| <= 1.5 rad/s "
                           "crossing the +-0.25 rad theta window in 0.33 s, almost every "
                           "characteristic exits the domain, and the extrapolation grows "
                           "an outflow-corner layer that amplifies (1). Treat V "
                           "MAGNITUDES anywhere near a domain face as unreliable; the "
                           "zero level set (and hence the BRT) is much less affected -- "
                           "see xy_mirror.sign_mismatch.",
        },
        "timing": {
            "x_seconds": round(t_x, 2),
            "y_seconds": round(t_y, 2),
            "z_seconds": round(t_z, 2),
        },
    }
    manifest_path = os.path.join(out_dir, "artifact_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved artifact_manifest.json  -> {manifest_path}")

    metrics = {"config": c, "artifact_manifest": "artifact_manifest.json"}
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics.json            -> {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run decomposed QuadrotorHover10D per-step BRS solver "
                    "(X 4D + Y 4D + Z 2D; no full 10D grid). Reconstruct the "
                    "BRT downstream as min-over-time of max-over-subsystems."
    )
    parser.add_argument(
        "--out_dir",
        default=DEFAULT_OUT_DIR,
        help=f"Directory to write outputs (default: {DEFAULT_OUT_DIR})",
    )
    args = parser.parse_args()

    main(args.out_dir)


# ---------------------------------------------------------------------------
# Reference dynamics (DeepReach joint 10D formulation, kept for record)
# ---------------------------------------------------------------------------
# Source: cmpt720_hybrid_hj/dynamics/dynamics.py :: QuadrotorHover10D
# Solved JOINTLY under the coupled constraint ux^2+uy^2+uz^2 <= u_max^2, which
# is why its Hamiltonian uses -u_max*||c|| rather than per-axis bang-bang. The
# three subsystem classes above use per-axis bang-bang because each is solved
# over an interval, not the ball — that difference IS the leaking corner.
#
# class QuadrotorHover10D(Dynamics):
#     def __init__(self, u_max: float = 2.0, target_radius: float = 0.2,
#                  gravity: float = 9.81):
#         self.u_max = u_max
#         self.target_radius = target_radius
#         self.gravity = gravity
#         super().__init__(
#             loss_type='brt_hjivi', set_mode='reach',
#             state_dim=10, input_dim=11, control_dim=3, disturbance_dim=0,
#             state_mean=[0.0]*10,
#             state_var=[2.0, 1.5, 0.25, 1.5, 2.0, 1.5, 0.25, 1.5, 2.0, 1.5],
#             value_mean=0.0, value_var=1.0, value_normto=0.02,
#             deepreach_model="exact",
#         )
#
#     def state_test_range(self):
#         return [
#             [-2.0, 2.0], [-1.5, 1.5], [-0.25, 0.25], [-1.5, 1.5],  # px,vx,theta,q
#             [-2.0, 2.0], [-1.5, 1.5], [-0.25, 0.25], [-1.5, 1.5],  # py,vy,phi,p
#             [-2.0, 2.0], [-1.5, 1.5],                              # pz,vz
#         ]
#
#     def equivalent_wrapped_state(self, state):
#         return torch.clone(state)   # no periodic dims (small-angle theta/phi)
#
#     def dsdt(self, state, control, disturbance):
#         dsdt = torch.zeros_like(state)
#         dsdt[..., 0] = state[..., 1]
#         dsdt[..., 1] = self.gravity * state[..., 2]
#         dsdt[..., 2] = state[..., 3]
#         dsdt[..., 3] = control[..., 0]
#         dsdt[..., 4] = state[..., 5]
#         dsdt[..., 5] = -self.gravity * state[..., 6]
#         dsdt[..., 6] = state[..., 7]
#         dsdt[..., 7] = control[..., 1]
#         dsdt[..., 8] = state[..., 9]
#         dsdt[..., 9] = control[..., 2]
#         return dsdt
#
#     def boundary_fn(self, state):
#         return torch.amax(torch.abs(state) - self.target_radius, dim=-1)
#
#     def cost_fn(self, state_traj):
#         return torch.min(self.boundary_fn(state_traj), dim=-1).values
#
#     def hamiltonian(self, state, dvds):
#         drift = dvds[..., 0]*state[..., 1] + self.gravity*dvds[..., 1]*state[..., 2] \
#               + dvds[..., 2]*state[..., 3] \
#               + dvds[..., 4]*state[..., 5] - self.gravity*dvds[..., 5]*state[..., 6] \
#               + dvds[..., 6]*state[..., 7] \
#               + dvds[..., 8]*state[..., 9]
#         c = torch.stack((dvds[..., 3], dvds[..., 7], dvds[..., 9]), dim=-1)
#         ctrl = -self.u_max * torch.norm(c, dim=-1)
#         return drift + ctrl
#
#     def optimal_control(self, state, dvds):
#         c = torch.stack((dvds[..., 3], dvds[..., 7], dvds[..., 9]), dim=-1)
#         c_norm = torch.norm(c, dim=-1, keepdim=True).clamp_min(1e-8)
#         return -self.u_max * c / c_norm
#
#     def optimal_disturbance(self, state, dvds):
#         return 0
