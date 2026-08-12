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

    V_10D(x, t) = max( Vx(x[0:4], t), Vy(x[4:8], t), Vz(x[8:10], t) )

on demand by interpolating each subsystem array at the projected state.

BRT, NOT BRS — and the ordering caveat that comes with it
---------------------------------------------------------
Each subsystem is solved with TargetSetMode="minVWithV0", so every step is
clamped as min(V_evolved, l) against the target function. The saved arrays are
therefore per-subsystem BRTs (running-min), NOT the per-time-step BRS.

That is a deliberate choice, and it is NOT the ordering the decomposition
theory wants. The max-decomposition (Chen et al. 2018, Prop. 1 + 4) holds per
time step, and the union-over-time does not commute with the
intersection-over-subsystems:

    EXACT      :  V_BRT(x, t) = min_{s <= t} max_i V_i^BRS(x_i, s)
    WHAT WE DO :  V_BRT(x, t) = max_i min_{s <= t} V_i(x_i, s) = max_i V_i^BRT

Since max_i min_s (...) <= min_s max_i (...), reconstructing from these arrays
UNDER-estimates the value and so OVER-estimates the BRT. The two agree only if
each subsystem can wait inside its own box until the others arrive, and the box
is not control-invariant here: from the vx = 0.2 corner, px overshoots to ~0.28
before theta can decelerate it. So this is a second over-approximation stacked
on top of the shared_l2_control one below.

Why clamp anyway: with TargetSetMode="none" the pure BRS is numerically
unusable at this resolution. Its true range is [min l, max l] = [-0.2, 5.8],
but odp's ghost-cell extrapolation at the outflow faces drives max |V| to 328
by t = 3 s, and the exact minimum -0.2 drifts to -0.182. Doubling the theta and
q extents (same dx) cuts that to 10.4 in the original window at ~10x the
runtime — the boundary is confirmed as the driver, but it is still not a clean
BRS. The clamp bounds the result exactly ([-0.2, 5.8] at every step, origin
re-pinned to -0.2) at the cost of the ordering above. It masks the divergence
rather than fixing it: treat V magnitudes near a domain face as unreliable.

Note the clamp is LOSSY — the per-step BRS cannot be recovered from these
arrays, so the exact ordering is not available downstream. Switch
CFG["target_set_mode"] back to "none" (and fix the divergence) if you need it.

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
  v_sub_x_brt.npy  — X BRT at all time steps, (npx, nvx, nth, nq, T)   ~1.4 GB
  v_sub_y_brt.npy  — Y BRT at all time steps, (npy, nvy, nph, np_, T)  ~1.4 GB
  v_sub_z_brt.npy  — Z BRT at all time steps, (npz, nvz, T)            ~3.2 MB
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
    # v_sub_x_brt / v_sub_y_brt: 41^4 x T x 8 bytes ~= 1.4 GB each (T = 61)
    # v_sub_z_brt:               81^2 x T x 8 bytes ~= 3.2 MB
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

    # "minVWithV0" -> clamp min(V, l) every step -> per-subsystem BRT, bounded
    # to [-0.2, 5.8]. "none" would give the per-step BRS the decomposition
    # theory actually wants, but it diverges to max |V| = 328 here. Read the
    # "BRT, NOT BRS" section of the module docstring before changing this: the
    # two choices are not interchangeable, and this one over-estimates the BRT.
    "target_set_mode": "minVWithV0",
    "u_mode": "min",     # reach

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


def value_bounds_report(v_all, l_min, l_max, tol=1e-5):
    """Check V against its a-priori range [min l, max l].

    Both the BRS (min_u l(x(t))) and the BRT (min over s of that) are values OF
    the terminal cost along some trajectory, so neither can leave the range of
    l itself. Anything outside is numerical error, full stop — which makes this
    the cheapest honest health check on the solve.

    It is what separates the two failure modes seen here: with
    TargetSetMode="minVWithV0" the clamp pins the range to exactly [min l,
    max l], while with "none" odp's ghost-cell extrapolation at the outflow
    faces pushes max |V| to ~328 against a true max of 5.8. Note that PASSING
    this check under the clamp proves very little — the clamp enforces the
    bound by construction and hides whatever the evolution did underneath.
    """
    vmin, vmax = np.inf, -np.inf
    for i in range(v_all.shape[-1]):
        v_t = v_all[..., i]
        vmin = min(vmin, float(v_t.min()))
        vmax = max(vmax, float(v_t.max()))
    return {
        "min": vmin,
        "max": vmax,
        "l_min": float(l_min),
        "l_max": float(l_max),
        "within_bounds": bool(vmin >= l_min - tol and vmax <= l_max + tol),
        "overshoot": max(0.0, vmax - l_max),
        "undershoot": max(0.0, l_min - vmin),
    }


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
    print("\nSolving X subsystem  [px, vx, theta, q]  (BRT) ...")
    t0 = time.time()
    v_x_all = solve_x(c, tau)
    t_x = time.time() - t0
    print(f"  shape : {v_x_all.shape}   time : {t_x:.1f}s")
    print(f"  BRT volume at tmax (V<0): {(v_x_all[..., 0] < 0).mean():.4f}")

    # -- Y subsystem ---------------------------------------------------------
    print("\nSolving Y subsystem  [py, vy, phi, p]  (BRT) ...")
    t0 = time.time()
    v_y_all = solve_y(c, tau)
    t_y = time.time() - t0
    print(f"  shape : {v_y_all.shape}   time : {t_y:.1f}s")
    print(f"  BRT volume at tmax (V<0): {(v_y_all[..., 0] < 0).mean():.4f}")

    # -- Z subsystem ---------------------------------------------------------
    print("\nSolving Z subsystem  [pz, vz]  (BRT) ...")
    t0 = time.time()
    v_z_all = solve_z(c, tau)
    t_z = time.time() - t0
    print(f"  shape : {v_z_all.shape}   time : {t_z:.1f}s")
    print(f"  BRT volume at tmax (V<0): {(v_z_all[..., 0] < 0).mean():.4f}")

    # -- Sanity check: a-priori value bounds ---------------------------------
    # ShapeRectangle is max_i(|x_i| - r), so over the grid it runs from -r (at
    # the origin, which is a node since every axis is symmetric with an odd
    # point count) up to (largest half-extent) - r.
    l0_max = max(c["px_max"], c["vx_max"], c["th_max"], c["q_max"]) - c["target_radius"]
    l0_min = -c["target_radius"]
    bounds = {
        "x": value_bounds_report(v_x_all, l0_min, l0_max),
        "y": value_bounds_report(v_y_all, l0_min, l0_max),
        "z": value_bounds_report(
            v_z_all, l0_min,
            max(c["pz_max"], c["vz_max"]) - c["target_radius"]),
    }
    print(f"\nValue bounds (V must stay within the range of l — anything outside "
          f"is pure numerical error):")
    for k, b in bounds.items():
        flag = "ok" if b["within_bounds"] else "OUT OF BOUNDS"
        print(f"  {k}: [{b['min']:+.4f}, {b['max']:+.4f}]  vs l = "
              f"[{b['l_min']:+.4f}, {b['l_max']:+.4f}]   {flag}")
    if all(b["within_bounds"] for b in bounds.values()):
        print("  (bounded BY CONSTRUCTION under TargetSetMode="
              f"{c['target_set_mode']!r} — the clamp enforces this, so it says "
              "nothing about the quality of the evolution underneath)")

    # -- Sanity check: X/Y mirror symmetry -----------------------------------
    # Read this together with the X self-symmetry below: the mirror number is
    # only evidence about the dynamics to the extent that it EXCEEDS what one
    # subsystem already does to itself.
    self_sym = self_symmetry_report(v_x_all)
    print("\nX self-symmetry  |Vx(x) - Vx(-x)|   (one solve, exact in theory):")
    print(f"  max           = {self_sym['max']:.3e}")
    print(f"  peak |Vx|     = {self_sym['peak_abs_v']:.3e}   "
          f"(target function maxes out at {l0_max:.2f})")

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
              f"<- the one that matters; the BRT only sees sign(V)")
        if mirror["max"] <= 4.0 * self_sym["max"]:
            print("  => same order as the X self-symmetry: this is the ENO/boundary "
                  "scheme, not an X-vs-Y dynamics error.")

    # -- Save subsystem arrays (no full 10D grid — see module docstring) -----
    # HJSolver computes in float32 but hands back float64: its saveAllTimeSteps
    # buffer is a plain np.zeros(...). So the arrays below are float64-typed
    # single-precision values, and the on-disk dtype is decided here and nowhere
    # else — adding .astype(np.float32) would halve the files to ~690 MB without
    # losing anything the solver actually computed.
    for fname, arr in [("v_sub_x_brt.npy", v_x_all),
                       ("v_sub_y_brt.npy", v_y_all),
                       ("v_sub_z_brt.npy", v_z_all)]:
        p = os.path.join(out_dir, fname)
        np.save(p, arr)
        print(f"Saved {fname:18s} shape={str(arr.shape):28s} "
              f"{str(arr.dtype):8s} {arr.nbytes/1e6:.1f} MB  -> {p}")

    # -- Manifest ------------------------------------------------------------
    # Paths are absolute: the manifest is consumed from a different directory
    # than the one holding the arrays (see odp_10d.sh).
    root = os.path.abspath(out_dir)
    manifest = {
        # version 2: arrays switched from per-step BRS (TargetSetMode="none")
        # to per-subsystem BRT ("minVWithV0"), and the keys/filenames went from
        # v_sub_*_brs to v_sub_*_brt. A consumer written against version 1 will
        # not find the old keys — that is intentional, because the CONTENTS
        # changed meaning and silently reusing the old names would hide it.
        "version": 2,
        "root": root,
        "system": "QuadrotorHover10D",
        "control_config": "shared_l2_control",
        "value_semantics": "per_subsystem_brt",
        "target_set_mode": c["target_set_mode"],
        "values": {
            "v_sub_x_brt": {
                "path": os.path.join(root, "v_sub_x_brt.npy"),
                "shape": list(v_x_all.shape),
                "dtype": str(v_x_all.dtype),
                "axes": ["px", "vx", "theta", "q", "time"],
                "state_idx": [0, 1, 2, 3],
                "grid_min": [c["px_min"], c["vx_min"], c["th_min"], c["q_min"]],
                "grid_max": [c["px_max"], c["vx_max"], c["th_max"], c["q_max"]],
                "note": "X subsystem BRT at each time step (min(V, l) clamped every step)",
            },
            "v_sub_y_brt": {
                "path": os.path.join(root, "v_sub_y_brt.npy"),
                "shape": list(v_y_all.shape),
                "dtype": str(v_y_all.dtype),
                "axes": ["py", "vy", "phi", "p", "time"],
                "state_idx": [4, 5, 6, 7],
                "grid_min": [c["py_min"], c["vy_min"], c["ph_min"], c["p_min"]],
                "grid_max": [c["py_max"], c["vy_max"], c["ph_max"], c["p_max"]],
                "note": "Y subsystem BRT at each time step (min(V, l) clamped every step)",
            },
            "v_sub_z_brt": {
                "path": os.path.join(root, "v_sub_z_brt.npy"),
                "shape": list(v_z_all.shape),
                "dtype": str(v_z_all.dtype),
                "axes": ["pz", "vz", "time"],
                "state_idx": [8, 9],
                "grid_min": [c["pz_min"], c["vz_min"]],
                "grid_max": [c["pz_max"], c["vz_max"]],
                "note": "Z subsystem BRT at each time step (min(V, l) clamped every step)",
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
            "method": "Proposition 1 + 4, Chen et al. 2018 (max over subsystems, per time step)",
            "formula_brt": "V10D_BRT(x, t) = max(interp(v_sub_x_brt, x[0:4], t), "
                           "interp(v_sub_y_brt, x[4:8], t), "
                           "interp(v_sub_z_brt, x[8:10], t))",
            "formula_note": "Take the max ONLY -- do NOT apply a further min over "
                            "time. The min-over-time is already baked into each "
                            "array by TargetSetMode='minVWithV0'.",
            "exact": False,
            "exactness_note": "TWO stacked over-approximations of the BRT. "
                              "(1) shared_l2_control: each subsystem uses the projected "
                              "interval [-u_max, u_max], whose product box strictly "
                              "contains the L2 ball. "
                              "(2) ordering: these arrays are per-subsystem BRTs, so the "
                              "reconstruction computes max_i min_s V_i, whereas the exact "
                              "BRT is min_s max_i V_i^BRS. Since max_i min_s <= min_s max_i, "
                              "the result lower-bounds the true coupled value. Equality "
                              "would require each subsystem to be able to wait inside its "
                              "own target box, and that box is not control-invariant here.",
            "brs_available": False,
            "brs_note": "The per-step BRS needed for the exact ordering is NOT recoverable "
                        "from these arrays -- the min(V, l) clamp is lossy. Re-run with "
                        "TargetSetMode='none' to get it, but note that mode diverges to "
                        "max|V| = 328 (true range [-0.2, 5.8]) at this resolution.",
        },
        "validation": {
            "value_bounds": bounds,
            "value_bounds_identity": "min(l) <= V(x, t) <= max(l) for every t",
            "value_bounds_note": "Satisfied by construction under "
                                 "TargetSetMode='minVWithV0'; it does NOT certify the "
                                 "evolution. Under 'none' the same solve reaches "
                                 "max|V| = 328 against a true max of 5.8.",
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
        description="Run decomposed QuadrotorHover10D BRT solver "
                    "(X 4D + Y 4D + Z 2D; no full 10D grid)"
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
