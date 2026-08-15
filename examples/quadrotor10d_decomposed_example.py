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

What removing the v2 clamp costs
--------------------------------
The clamp bounded |V| by max l over the grid, and that bound does not survive
its removal: V_BRS(x, t) = min_u l(x(0)) is the terminal cost at a state the
trajectory has actually REACHED, and over t = 3 s these trajectories leave the
grid by a wide margin, so V is not bounded above by anything the grid knows.
(The bound V(x,t) <= l(x) is valid for a BRT — stay put — which is why the
clamped v2 solve satisfied it by construction.) The only a-priori bound left is
the global lower one, V(z,t) >= min l = -r, since l = max_i|z_i| - r >= -r
everywhere.

Treat V magnitudes near a domain face as unreliable: odp closes non-periodic
boundaries with ghost-cell extrapolation, and switching the clamp off exposes
that layer rather than removing it. The layer grows with horizon and padding
the grid does not cure it (measured: doubling the theta/q AND px/vx extents,
14.5x the cells, only reduced the in-window overshoot from ~12 to ~4.7),
because over this horizon the characteristics cross the whole domain, so the
truncated-domain BRS genuinely depends on data off the grid. The zero level set
— and hence the BRT — degrades far more slowly than the magnitudes do.

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
    # padded — they are the dims where that layer reaches the reported region
    # first. Recorded in the manifest; nothing in this script masks by it.
    "report_window": {
        "x": [[-2.0, 2.0], [-1.5, 1.5], [-0.25, 0.25], [-1.5, 1.5]],
        "y": [[-2.0, 2.0], [-1.5, 1.5], [-0.25, 0.25], [-1.5, 1.5]],
        "z": [[-2.0, 2.0], [-1.5, 1.5]],
    },
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
