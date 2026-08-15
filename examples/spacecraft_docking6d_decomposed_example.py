"""
Example: Decomposed BRAT for SpacecraftDocking6D
(Clohessy-Wiltshire translation + single-axis rotation)
Reference: Thorup et al., arXiv:2605.02021, 2026.

Grid dimensions, goal/avoid sets, and all parameters match the paper's
gridBased6DImplementation (Docking4D.py + Docking2D.py).

Solves the BRAT/BRS independently for each subsystem:
  - SpacecraftDocking6DTrans : state = [px, py, vx, vy]   (4D, CW dynamics)
      BRAT: must reach pos/vel goal while avoiding body + post obstacles.
  - SpacecraftDocking6DRot   : state = [theta, omega]     (2D, pure rotation)
      BRS: must reach attitude/rate goal; no obstacle in rotation state.

BRAT mechanics in ODP (solver.py):
  Pass  multiple_value = [reach_target, avoid_constraint]  to HJSolver.
  - init_value = max(reach_target, -avoid_constraint)      [boundary_fn]
  - compMethod["ObstacleSetMode"] = "maxVWithObstacle"
    → clamps V = max(V, -avoid_constraint) at every time step
  - compMethod["TargetSetMode"]   = "none"
    → no running-min clamping; solver returns pure BRAS at each saved time step

INDEPENDENCE NOTE:
  Translation [px, py, vx, vy] uses [Fx, Fy]; rotation [theta, omega] uses [tau].
  No shared state or control → full 6D BRAT reconstruction is EXACT (Prop 1 + Prop 4,
  Chen et al. 2018). The reconstruct_brat_6d helper below is correct but infeasible
  at this resolution (full 6D array would be ~83 TB); evaluate V6D on demand as
  max(interp(V_trans, x_trans, t), interp(V_rot, x_rot, t)).

Only the two subsystem value functions are written to disk — the full 6D BRAT is
never materialised, and no |V_trans - V_rot| gap array is produced.

Output (written to --out_dir, default <repo root>/output_SpacecraftDocking6D_decomposed/,
alongside the other output_* dirs and covered by .gitignore's /output_* rule):
  v_trans_brs.npy  — translation BRAS at all time steps, (npx, npy, nvx, nvy, T) ~1.6 GB
  v_rot_brs.npy    — rotation    BRS  at all time steps, (nth, nom, T)            ~21 MB
  artifact_manifest.json / metrics.json
"""

import argparse
import json
import math
import os
import time

import numpy as np

from odp.Grid import Grid
from odp.dynamics import SpacecraftDocking6DTrans, SpacecraftDocking6DRot
from odp.solver import HJSolver


# ---------------------------------------------------------------------------
# Pre-computed constants — match Docking4D.py / Docking2D.py in paper code
# ---------------------------------------------------------------------------
_MU        = 3.986004418e14          # Earth gravitational parameter [m^3/s^2]
_R_EARTH   = 6371e3                  # Earth mean radius [m]
_ORBIT_ALT = 400e3                   # Orbital altitude [m]
_N = math.sqrt(_MU / (_R_EARTH + _ORBIT_ALT) ** 3)   # mean motion ≈ 0.001133 rad/s

_W_C = 1.0                           # chaser width  [m]
_H_C = 1.0                           # chaser height [m]
_RC  = math.sqrt(_W_C**2 + _H_C**2) / 2   # bounding-circle radius = sqrt(2)/2 ≈ 0.7071 m

# Default output lives at the repo root, not next to this file, so the path does
# not depend on the directory the script happens to be launched from.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT_DIR = os.path.join(_REPO_ROOT, "output_SpacecraftDocking6D_decomposed")

_POST_LENGTH      = 0.2              # docking post length [m]
_GOAL_CLEARANCE   = 0.143           # goal top is 7 mm inside inflated post boundary [m]
_GOAL_BAND_HEIGHT = 0.2              # goal band height [m]
_GOAL_Y_MAX = -(_POST_LENGTH + _RC + _GOAL_CLEARANCE)   # ≈ -0.900 m
_GOAL_Y_MIN = _GOAL_Y_MAX - _GOAL_BAND_HEIGHT           # = -1.400 m

# ---------------------------------------------------------------------------
# Configuration — matches Docking4D.py + Docking2D.py in paper code exactly
# ---------------------------------------------------------------------------
CFG = {
    # time — matches DeepReach training horizon
    "tmax": 10.0,
    "dt": 0.1,
    "small_number": 1e-5,

    # grids — dense resolution; only the subsystem arrays are stored
    # v_trans_brs shape: (91, 101, 21, 21, T) × 4 bytes ≈ 1.6 GB   (T=101)
    # v_rot_brs   shape: (361, 141, T)        × 4 bytes ≈ 21 MB
    "px_min": -15.0, "px_max": 15.0,
    "py_min": -15.0, "py_max": 15.0,
    "vx_min":  -1.5, "vx_max":  1.5,
    "vy_min":  -2.0, "vy_max":  2.0,
    "npx": 91, "npy": 101, "nvx": 21, "nvy": 21,

    "th_min": -math.pi, "th_max": math.pi,
    "om_min":  -1.0,    "om_max":  1.0,
    "nth": 361, "nom": 141,

    # spacecraft parameters — match paper code
    "n": _N,             # mean motion [rad/s], computed from 400 km LEO
    "m": 200.0,          # chaser mass [kg]
    "I": 200.0 / 6.0,   # moment of inertia [kg*m^2] = 1/12 * m * (w_c^2 + h_c^2)

    # control bounds
    "Fx_min": -20.0, "Fx_max": 20.0,    # [N]
    "Fy_min": -20.0, "Fy_max": 20.0,    # [N]
    "tau_min": -1.5, "tau_max":  1.5,   # [N*m]

    # goal tolerances — match paper code
    "eps_p":  0.1,           # position  [m]
    "eps_v":  0.1,           # velocity  [m/s]
    "eps_th": 0.04,          # attitude  [rad]
    "eps_w":  0.05,          # angular rate  [rad/s]
    "theta_goal": math.pi / 2,

    # chaser bounding-circle radius — matches Docking4D.py: sqrt(w_c^2+h_c^2)/2
    "rc": _RC,

    # target body: 6×3 m rectangle, y ∈ [0, h_t], x ∈ [-w_t/2, w_t/2]
    "w_t": 6.0,
    "h_t": 3.0,

    # docking post: y ∈ [-post_length, 0], x ∈ [-post_hw_x, post_hw_x]
    "post_hw_x":   0.6,
    "post_length": _POST_LENGTH,

    # goal band — matches Docking4D.py exactly
    #   goal_y_max ≈ -0.900 m (7 mm inside inflated post; masked by avoid constraint)
    #   goal_y_min = -1.400 m
    "goal_y_max": _GOAL_Y_MAX,
    "goal_y_min": _GOAL_Y_MIN,

    # solver — "none" → no running-min clamping → pure BRAS per step (needed for decomp)
    "accuracy": "medium",
    "target_set_mode": "none",
}


# ---------------------------------------------------------------------------
# Level-set shape helpers (computed on ODP grid.vs arrays via broadcasting)
# ---------------------------------------------------------------------------
#
# DeepReach-matching value shaping
# --------------------------------
# DeepReach's SpacecraftDocking6D.reach_fn / avoid_fn do NOT use raw signed
# distances. Each component distance d is reshaped by a piecewise map
#   inside  tolerance (d <  0): d * <steep_gain>        (amplify)
#   outside tolerance (d >= 0): tanh(d * <tanh_scale>)  (saturate)
# the reach value is the max over components times _REACH_SCALE (1.2), and the
# obstacle distance is amplified 1.5x inside the obstacle. Applying the SAME
# shaping here keeps this ground-truth grid's value MAGNITUDES comparable to
# the DeepReach model (both share the same zero level set either way).
#
# The max-decomposition V6D = max(V_trans, V_rot) stays EXACT under this
# reshaping: each per-component map is monotonic, max distributes over the
# trans/rot split, and the positive scalars 1.2 / 1.5 commute with max. So the
# Chen et al. 2018 (Prop 1+4) hypotheses — decoupled dynamics + max-decomposed
# terminal cost — are unchanged.

_REACH_SCALE = 1.2   # DeepReach reach_fn final multiplier


def _shape_dist(d, steep_gain, tanh_scale):
    """DeepReach per-component reshaping: amplify inside, tanh-saturate outside."""
    return np.where(d < 0, d * steep_gain, np.tanh(d * tanh_scale))


def trans_reach_fn(g, c):
    """Translation reach set level-function on 4D grid [px, py, vx, vy].

    g(z_trans) ≤ 0  iff  |px| ≤ eps_p  AND  py ∈ [goal_y_min, goal_y_max]
                          AND  ||[vx,vy]|| ≤ eps_v

    Sign convention: level ≤ 0 → in set. Component distances are reshaped to
    match DeepReach reach_fn (tanh-saturated outside, amplified inside, ×1.2).
    """
    px = g.vs[0]   # (npx,  1,   1,   1)
    py = g.vs[1]   # ( 1,  npy,  1,   1)
    vx = g.vs[2]   # ( 1,   1,  nvx,  1)
    vy = g.vs[3]   # ( 1,   1,   1,  nvy)

    # Lateral position band: |px| ≤ eps_p
    px_dist = np.abs(px) - c["eps_p"]

    # Longitudinal goal band: py ∈ [goal_y_min, goal_y_max]
    py_dist = np.maximum(c["goal_y_min"] - py, py - c["goal_y_max"])

    pos_dist = np.maximum(px_dist, py_dist)     # (npx, npy, 1, 1) via broadcast

    # Velocity: L2 norm ≤ eps_v
    vel_dist = np.sqrt(vx**2 + vy**2 + 1e-8) - c["eps_v"]   # (1, 1, nvx, nvy)

    # DeepReach reshaping (reach_fn): pos *20 / tanh(*0.5), vel *20 / tanh(*1.0)
    pos_dist = _shape_dist(pos_dist, 20.0, 0.5)
    vel_dist = _shape_dist(vel_dist, 20.0, 1.0)

    # Reach: must satisfy BOTH position AND velocity; ×1.2 (see rot_reach_fn note)
    reach = np.maximum(pos_dist, vel_dist) * _REACH_SCALE
    return np.broadcast_to(reach, tuple(g.pts_each_dim)).copy().astype(np.float64)


def avoid_fn(g, c):
    """Obstacle (avoid) level-function on 4D grid [px, py, vx, vy].

    l(z_trans) > 0  iff  outside both inflated body and post (SAFE).
    l(z_trans) ≤ 0  iff  inside at least one obstacle (UNSAFE).

    Passed as the 'constraint' to HJSolver; solver enforces V = max(V, -l)
    at every time step, which keeps obstacle-interior states infeasible.
    """
    px = g.vs[0]   # (npx, 1, 1, 1)
    py = g.vs[1]   # ( 1, npy, 1, 1)
    cb = c["rc"]

    # Target body: y ∈ [0, h_t], x ∈ [-w_t/2, w_t/2], inflated by cb
    # s_body > 0 ↔ outside inflated body (safe)
    s_body = np.maximum(
        np.abs(px) - (c["w_t"] / 2.0 + cb),
        np.maximum(-(py + cb), py - (c["h_t"] + cb)),
    )

    # Docking post: y ∈ [-post_length, 0], x ∈ [-post_hw_x, post_hw_x], inflated by cb
    # s_post > 0 ↔ outside inflated post (safe)
    s_post = np.maximum(
        np.abs(px) - (c["post_hw_x"] + cb),
        np.maximum(-(py + c["post_length"] + cb), py - cb),
    )

    # Union of obstacles: l > 0 ↔ outside BOTH (safe)
    s_fail = np.minimum(s_body, s_post)     # (npx, npy, 1, 1)

    # DeepReach reshaping (avoid_fn): amplify 1.5x inside obstacle, raw outside
    s_fail = np.where(s_fail < 0, s_fail * 1.5, s_fail)

    return np.broadcast_to(s_fail, tuple(g.pts_each_dim)).copy().astype(np.float64)


def rot_reach_fn(g, c):
    """Rotation reach set level-function on 2D grid [theta, omega].

    h(z_rot) ≤ 0  iff  |theta - theta_goal| ≤ eps_th  AND  |omega| ≤ eps_w
    """
    theta = g.vs[0]   # (nth, 1)
    omega = g.vs[1]   # ( 1, nom)

    # Wrapped angle error to theta_goal = pi/2
    theta_err = np.abs(
        np.arctan2(
            np.sin(theta - c["theta_goal"]),
            np.cos(theta - c["theta_goal"]),
        )
    )
    theta_dist = theta_err - c["eps_th"]

    omega_dist = np.abs(omega) - c["eps_w"]

    # DeepReach reshaping (reach_fn): theta *150 / tanh(*1.0), omega *30 / tanh(*1.0).
    # ×1.2 is applied to BOTH subsystems so the reconstruction
    #   max(V_trans, V_rot) = 1.2 * max(pos, vel, theta, rate)
    # reproduces DeepReach's reach_fn (which multiplies the 4-way max by 1.2).
    theta_dist = _shape_dist(theta_dist, 150.0, 1.0)
    omega_dist = _shape_dist(omega_dist, 30.0, 1.0)

    reach = np.maximum(theta_dist, omega_dist) * _REACH_SCALE
    return np.broadcast_to(reach, tuple(g.pts_each_dim)).copy().astype(np.float64)


# ---------------------------------------------------------------------------
# Subsystem solvers
# ---------------------------------------------------------------------------

def solve_trans(c, tau):
    """Solve translation subsystem BRAS at every time step.

    State = [px, py, vx, vy].
    This is a BRAT (reach-avoid) problem:
      - Reach: pos/vel docking goal (trans_reach_fn ≤ 0)
      - Avoid: target body + post obstacles (avoid_fn ≤ 0 → unsafe)

    ODP BRAT:
      initial value = max(reach_target, -avoid_constraint)   [boundary_fn]
      at each step  V = max(V, -avoid_constraint)            [obstacle enforcement]

    Returns array of shape (npx, npy, nvx, nvy, T).
    Index -1 (last)  = t = 0     (initial boundary_fn).
    Index  0 (first) = t = tmax  (full BRAS backward from tmax).
    """
    g = Grid(
        np.array([c["px_min"], c["py_min"], c["vx_min"], c["vy_min"]]),
        np.array([c["px_max"], c["py_max"], c["vx_max"], c["vy_max"]]),
        4,
        np.array([c["npx"], c["npy"], c["nvx"], c["nvy"]]),
        [],  # no periodic dimensions
    )

    reach_target    = trans_reach_fn(g, c)   # (npx, npy, nvx, nvy)
    avoid_constr    = avoid_fn(g, c)         # (npx, npy, nvx, nvy)

    dyn = SpacecraftDocking6DTrans(
        uMin=[c["Fx_min"], c["Fy_min"]],
        uMax=[c["Fx_max"], c["Fy_max"]],
        uMode="min",
        n=c["n"],
        m=c["m"],
    )

    return HJSolver(
        dyn, g,
        [reach_target, avoid_constr],       # BRAT: [goal, safe-set]
        tau,
        {
            "TargetSetMode":   c["target_set_mode"],   # "none" → pure BRAS per step
            "ObstacleSetMode": "maxVWithObstacle",     # V = max(V, -l) at every step
        },
        saveAllTimeSteps=True,
        accuracy=c["accuracy"],
    )   # shape (npx, npy, nvx, nvy, T)


def solve_rot(c, tau):
    """Solve rotation subsystem BRS at every time step.

    State = [theta, omega].
    Pure BRS (no obstacle in rotation state — obstacle only depends on px, py).

    Returns array of shape (nth, nom, T).
    """
    g = Grid(
        np.array([c["th_min"], c["om_min"]]),
        np.array([c["th_max"], c["om_max"]]),
        2,
        np.array([c["nth"], c["nom"]]),
        [0],  # theta (dim 0) is periodic
    )

    reach_target = rot_reach_fn(g, c)   # (nth, nom)

    dyn = SpacecraftDocking6DRot(
        uMin=[c["tau_min"]],
        uMax=[c["tau_max"]],
        uMode="min",
        I=c["I"],
    )

    return HJSolver(
        dyn, g,
        reach_target,                           # BRS only (no obstacle)
        tau,
        {"TargetSetMode": c["target_set_mode"]},
        saveAllTimeSteps=True,
        accuracy=c["accuracy"],
    )   # shape (nth, nom, T)


# ---------------------------------------------------------------------------
# Reconstruction (reference only — never called at this resolution)
# ---------------------------------------------------------------------------

def reconstruct_brat_6d(v_trans_all, v_rot_all):
    """Reconstruct full 6D BRAT at every time step (Proposition 4, Chen et al. 2018).

    The two subsystems are independent (no shared state, no shared control),
    so this reconstruction is EXACT.

    At each time step s:
      BRAS_6D(s) = proj^{-1}(BRAS_trans(s)) ∩ proj^{-1}(BRS_rot(s))
                 = max( V_trans_6D(s), V_rot_6D(s) )

    BRAT_6D accumulated over time:
      BRAT_6D(s) = union_{r<=s} BRAS_6D(r)
                 = min_{r<=s}  max(V_trans_6D(r), V_rot_6D(r))   [level-set union]

    Time axis convention (same as direct solver):
      index -1 (last)  → t = 0     (initial boundary_fn / target set)
      index  0 (first) → t = tmax  (full BRAT)

    Args:
        v_trans_all : (npx, npy, nvx, nvy, T)  — translation BRAS at each step
        v_rot_all   : (nth, nom, T)             — rotation    BRS  at each step

    Returns:
        (npx, npy, nvx, nvy, nth, nom, T) — full 6D BRAT at every time step
    """
    T                    = v_trans_all.shape[-1]
    npx, npy, nvx, nvy  = v_trans_all.shape[:4]
    nth, nom             = v_rot_all.shape[:2]

    brat_all    = np.zeros((npx, npy, nvx, nvy, nth, nom, T), dtype=np.float32)
    running_min = np.full((npx, npy, nvx, nvy, nth, nom), fill_value=np.inf, dtype=np.float32)

    for i in range(T - 1, -1, -1):
        # Lift each subsystem value to 6D via broadcasting
        brs_t = v_trans_all[..., i][:, :, :, :, np.newaxis, np.newaxis]  # (npx,npy,nvx,nvy,1,1)
        brs_r = v_rot_all[..., i][np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :]  # (1,1,1,1,nth,nom)

        bras_6d     = np.maximum(brs_t, brs_r)           # intersection in level-set
        running_min = np.minimum(running_min, bras_6d)   # union over time
        brat_all[..., i] = running_min

    return brat_all   # (npx, npy, nvx, nvy, nth, nom, T)


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def main(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    c = CFG

    tau = np.arange(
        start=0,
        stop=c["tmax"] + c["small_number"],
        step=c["dt"],
    )
    T = len(tau)
    print(f"Time steps T = {T}  (tmax={c['tmax']} s, dt={c['dt']} s)")

    # -- Translation subsystem (BRAT) ----------------------------------------
    print("\nSolving translation subsystem  [px, py, vx, vy]  (BRAT) ...")
    t0 = time.time()
    v_trans_all = solve_trans(c, tau)
    t_trans = time.time() - t0
    print(f"  shape : {v_trans_all.shape}   time : {t_trans:.1f}s")
    print(f"  BRAS volume at tmax (V<0): {(v_trans_all[..., 0] < 0).mean():.4f}")

    # -- Rotation subsystem (BRS) --------------------------------------------
    print("\nSolving rotation subsystem  [theta, omega]  (BRS) ...")
    t0 = time.time()
    v_rot_all = solve_rot(c, tau)
    t_rot = time.time() - t0
    print(f"  shape : {v_rot_all.shape}   time : {t_rot:.1f}s")
    print(f"  BRS  volume at tmax (V<0): {(v_rot_all[..., 0] < 0).mean():.4f}")

    # -- Save subsystem arrays (small — fits in RAM) -------------------------
    v_trans_f32 = v_trans_all.astype(np.float32)
    v_rot_f32   = v_rot_all.astype(np.float32)
    for fname, arr in [("v_trans_brs.npy", v_trans_f32), ("v_rot_brs.npy", v_rot_f32)]:
        p = os.path.join(out_dir, fname)
        np.save(p, arr)
        print(f"Saved {fname:30s} shape={str(arr.shape):30s} {arr.nbytes/1e6:.1f} MB  → {p}")

    # -- Manifest ------------------------------------------------------------
    # Paths are absolute: the manifest is consumed from a different directory
    # than the one holding the arrays (see odp_6d.sh).
    root = os.path.abspath(out_dir)
    manifest = {
        "version": 1,
        "root": root,
        "values": {
            "v_trans_brs": {
                "path": os.path.join(root, "v_trans_brs.npy"),
                "shape": list(v_trans_f32.shape),
                "axes": ["px", "py", "vx", "vy", "time"],
                "note": "translation BRAS at each time step (obstacle-enforced, no running-min clamping)",
            },
            "v_rot_brs": {
                "path": os.path.join(root, "v_rot_brs.npy"),
                "shape": list(v_rot_f32.shape),
                "axes": ["theta", "omega", "time"],
                "note": "rotation BRS at each time step (no obstacle, no running-min clamping)",
            },
        },
        "reconstruction": {
            "method": "Proposition 1 + 4, Chen et al. 2018 (exact for independent subsystems)",
            "formula": "V6D(x,t) = max(V_trans(x[:4],t), V_rot(x[4:],t))",
            "note": "evaluated on demand by consumers; the full 6D array is never stored",
        },
        "timing": {
            "trans_seconds": round(t_trans, 2),
            "rot_seconds":   round(t_rot,   2),
        },
    }
    manifest_path = os.path.join(out_dir, "artifact_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved artifact_manifest.json  → {manifest_path}")

    metrics = {"config": c, "artifact_manifest": "artifact_manifest.json"}
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics.json            → {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run decomposed SpacecraftDocking6D BRAT solver "
                    "(translation BRAT 4D + rotation BRS 2D)"
    )
    parser.add_argument(
        "--out_dir",
        default=DEFAULT_OUT_DIR,
        help=f"Directory to write outputs (default: {DEFAULT_OUT_DIR})",
    )
    args = parser.parse_args()

    main(args.out_dir)


# ---------------------------------------------------------------------------
# Reference dynamics (DeepReach / Thorup et al. formulation, kept for record)
# ---------------------------------------------------------------------------

# class SpacecraftDocking6D(Dynamics):
#     """
#     6D Planar Spacecraft Docking: Clohessy-Wiltshire + single-axis rotation.
#
#     State:   [px, py, vx, vy, theta, omega]
#     Control: [Fx, Fy, tau]   (bang-bang)
#
#     Dynamics (LVLH frame):
#       px_dot    = vx
#       py_dot    = vy
#       vx_dot    = 3n^2*px + 2n*vy + Fx/m
#       vy_dot    = -2n*vx  + Fy/m
#       theta_dot = omega
#       omega_dot = tau/I
#
#     Target body modelled as a rectangle centred at (0, 1.5) m with
#     half-extents (3, 1.5) m.  Docking port extends below at (0, -0.5)
#     with half-extents (0.25, 0.5) m.  Both are inflated by rc = 0.707 m.
#     Goal: reach (0, -1.5) m with theta_g = pi/2 within paper tolerances.
#
#     Reference: Thorup et al., arXiv:2605.02021, 2026.
