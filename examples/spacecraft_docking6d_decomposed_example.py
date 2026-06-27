"""
Example: Decomposed BRAT for SpacecraftDocking6D
(Clohessy-Wiltshire translation + single-axis rotation)
Reference: Thorup et al., arXiv:2605.02021, 2026.

Goal: reach the docking port approach corridor with correct attitude while
avoiding the target body (6×3 m) and docking post (both inflated by rc=0.707 m).

Solves the BRAT/BRS at every time step independently for each subsystem:
  - SpacecraftDocking6DTrans : state = [px, py, vx, vy]   (4D, CW dynamics)
      BRAT: must reach pos/vel goal while avoiding body + post obstacles.
  - SpacecraftDocking6DRot   : state = [theta, omega]     (2D, pure rotation)
      BRS: must reach attitude/rate goal; no obstacle in rotation state.

Then reconstructs the full 6D BRAT at every time step:

  BRAS_6D(s) = proj^{-1}(BRAS_trans(s)) ∩ proj^{-1}(BRS_rot(s))
             = max( V_trans_6D(s), V_rot_6D(s) )   [intersection in level-set]

  BRAT_6D    = union_{s} BRAS_6D(s)
             = min_{s}   max(V_trans_6D(s), V_rot_6D(s))

BRAT mechanics in ODP (solver.py):
  Pass  multiple_value = [reach_target, avoid_constraint]  to HJSolver.
  - init_value = max(reach_target, -avoid_constraint)      [boundary_fn]
  - compMethod["ObstacleSetMode"] = "maxVWithObstacle"
    → clamps V = max(V, -avoid_constraint) at every time step
    → states inside the obstacle stay infeasible for all time
  - compMethod["TargetSetMode"]   = "none"
    → no running-min clamping; solver returns pure BRAS at each saved time step

INDEPENDENCE NOTE:
  Translation [px, py, vx, vy] uses [Fx, Fy]; rotation [theta, omega] uses [tau].
  No shared state or control → full 6D BRAT reconstruction is EXACT.
  The obstacle only depends on (px, py) so it only enters the translation subsystem.

Output (written to --out_dir, default ./output_SpacecraftDocking6D_decomposed/):
  v_trans_brs.npy         — translation BRAS at all time steps, (npx, npy, nvx, nvy, T)
  v_rot_brs.npy           — rotation    BRS  at all time steps, (nth, nom, T)
  v_brat.npy              — reconstructed full 6D BRAT,         (npx, npy, nvx, nvy, nth, nom, T)
  close_value_gap_all.npy — |V_trans_6D - V_rot_6D|,            (npx, npy, nvx, nvy, nth, nom, T)
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
# Configuration — matches Thorup et al., arXiv:2605.02021, 2026
# ---------------------------------------------------------------------------
CFG = {
    # time
    "tmax": 10.0,
    "dt": 0.1,
    "small_number": 1e-5,

    # translation grid  [px (m), py (m), vx (m/s), vy (m/s)]
    # Grid resolution is kept coarse so the 6D output arrays fit in memory:
    #   v_brat and close_value_gap_all each have shape
    #   (npx, npy, nvx, nvy, nth, nom, T).
    #   At float32: 10*10*8*8*12*8*101 * 4 bytes ≈ 248 MB per array.
    "px_min": -15.0, "px_max": 15.0,
    "py_min": -15.0, "py_max": 15.0,
    "vx_min":  -1.5, "vx_max":  1.5,
    "vy_min":  -2.0, "vy_max":  2.0,
    "npx": 10, "npy": 10, "nvx": 8, "nvy": 8,

    # rotation grid  [theta (rad), omega (rad/s)]
    "th_min": -math.pi, "th_max": math.pi,
    "om_min":  -2.0,    "om_max":  2.0,
    "nth": 12, "nom": 8,

    # spacecraft parameters (paper values)
    "n": 0.001131,       # orbital mean motion [rad/s], ~400 km LEO
    "m": 200.0,          # chaser mass [kg]
    "I": 200.0 / 6.0,   # moment of inertia [kg*m^2]

    # control bounds
    "Fx_min": -20.0, "Fx_max": 20.0,    # [N]
    "Fy_min": -20.0, "Fy_max": 20.0,    # [N]
    "tau_min": -1.5, "tau_max":  1.5,   # [N*m]

    # goal tolerances (paper values — used in DeepReach reach_fn with tanh scaling)
    "eps_p":  0.1,           # position  [m]
    "eps_v":  0.1,           # velocity  [m/s]
    "eps_th": 0.04,          # attitude  [rad]
    "eps_w":  0.05,          # angular rate  [rad/s]
    "theta_goal": math.pi / 2,

    # ODP-specific goal sizes — relaxed so that several grid points fall inside
    # the goal at t=0.  ODP's grid-based PDE needs the initial value function to
    # be negative somewhere; if the goal is smaller than the grid spacing it never
    # is.  Rule of thumb: goal radius ≥ 1–2 × grid spacing.
    #
    # DubinsCar4D3 worked because box_radius=0.8 m on a 0.1 m grid (8 cells).
    # Here the grid spacings are:
    #   px/py : 30 m / 9 gaps = 3.33 m  →  odp_px_radius = 3.5 m  (~1 cell)
    #   vx    :  3 m/s / 7 gaps = 0.43 m/s  →  odp_vxy_radius = 0.5 m/s (~1 cell)
    #   vy    :  4 m/s / 7 gaps = 0.57 m/s  →  same
    #   theta : 2π / 11 gaps = 0.57 rad  →  odp_th_radius = 0.8 rad  (~1-2 cells)
    #   omega :  4 / 7 gaps = 0.57 rad/s  →  odp_om_radius = 0.6 rad/s (~1 cell)
    #
    # The py goal band is widened to [-5, -0.5] (4.5 m, ~1-2 cells) to ensure
    # grid points fall below the inflated obstacle at py ≈ -0.9 m.
    "odp_px_radius":  3.5,    # [m]      replaces eps_p for ODP
    "odp_py_min":    -5.0,    # [m]      wide approach corridor below obstacle
    "odp_py_max":    -0.5,    # [m]
    "odp_vxy_radius": 0.5,    # [m/s]   replaces eps_v for ODP
    "odp_th_radius":  0.8,    # [rad]   replaces eps_th for ODP
    "odp_om_radius":  0.6,    # [rad/s] replaces eps_w for ODP

    # chaser bounding-circle radius (paper: rc ≈ 0.707 m)
    "rc": 0.707,

    # target body: 6×3 m rectangle, y ∈ [0, h_t], x ∈ [-w_t/2, w_t/2]
    "w_t": 6.0,
    "h_t": 3.0,

    # docking post: y ∈ [-post_length, 0], x ∈ [-post_hw_x, post_hw_x]
    "post_hw_x":   0.6,
    "post_length": 0.2,

    # goal band (paper geometry, kept for reference — not used in ODP level-sets)
    "goal_y_max": -(0.2 + 0.707 + 0.093),       # ≈ -1.0 m
    "goal_y_min": -(0.2 + 0.707 + 0.093) - 0.4, # ≈ -1.4 m

    # solver
    "accuracy": "medium",
    # "none" is unrecognised by solver post-processing → no running-min clamping
    # → solver saves pure BRS/BRAS at each time step (correct for decomposition)
    "target_set_mode": "none",
}


# ---------------------------------------------------------------------------
# Level-set shape helpers (computed on ODP grid.vs arrays via broadcasting)
# ---------------------------------------------------------------------------

def trans_reach_fn(g, c):
    """Translation reach set level-function on 4D grid [px, py, vx, vy].

    Uses ODP-specific goal sizes (odp_*) — relaxed from paper tolerances so
    that multiple grid cells fall inside the goal at t=0.

    g(z_trans) ≤ 0  iff  |px| ≤ odp_px_radius
                          AND  py ∈ [odp_py_min, odp_py_max]
                          AND  ||[vx,vy]|| ≤ odp_vxy_radius
    """
    px = g.vs[0]   # (npx,  1,   1,   1)
    py = g.vs[1]   # ( 1,  npy,  1,   1)
    vx = g.vs[2]   # ( 1,   1,  nvx,  1)
    vy = g.vs[3]   # ( 1,   1,   1,  nvy)

    px_dist  = np.abs(px) - c["odp_px_radius"]
    py_dist  = np.maximum(c["odp_py_min"] - py, py - c["odp_py_max"])
    pos_dist = np.maximum(px_dist, py_dist)

    vel_dist = np.sqrt(vx**2 + vy**2) - c["odp_vxy_radius"]

    return np.broadcast_to(
        np.maximum(pos_dist, vel_dist),
        tuple(g.pts_each_dim),
    ).copy().astype(np.float64)


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

    return np.broadcast_to(s_fail, tuple(g.pts_each_dim)).copy().astype(np.float64)


def rot_reach_fn(g, c):
    """Rotation reach set level-function on 2D grid [theta, omega].

    Uses ODP-specific goal sizes (odp_*) — relaxed from paper tolerances.

    h(z_rot) ≤ 0  iff  |theta - theta_goal| ≤ odp_th_radius
                        AND  |omega| ≤ odp_om_radius
    """
    theta = g.vs[0]   # (nth, 1)
    omega = g.vs[1]   # ( 1, nom)

    theta_err  = np.abs(
        np.arctan2(
            np.sin(theta - c["theta_goal"]),
            np.cos(theta - c["theta_goal"]),
        )
    )
    theta_dist = theta_err - c["odp_th_radius"]
    omega_dist = np.abs(omega) - c["odp_om_radius"]

    return np.broadcast_to(
        np.maximum(theta_dist, omega_dist),
        tuple(g.pts_each_dim),
    ).copy().astype(np.float64)


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
# Reconstruction and gap
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


def compute_close_value_gap(v_trans_all, v_rot_all):
    """Compute per-grid-point gap |V_trans_6D - V_rot_6D| at every time step.

    For independent subsystems this gap is NOT an approximation error.
    It indicates which subsystem is the binding constraint at each 6D point
    and is used in DeepReach as an adaptive guidance weight.

    Args:
        v_trans_all : (npx, npy, nvx, nvy, T)
        v_rot_all   : (nth, nom, T)

    Returns:
        (npx, npy, nvx, nvy, nth, nom, T) float32
    """
    vt = v_trans_all[:, :, :, :, np.newaxis, np.newaxis, :]   # (npx,npy,nvx,nvy,1,1,T)
    vr = v_rot_all[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :, :]  # (1,1,1,1,nth,nom,T)
    return np.abs(vt - vr).astype(np.float32)   # (npx,npy,nvx,nvy,nth,nom,T)


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

    # -- Reconstruct full 6D BRAT (exact, Proposition 4) --------------------
    print(f"\nReconstructing 6D BRAT over {T} time steps ...")
    t0 = time.time()
    v_brat = reconstruct_brat_6d(v_trans_all, v_rot_all)
    t_recon = time.time() - t0
    print(f"  shape : {v_brat.shape}   time : {t_recon:.1f}s")
    print(f"  BRAT volume at tmax (V<0): {(v_brat[..., 0] < 0).mean():.4f}")

    # -- Gap -----------------------------------------------------------------
    print("\nComputing close_value_gap_all  |V_trans_6D - V_rot_6D| ...")
    print("  NOTE: for independent subsystems this gap reflects constraint dominance,")
    print("        NOT approximation error.  Large gaps are expected and benign.")
    t0 = time.time()
    close_value_gap_all = compute_close_value_gap(v_trans_all, v_rot_all)
    t_gap = time.time() - t0
    print(f"  shape : {close_value_gap_all.shape}   time : {t_gap:.1f}s")
    print(f"  gap   mean={close_value_gap_all.mean():.4f}  max={close_value_gap_all.max():.4f}")

    # -- Save ----------------------------------------------------------------
    v_trans_f32 = v_trans_all.astype(np.float32)
    v_rot_f32   = v_rot_all.astype(np.float32)

    saves = {
        "v_trans_brs.npy":         v_trans_f32,
        "v_rot_brs.npy":           v_rot_f32,
        "v_brat.npy":              v_brat,
        "close_value_gap_all.npy": close_value_gap_all,
    }
    for fname, arr in saves.items():
        p = os.path.join(out_dir, fname)
        np.save(p, arr)
        mb = arr.nbytes / 1e6
        print(f"Saved {fname:30s} shape={str(arr.shape):40s} {mb:.1f} MB  → {p}")

    # -- Manifest ------------------------------------------------------------
    manifest = {
        "version": 1,
        "values": {
            "v_trans_brs": {
                "path": "v_trans_brs.npy",
                "shape": list(v_trans_f32.shape),
                "axes": ["px", "py", "vx", "vy", "time"],
                "note": "translation BRAS at each time step (obstacle-enforced, no running-min clamping)",
            },
            "v_rot_brs": {
                "path": "v_rot_brs.npy",
                "shape": list(v_rot_f32.shape),
                "axes": ["theta", "omega", "time"],
                "note": "rotation BRS at each time step (no obstacle, no running-min clamping)",
            },
            "v_brat": {
                "path": "v_brat.npy",
                "shape": list(v_brat.shape),
                "axes": ["px", "py", "vx", "vy", "theta", "omega", "time"],
                "note": "exact 6D BRAT at every time step; index 0 = full BRAT at tmax, index -1 = initial boundary_fn",
            },
            "close_value_gap_all": {
                "path": "close_value_gap_all.npy",
                "shape": list(close_value_gap_all.shape),
                "axes": ["px", "py", "vx", "vy", "theta", "omega", "time"],
                "note": (
                    "|V_trans_6D - V_rot_6D|.  "
                    "Subsystems are independent so this is NOT approximation error — "
                    "it reflects which constraint dominates.  "
                    "Used as adaptive guidance weight in DeepReach training."
                ),
            },
        },
        "reconstruction": {
            "method": "Proposition 4, Chen et al. 2018 (exact for independent subsystems)",
            "formula": "min_s max(v_trans_brs[...,s][...,None,None], v_rot_brs[...,s][None,None,None,None,:,:])",
            "time_axis_convention": "index 0 = t=tmax (full BRAT), index -1 = t=0 (boundary_fn / target set)",
            "brat_note": (
                "Translation subsystem is solved as BRAT (reach pos/vel goal while avoiding "
                "body+post obstacles, both inflated by rc=0.707 m).  "
                "Rotation subsystem is solved as pure BRS (no obstacle in [theta,omega]).  "
                "Full 6D BRAT = max(V_trans, V_rot); obstacle is embedded in V_trans."
            ),
        },
        "timing": {
            "trans_seconds":  round(t_trans,  2),
            "rot_seconds":    round(t_rot,    2),
            "recon_seconds":  round(t_recon,  2),
            "gap_seconds":    round(t_gap,    2),
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


def main_gap_only(out_dir: str):
    """Recompute close_value_gap_all.npy from already-saved subsystem arrays."""
    trans_path = os.path.join(out_dir, "v_trans_brs.npy")
    rot_path   = os.path.join(out_dir, "v_rot_brs.npy")
    print(f"Loading {trans_path}")
    v_trans_all = np.load(trans_path)
    print(f"Loading {rot_path}")
    v_rot_all   = np.load(rot_path)
    print(f"  v_trans_brs shape : {v_trans_all.shape}")
    print(f"  v_rot_brs   shape : {v_rot_all.shape}")

    print("Computing close_value_gap_all ...")
    t0 = time.time()
    close_value_gap_all = compute_close_value_gap(v_trans_all, v_rot_all)
    t_gap = time.time() - t0
    print(f"  shape : {close_value_gap_all.shape}   time : {t_gap:.1f}s")
    print(f"  gap   mean={close_value_gap_all.mean():.4f}  max={close_value_gap_all.max():.4f}")

    gap_path = os.path.join(out_dir, "close_value_gap_all.npy")
    np.save(gap_path, close_value_gap_all)
    print(f"Saved close_value_gap_all.npy  shape={close_value_gap_all.shape}  → {gap_path}")

    manifest_path = os.path.join(out_dir, "artifact_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest.setdefault("values", {})["close_value_gap_all"] = {
            "path": "close_value_gap_all.npy",
            "shape": list(close_value_gap_all.shape),
            "axes": ["px", "py", "vx", "vy", "theta", "omega", "time"],
            "note": "|V_trans_6D - V_rot_6D|; adaptive guidance weight in DeepReach training",
        }
        manifest.setdefault("timing", {})["gap_seconds"] = round(t_gap, 2)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Updated artifact_manifest.json → {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run decomposed SpacecraftDocking6D BRAT solver "
                    "(translation BRAT 4D + rotation BRS 2D)"
    )
    parser.add_argument(
        "--out_dir",
        default="output_SpacecraftDocking6D_decomposed",
        help="Directory to write outputs "
             "(default: ./output_SpacecraftDocking6D_decomposed)",
    )
    parser.add_argument(
        "--gap_only",
        action="store_true",
        help="Skip the solve; load existing subsystem arrays and recompute gap only.",
    )
    args = parser.parse_args()

    if args.gap_only:
        main_gap_only(args.out_dir)
    else:
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
#     """
#
#     def __init__(self, n: float = 0.001131, m: float = 200.0, I: float = 200/6):
#         super().__init__(
#             loss_type='brat_hjivi',
#             set_mode='reach',
#             state_dim=6,
#             input_dim=7,
#             control_dim=3,
#             disturbance_dim=0,
#             state_mean=[0.0, 0.0, 0.0, 0.0, math.pi / 2, 0.0],
#             state_var=[20.0, 20.0, 5.0, 5.0, math.pi, 2.0],
#             value_mean=0.0,
#             value_var=1.0,
#             value_normto=0.02,
#             deepreach_model="exact",
#         )
#         self.n = n
#         self.m = m
#         self.I = I
#
#         # Control bounds (paper values)
#         self.Fmax    = 20.0   # [N]
#         self.tau_max = 1.5    # [N*m]
#
#         # Goal tolerances (paper values)
#         self.eps_p  = 0.1    # position [m]
#         self.eps_v  = 0.1    # velocity [m/s]
#         self.eps_th = 0.04   # attitude [rad]
#         self.eps_w  = 0.05   # angular rate [rad/s]
#
#         self.theta_goal = math.pi / 2
#
#         # Chaser bounding-circle radius (paper: rc ≈ 0.707 m)
#         self.rc = 0.707
#
#         # Target body: 6×3 m, y ∈ [0, 3], x ∈ [-3, 3]
#         self.w_t = 6.0
#         self.h_t = 3.0
#
#         # Docking post: extends below body, y ∈ [-0.2, 0], x ∈ [-0.6, 0.6]
#         self.post_hw_x    = 0.6
#         self.post_length  = 0.2
#
#         # Goal band: just below inflated post, with clearance
#         self.goal_y_max = -(self.post_length + self.rc + 0.093)  # ≈ -1.0
#         self.goal_y_min = self.goal_y_max - 0.4                  # ≈ -1.4
#
#     # ------------------------------------------------------------------
#     # Required interface
#     # ------------------------------------------------------------------
#
#     def state_test_range(self):
#         return [
#             [-15.0, 15.0],               # px [m]
#             [-15.0, 15.0],               # py [m]
#             [-1.5,   1.5],               # vx [m/s]
#             [-2.0,   2.0],               # vy [m/s]
#             [-math.pi, math.pi],         # theta [rad]
#             [-2.0,   2.0],               # omega [rad/s]
#         ]
#
#     def equivalent_wrapped_state(self, state):
#         wrapped = torch.clone(state)
#         wrapped[..., 4] = (wrapped[..., 4] + math.pi) % (2 * math.pi) - math.pi
#         return wrapped
#
#     def dsdt(self, state, control, disturbance):
#         dsdt = torch.zeros_like(state)
#         n = self.n
#         dsdt[..., 0] = state[..., 2]
#         dsdt[..., 1] = state[..., 3]
#         dsdt[..., 2] = 3.0*n**2*state[..., 0] + 2.0*n*state[..., 3] + control[..., 0] / self.m
#         dsdt[..., 3] = -2.0*n*state[..., 2]                           + control[..., 1] / self.m
#         dsdt[..., 4] = state[..., 5]
#         dsdt[..., 5] = control[..., 2] / self.I
#         return dsdt
#
#     def reach_fn(self, state):
#         """g(x) ≤ 0 iff in docking goal set. Matches reference Docking6D formulation."""
#         px, py     = state[..., 0], state[..., 1]
#         vx, vy     = state[..., 2], state[..., 3]
#         theta, omega = state[..., 4], state[..., 5]
#
#         # Position: x near port axis + y inside goal band
#         x_dist  = torch.abs(px) - self.eps_p
#         y_lo    = torch.tensor(self.goal_y_min, device=state.device, dtype=state.dtype)
#         y_hi    = torch.tensor(self.goal_y_max, device=state.device, dtype=state.dtype)
#         y_dist  = torch.maximum(y_lo - py, py - y_hi)
#         pos_dist = torch.maximum(x_dist, y_dist)
#
#         # Velocity: L2 norm
#         vel_dist = torch.sqrt(vx**2 + vy**2 + 1e-8) - self.eps_v
#
#         # Attitude: wrapped angle error
#         theta_dist = torch.abs(torch.atan2(
#             torch.sin(theta - self.theta_goal),
#             torch.cos(theta - self.theta_goal)
#         )) - self.eps_th
#
#         # Angular rate
#         rate_dist = torch.abs(omega) - self.eps_w
#
#         # Tanh outside goal (bounds gradients), amplified inside
#         pos_dist   = torch.where(pos_dist   < 0, pos_dist   * 20,  torch.tanh(pos_dist   * 0.5))
#         vel_dist   = torch.where(vel_dist   < 0, vel_dist   * 20,  torch.tanh(vel_dist   * 1.0))
#         theta_dist = torch.where(theta_dist < 0, theta_dist * 150, torch.tanh(theta_dist * 1.0))
#         rate_dist  = torch.where(rate_dist  < 0, rate_dist  * 30,  torch.tanh(rate_dist  * 1.0))
#
#         return torch.max(
#             torch.stack([pos_dist, vel_dist, theta_dist, rate_dist], dim=-1), dim=-1
#         ).values * 1.2
#
#     def avoid_fn(self, state):
#         """ℓ(x): positive = safe, negative = inside obstacle (body or post, inflated by rc).
#
#         Uses per-axis inflation matching the reference implementation.
#         """
#         px, py = state[..., 0], state[..., 1]
#         cb = self.rc
#
#         # Body: y ∈ [0, h_t], x ∈ [-w_t/2, w_t/2], inflated by cb
#         s_body = torch.maximum(
#             torch.abs(px) - (self.w_t / 2.0 + cb),
#             torch.maximum(-(py + cb), py - (self.h_t + cb)),
#         )
#
#         # Post: y ∈ [-post_length, 0], x ∈ [-post_hw_x, post_hw_x], inflated by cb
#         s_post = torch.maximum(
#             torch.abs(px) - (self.post_hw_x + cb),
#             torch.maximum(-(py + self.post_length + cb), py - cb),
#         )
#
#         s_fail = torch.minimum(s_body, s_post)
#         return torch.where(s_fail < 0, s_fail * 1.5, s_fail)
#
#     def boundary_fn(self, state):
#         """≤ 0 iff in goal set AND outside both obstacles (BRAT terminal condition)."""
#         return torch.maximum(self.reach_fn(state), -self.avoid_fn(state))
#
#     def hamiltonian(self, state, dvds):
#         """H(x, p) = p·f_drift(x) + min_u p·B·u  (reach: min over u)."""
#         n = self.n
#         ham_drift = (
#             dvds[..., 0] * state[..., 2] +
#             dvds[..., 1] * state[..., 3] +
#             dvds[..., 2] * (3.0*n**2*state[..., 0] + 2.0*n*state[..., 3]) +
#             dvds[..., 3] * (-2.0*n*state[..., 2]) +
#             dvds[..., 4] * state[..., 5]
#         )
#         # Bang-bang minimisation: min_{|u_i|≤ū_i} p_i * u_i = -ū_i * |p_i|
#         ham_ctrl = (
#             -self.Fmax    / self.m * torch.abs(dvds[..., 2]) +
#             -self.Fmax    / self.m * torch.abs(dvds[..., 3]) +
#             -self.tau_max / self.I * torch.abs(dvds[..., 5])
#         )
#         return ham_drift + ham_ctrl
#
#     def optimal_control(self, state, dvds):
#         """u* = -ū·sign(B^T p), where B^T p = [p2/m, p3/m, p5/I]."""
#         opt_Fx  = -self.Fmax    * torch.sign(dvds[..., 2])
#         opt_Fy  = -self.Fmax    * torch.sign(dvds[..., 3])
#         opt_tau = -self.tau_max * torch.sign(dvds[..., 5])
#         return torch.stack([opt_Fx, opt_Fy, opt_tau], dim=-1)
#
#     def optimal_disturbance(self, state, dvds):
#         return 0
#
#     def sample_target_state(self, num_samples):
#         """Uniform sample inside the docking goal set (all six tolerances satisfied)."""
#         lo = torch.tensor([
#             -self.eps_p,
#             self.goal_y_min,
#             -self.eps_v,
#             -self.eps_v,
#             self.theta_goal - self.eps_th,
#             -self.eps_w,
#         ])
#         hi = torch.tensor([
#             self.eps_p,
#             self.goal_y_max,
#             self.eps_v,
#             self.eps_v,
#             self.theta_goal + self.eps_th,
#             self.eps_w,
#         ])
#         return lo + torch.rand(num_samples, self.state_dim) * (hi - lo)
#
#     def cost_fn(self, state_traj):
#         """BRAT trajectory cost: min_t max(g(x(t)), max_{k<=t} -ℓ(x(k)))."""
#         reach_values = self.reach_fn(state_traj)
#         avoid_values = self.avoid_fn(state_traj)
#         return torch.min(
#             torch.maximum(reach_values, torch.cummax(-avoid_values, dim=-1).values),
#             dim=-1,
#         ).values
#
#     def periodic_state_dims(self):
#         return (4,)  # theta is periodic
#
#     def plot_config(self):
#         return {
#             'state_slices': [0.0, -1.5, 0.0, 0.0, math.pi / 2, 0.0],
#             'state_labels': [r'$p_x$', r'$p_y$', r'$v_x$', r'$v_y$',
#                              r'$\theta$', r'$\omega$'],
#             'x_axis_idx': 0,
#             'y_axis_idx': 1,
#             'z_axis_idx': 2,
#         }
