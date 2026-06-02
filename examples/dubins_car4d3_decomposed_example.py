"""
Example: Decomposed 3D subsystems of DubinsCar4D3 (bicycle model, no disturbance)

Solves the BRS at every time step independently for each subsystem:
  - DubinsCar4D3Vx : state = [x, v, theta]
  - DubinsCar4D3Vy : state = [y, v, theta]

Then reconstructs the full 4D BRT via Proposition 4 (Chen et al. 2018):

  BRT_full = union_s [ proj^-1(BRS_Vx(s)) ∩ proj^-1(BRS_Vy(s)) ]
           = min_s     max(V_Vx_4D(s),      V_Vy_4D(s))

Why BRS per time step (not BRT per subsystem):
  Table II of the paper shows that for an intersection target with shared
  controls, the full BRT CANNOT be recovered by intersecting subsystem BRTs.
  The correct route is Proposition 4 / Table III: save BRS at each time step,
  reconstruct the 4D BRS at each step, then union over time.

  TargetSetMode "none" is unrecognised by solver.py's post-processing, so
  no running-min clamping is applied → the solver returns the pure BRS at
  each saved time step (clean HJ PDE evolution from the target set).

Grid bounds and target box match dubins_car4d3_example.py so results can
be compared directly.

Output (written to --out_dir, default ./output_DubinsCar4D3_decomposed/):
  v_vx_brs.npy   — Vx subsystem BRS at all time steps, shape (nx, nv, nth, T)
  v_vy_brs.npy   — Vy subsystem BRS at all time steps, shape (ny, nv, nth, T)
  v_brt.npy      — reconstructed full 4D BRT,           shape (nx, ny, nv, nth)
"""

import argparse
import json
import math
import os
import time

import numpy as np

from odp.Grid import Grid
from odp.Shapes import Lower_Half_Space, Upper_Half_Space, Intersection
from odp.dynamics import DubinsCar4D3Vx, DubinsCar4D3Vy
from odp.solver import HJSolver


# ---------------------------------------------------------------------------
# Configuration — identical bounds/resolution to dubins_car4d3_example.py
# ---------------------------------------------------------------------------
CFG = {
    # time
    "lookback_length": 1.5,
    "dt": 0.05,
    "small_number": 1e-5,
    # grid bounds
    "x_min": -3.0,  "x_max": 3.0,
    "y_min": -1.0,  "y_max": 4.0,
    "v_min":  0.0,  "v_max": 4.0,
    "th_min": -math.pi, "th_max": math.pi,
    # grid resolution
    "nx": 60, "ny": 60, "nv": 20, "nth": 36,
    # target box  T = T1 ∩ T2
    #   T1 = { |x - x_c| <= rx }   (used by Vx subsystem)
    #   T2 = { |y - y_c| <= ry }   (used by Vy subsystem)
    "x_center": 0.0, "y_center": 2.0,
    "box_radius_x": 0.8, "box_radius_y": 0.8,
    # control bounds
    "umin_a": -1.5,            "umax_a": 1.5,
    "umin_w": -math.pi / 18,   "umax_w":  math.pi / 18,
    # solver — "none" is unrecognised, so no running-min clamping → pure BRS
    "accuracy": "medium",
    "target_set_mode": "none",
}


def solve_vx(c, tau):
    """Solve Vx subsystem BRS at every time step. State = [x, v, theta].

    Returns array of shape (nx, nv, nth, T).
    valfuncs[..., -1] = initial target (t=0)
    valfuncs[..., 0]  = BRS at t = lookback_length
    """
    g = Grid(
        np.array([c["x_min"], c["v_min"], c["th_min"]]),
        np.array([c["x_max"], c["v_max"], c["th_max"]]),
        3,
        np.array([c["nx"], c["nv"], c["nth"]]),
        [2],   # theta (dim 2) is periodic
    )

    # T1 : |x - x_c| <= rx  (v and theta unconstrained)
    target = Intersection(
        Lower_Half_Space(g, 0, c["x_center"] + c["box_radius_x"]),
        Upper_Half_Space(g, 0, c["x_center"] - c["box_radius_x"]),
    )

    dyn = DubinsCar4D3Vx(
        uMin=[c["umin_a"], c["umin_w"]],
        uMax=[c["umax_a"], c["umax_w"]],
        uMode="max",
    )

    return HJSolver(
        dyn, g, target, tau,
        {"TargetSetMode": c["target_set_mode"]},
        saveAllTimeSteps=True,
        accuracy=c["accuracy"],
    )   # shape (nx, nv, nth, T)


def solve_vy(c, tau):
    """Solve Vy subsystem BRS at every time step. State = [y, v, theta].

    Returns array of shape (ny, nv, nth, T).
    """
    g = Grid(
        np.array([c["y_min"], c["v_min"], c["th_min"]]),
        np.array([c["y_max"], c["v_max"], c["th_max"]]),
        3,
        np.array([c["ny"], c["nv"], c["nth"]]),
        [2],   # theta (dim 2) is periodic
    )

    # T2 : |y - y_c| <= ry  (v and theta unconstrained)
    target = Intersection(
        Lower_Half_Space(g, 0, c["y_center"] + c["box_radius_y"]),
        Upper_Half_Space(g, 0, c["y_center"] - c["box_radius_y"]),
    )

    dyn = DubinsCar4D3Vy(
        uMin=[c["umin_a"], c["umin_w"]],
        uMax=[c["umax_a"], c["umax_w"]],
        uMode="max",
    )

    return HJSolver(
        dyn, g, target, tau,
        {"TargetSetMode": c["target_set_mode"]},
        saveAllTimeSteps=True,
        accuracy=c["accuracy"],
    )   # shape (ny, nv, nth, T)


def reconstruct_brt_4d(v_vx_all, v_vy_all):
    """Reconstruct full 4D BRT at every time step from subsystem BRS arrays
    (Proposition 4, Chen et al. 2018).

    At each time step s:
      BRS_full(s) = proj^-1(BRS_Vx(s)) ∩ proj^-1(BRS_Vy(s))
                  = max(V_Vx_4D(s), V_Vy_4D(s))     [intersection in level-set]

    BRT_full(s) = union_{r=0}^{s} BRS_full(r)
               = min_{r=0}^{s} BRS_full(r)           [union in level-set]

    Time axis convention (matches the direct solver and dubins_car4d3_example.py):
      index -1  →  t = 0                (initial target set)
      index  0  →  t = lookback_length  (full BRT)

    Iterates from i=T-1 (t=0) down to i=0 (t=lookback_length), accumulating a
    running minimum so brt_all[..., i] = BRT up to the backward time at step i.

    Args:
        v_vx_all : (nx, nv, nth, T) — BRS of Vx subsystem at each time step
        v_vy_all : (ny, nv, nth, T) — BRS of Vy subsystem at each time step

    Returns:
        (nx, ny, nv, nth, T) — full 4D BRT at every time step
    """
    T   = v_vx_all.shape[-1]
    nx  = v_vx_all.shape[0]
    ny  = v_vy_all.shape[0]
    nv  = v_vx_all.shape[1]
    nth = v_vx_all.shape[2]

    brt_all     = np.zeros((nx, ny, nv, nth, T))
    running_min = np.full((nx, ny, nv, nth), fill_value=np.inf)

    # Walk from t=0 (i=T-1) toward t=lookback_length (i=0)
    for i in range(T - 1, -1, -1):
        brs_vx = v_vx_all[..., i][:, np.newaxis, :, :]   # (nx, 1,  nv, nth)
        brs_vy = v_vy_all[..., i][np.newaxis, :, :, :]   # (1,  ny, nv, nth)

        brs_4d      = np.maximum(brs_vx, brs_vy)          # intersection
        running_min = np.minimum(running_min, brs_4d)      # union over time
        brt_all[..., i] = running_min

    return brt_all  # (nx, ny, nv, nth, T)


def main(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    c = CFG

    tau = np.arange(
        start=0,
        stop=c["lookback_length"] + c["small_number"],
        step=c["dt"],
    )
    T = len(tau)

    # ── Vx subsystem ─────────────────────────────────────────────────────
    print("Solving Vx subsystem  [x, v, theta] ...")
    t0 = time.time()
    v_vx_all = solve_vx(c, tau)
    t_vx = time.time() - t0
    print(f"  shape : {v_vx_all.shape}   time : {t_vx:.1f}s")
    print(f"  BRS volume at final step (V<0): {(v_vx_all[..., 0] < 0).mean():.3f}")

    # ── Vy subsystem ─────────────────────────────────────────────────────
    print("Solving Vy subsystem  [y, v, theta] ...")
    t0 = time.time()
    v_vy_all = solve_vy(c, tau)
    t_vy = time.time() - t0
    print(f"  shape : {v_vy_all.shape}   time : {t_vy:.1f}s")
    print(f"  BRS volume at final step (V<0): {(v_vy_all[..., 0] < 0).mean():.3f}")

    # ── Reconstruct full 4D BRT (Proposition 4) ───────────────────────────
    print(f"Reconstructing 4D BRT over {T} time steps ...")
    t0 = time.time()
    v_brt = reconstruct_brt_4d(v_vx_all, v_vy_all)
    t_recon = time.time() - t0
    print(f"  shape : {v_brt.shape}   time : {t_recon:.1f}s")
    print(f"  BRT volume at final step (V<0): {(v_brt[..., 0] < 0).mean():.3f}")

    # ── Save ──────────────────────────────────────────────────────────────
    saves = {
        "v_vx_brs.npy": v_vx_all,
        "v_vy_brs.npy": v_vy_all,
        "v_brt.npy":    v_brt,
    }
    for fname, arr in saves.items():
        p = os.path.join(out_dir, fname)
        np.save(p, arr)
        print(f"Saved {fname:20s} shape={arr.shape}  → {p}")

    # ── Manifest ──────────────────────────────────────────────────────────
    manifest = {
        "version": 1,
        "values": {
            "v_vx_brs": {
                "path": "v_vx_brs.npy",
                "shape": list(v_vx_all.shape),
                "axes": ["x", "v", "theta", "time"],
                "note": "pure BRS at each time step (no running-min clamping)",
            },
            "v_vy_brs": {
                "path": "v_vy_brs.npy",
                "shape": list(v_vy_all.shape),
                "axes": ["y", "v", "theta", "time"],
                "note": "pure BRS at each time step (no running-min clamping)",
            },
            "v_brt": {
                "path": "v_brt.npy",
                "shape": list(v_brt.shape),
                "axes": ["x", "y", "v", "theta", "time"],
                "note": "full 4D BRT at every time step; index 0 = full BRT, index -1 = target set",
            },
        },
        "reconstruction": {
            "method": "Proposition 4, Chen et al. 2018",
            "formula": "min_s max(v_vx_brs[...,s][:,None,:,:], v_vy_brs[...,s][None,:,:,:])",
            "time_axis_convention": "index 0 = t=lookback_length, index -1 = t=0 (target)",
        },
        "timing": {
            "vx_seconds":    round(t_vx,    2),
            "vy_seconds":    round(t_vy,    2),
            "recon_seconds": round(t_recon, 2),
        },
    }
    manifest_path = os.path.join(out_dir, "artifact_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved artifact_manifest.json → {manifest_path}")

    metrics = {"config": c, "artifact_manifest": "artifact_manifest.json"}
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics.json           → {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run decomposed DubinsCar4D3 BRT solver (Vx + Vy subsystems)"
    )
    parser.add_argument(
        "--out_dir",
        default="output_DubinsCar4D3_decomposed",
        help="Directory to write outputs (default: ./output_DubinsCar4D3_decomposed)",
    )
    args = parser.parse_args()
    main(args.out_dir)
