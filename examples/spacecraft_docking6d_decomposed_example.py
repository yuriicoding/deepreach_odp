"""
Example: Decomposed BRT for SpacecraftDocking6D
(Clohessy-Wiltshire translation + single-axis rotation)

Solves the BRS at every time step independently for each subsystem:
  - SpacecraftDocking6DTrans : state = [px, py, vx, vy]   (4D, CW dynamics)
  - SpacecraftDocking6DRot   : state = [theta, omega]     (2D, pure rotation)

Then reconstructs the full 6D BRT via Proposition 4 (Chen et al. 2018):

  BRS_6D(s) = proj^{-1}(BRS_trans(s)) ∩ proj^{-1}(BRS_rot(s))
            = max( V_trans_6D(s), V_rot_6D(s) )          [intersection in level-set]

  BRT_6D    = union_{s} BRS_6D(s)
            = min_{s}   max(V_trans_6D(s), V_rot_6D(s))  [union in level-set]

INDEPENDENCE NOTE — this decomposition differs from DubinsCar4D3:
  Translation [px, py, vx, vy] uses [Fx, Fy].
  Rotation    [theta, omega]   uses [tau].
  No shared state, no shared control → reconstruction is EXACT (no approximation).

  The gap |V_trans_6D - V_rot_6D| is therefore NOT an approximation error.
  It reflects which subsystem is the binding constraint at each 6D point.
  Expected to be LARGE on average (two value functions measuring different
  physical quantities at different scales), but this carries no negative
  implication for the quality of the reconstructed BRT.

Output (written to --out_dir, default ./output_SpacecraftDocking6D_decomposed/):
  v_trans_brs.npy        — translation BRS at all time steps, shape (npx, npy, nvx, nvy, T)
  v_rot_brs.npy          — rotation    BRS at all time steps, shape (nth, nom, T)
  v_brt.npy              — reconstructed full 6D BRT,         shape (npx, npy, nvx, nvy, nth, nom, T)
  close_value_gap_all.npy — |V_trans_6D - V_rot_6D|,          shape (npx, npy, nvx, nvy, nth, nom, T)
"""

import argparse
import json
import math
import os
import time

import numpy as np

from odp.Grid import Grid
from odp.Shapes import ShapeRectangle
from odp.dynamics import SpacecraftDocking6DTrans, SpacecraftDocking6DRot
from odp.solver import HJSolver


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CFG = {
    # time
    "tmax": 10.0,
    "dt": 0.1,
    "small_number": 1e-5,

    # translation grid  [px (m), py (m), vx (m/s), vy (m/s)]
    # Grid resolution is kept coarse so the 6D output arrays fit in memory:
    #   v_brt and close_value_gap_all each have shape
    #   (npx, npy, nvx, nvy, nth, nom, T).
    #   At float32: 10*10*8*8*12*8*101 * 4 bytes ≈ 248 MB per array.
    #   Increase resolution only if you have sufficient RAM.
    "px_min": -100.0, "px_max": 100.0,
    "py_min": -100.0, "py_max": 100.0,
    "vx_min":   -3.0, "vx_max":   3.0,
    "vy_min":   -3.0, "vy_max":   3.0,
    "npx": 10, "npy": 10, "nvx": 8, "nvy": 8,

    # rotation grid  [theta (rad), omega (rad/s)]
    "th_min": -math.pi, "th_max": math.pi,
    "om_min":     -0.5, "om_max":     0.5,
    "nth": 12, "nom": 8,

    # docking target box
    #   T_trans = { |px| <= r_px, |py| <= r_py, |vx| <= r_vx, |vy| <= r_vy }
    #   T_rot   = { |theta| <= r_th, |omega| <= r_om }
    "px_center": 0.0, "px_radius": 10.0,
    "py_center": 0.0, "py_radius": 10.0,
    "vx_center": 0.0, "vx_radius":  0.5,
    "vy_center": 0.0, "vy_radius":  0.5,
    "th_center": 0.0, "th_radius":  0.3,
    "om_center": 0.0, "om_radius":  0.05,

    # spacecraft parameters
    "n": 0.001131,  # orbital mean motion [rad/s], ~400 km LEO
    "m": 100.0,     # chaser mass [kg]
    "I": 50.0,      # moment of inertia [kg*m^2]

    # control bounds
    "Fx_min": -20.0, "Fx_max": 20.0,   # [N]
    "Fy_min": -20.0, "Fy_max": 20.0,   # [N]
    "tau_min":  -1.5, "tau_max":  1.5,  # [N*m]

    # solver
    "accuracy": "medium",
    # "none" is unrecognised by solver post-processing → no running-min clamping
    # → solver returns pure BRS at each saved time step
    "target_set_mode": "none",
}


def solve_trans(c, tau):
    """Solve translation subsystem BRS at every time step.

    State = [px, py, vx, vy].  Returns array of shape (npx, npy, nvx, nvy, T).
    Index  -1 (last)  = t = 0                (initial target set).
    Index   0 (first) = t = tmax             (full BRS backward from tmax).
    """
    g = Grid(
        np.array([c["px_min"], c["py_min"], c["vx_min"], c["vy_min"]]),
        np.array([c["px_max"], c["py_max"], c["vx_max"], c["vy_max"]]),
        4,
        np.array([c["npx"], c["npy"], c["nvx"], c["nvy"]]),
        [],  # no periodic dimensions
    )

    target = ShapeRectangle(
        g,
        np.array([
            c["px_center"] - c["px_radius"],
            c["py_center"] - c["py_radius"],
            c["vx_center"] - c["vx_radius"],
            c["vy_center"] - c["vy_radius"],
        ]),
        np.array([
            c["px_center"] + c["px_radius"],
            c["py_center"] + c["py_radius"],
            c["vx_center"] + c["vx_radius"],
            c["vy_center"] + c["vy_radius"],
        ]),
    )

    dyn = SpacecraftDocking6DTrans(
        uMin=[c["Fx_min"], c["Fy_min"]],
        uMax=[c["Fx_max"], c["Fy_max"]],
        uMode="min",
        n=c["n"],
        m=c["m"],
    )

    return HJSolver(
        dyn, g, target, tau,
        {"TargetSetMode": c["target_set_mode"]},
        saveAllTimeSteps=True,
        accuracy=c["accuracy"],
    )   # shape (npx, npy, nvx, nvy, T)


def solve_rot(c, tau):
    """Solve rotation subsystem BRS at every time step.

    State = [theta, omega].  Returns array of shape (nth, nom, T).
    """
    g = Grid(
        np.array([c["th_min"], c["om_min"]]),
        np.array([c["th_max"], c["om_max"]]),
        2,
        np.array([c["nth"], c["nom"]]),
        [0],  # theta (dim 0) is periodic
    )

    target = ShapeRectangle(
        g,
        np.array([
            c["th_center"] - c["th_radius"],
            c["om_center"] - c["om_radius"],
        ]),
        np.array([
            c["th_center"] + c["th_radius"],
            c["om_center"] + c["om_radius"],
        ]),
    )

    dyn = SpacecraftDocking6DRot(
        uMin=[c["tau_min"]],
        uMax=[c["tau_max"]],
        uMode="min",
        I=c["I"],
    )

    return HJSolver(
        dyn, g, target, tau,
        {"TargetSetMode": c["target_set_mode"]},
        saveAllTimeSteps=True,
        accuracy=c["accuracy"],
    )   # shape (nth, nom, T)


def reconstruct_brt_6d(v_trans_all, v_rot_all):
    """Reconstruct full 6D BRT at every time step (Proposition 4, Chen et al. 2018).

    Since the translational and rotational subsystems are independent (no shared
    state or control), this reconstruction is EXACT — not a conservative bound.

    At each time step s:
      BRS_6D(s) = proj^{-1}(BRS_trans(s)) ∩ proj^{-1}(BRS_rot(s))
               = max( V_trans_6D(s), V_rot_6D(s) )

    BRT_6D accumulated over time:
      BRT_6D(s) = union_{r<=s} BRS_6D(r)
               = min_{r<=s}  max(V_trans_6D(r), V_rot_6D(r))   [level-set union]

    Time axis convention (same as direct solver):
      index -1 (last)  → t = 0          (initial target set)
      index  0 (first) → t = tmax       (full BRT)

    Iterates from i = T-1 (t=0) down to i = 0 (t=tmax), accumulating a
    running minimum so brt_all[..., i] = BRT up to backward time at step i.

    Args:
        v_trans_all : (npx, npy, nvx, nvy, T)  — translation BRS at each step
        v_rot_all   : (nth, nom, T)             — rotation    BRS at each step

    Returns:
        (npx, npy, nvx, nvy, nth, nom, T) — full 6D BRT at every time step
    """
    T   = v_trans_all.shape[-1]
    npx, npy, nvx, nvy = v_trans_all.shape[:4]
    nth, nom            = v_rot_all.shape[:2]

    brt_all     = np.zeros((npx, npy, nvx, nvy, nth, nom, T), dtype=np.float32)
    running_min = np.full((npx, npy, nvx, nvy, nth, nom), fill_value=np.inf,
                          dtype=np.float32)

    for i in range(T - 1, -1, -1):
        # Lift each subsystem BRS to 6D via broadcasting
        brs_t = v_trans_all[..., i][:, :, :, :, np.newaxis, np.newaxis]  # (npx,npy,nvx,nvy,1,1)
        brs_r = v_rot_all[..., i][np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :]  # (1,1,1,1,nth,nom)

        brs_6d      = np.maximum(brs_t, brs_r)           # intersection in level-set
        running_min = np.minimum(running_min, brs_6d)    # union over time
        brt_all[..., i] = running_min

    return brt_all   # (npx, npy, nvx, nvy, nth, nom, T)


def compute_close_value_gap(v_trans_all, v_rot_all):
    """Compute per-grid-point gap |V_trans_6D - V_rot_6D| at every time step.

    For independent subsystems the gap is NOT an approximation error.
    It indicates which subsystem is the binding constraint at each 6D point
    and is used in DeepReach as an adaptive guidance weight.

    The gap is expected to be LARGE on average because:
      - V_trans and V_rot measure different physics (position/velocity vs angle/rate).
      - At most 6D state points, one constraint dominates the other.
      - Zero gap occurs only where both constraints are simultaneously binding.

    Args:
        v_trans_all : (npx, npy, nvx, nvy, T)
        v_rot_all   : (nth, nom, T)

    Returns:
        (npx, npy, nvx, nvy, nth, nom, T) float32
    """
    vt = v_trans_all[:, :, :, :, np.newaxis, np.newaxis, :]   # (npx,npy,nvx,nvy,1,1,T)
    vr = v_rot_all[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :, :]  # (1,1,1,1,nth,nom,T)
    return np.abs(vt - vr).astype(np.float32)   # (npx,npy,nvx,nvy,nth,nom,T)


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

    # -- Translation subsystem -----------------------------------------------
    print("\nSolving translation subsystem  [px, py, vx, vy] ...")
    t0 = time.time()
    v_trans_all = solve_trans(c, tau)
    t_trans = time.time() - t0
    print(f"  shape : {v_trans_all.shape}   time : {t_trans:.1f}s")
    print(f"  BRS volume at tmax (V<0): {(v_trans_all[..., 0] < 0).mean():.4f}")

    # -- Rotation subsystem --------------------------------------------------
    print("\nSolving rotation subsystem  [theta, omega] ...")
    t0 = time.time()
    v_rot_all = solve_rot(c, tau)
    t_rot = time.time() - t0
    print(f"  shape : {v_rot_all.shape}   time : {t_rot:.1f}s")
    print(f"  BRS volume at tmax (V<0): {(v_rot_all[..., 0] < 0).mean():.4f}")

    # -- Reconstruct full 6D BRT (exact, Proposition 4) ---------------------
    print(f"\nReconstructing 6D BRT over {T} time steps ...")
    t0 = time.time()
    v_brt = reconstruct_brt_6d(v_trans_all, v_rot_all)
    t_recon = time.time() - t0
    print(f"  shape : {v_brt.shape}   time : {t_recon:.1f}s")
    print(f"  BRT volume at tmax (V<0): {(v_brt[..., 0] < 0).mean():.4f}")

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
        "v_brt.npy":               v_brt,
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
                "note": "pure BRS at each time step (no running-min clamping)",
            },
            "v_rot_brs": {
                "path": "v_rot_brs.npy",
                "shape": list(v_rot_f32.shape),
                "axes": ["theta", "omega", "time"],
                "note": "pure BRS at each time step (no running-min clamping)",
            },
            "v_brt": {
                "path": "v_brt.npy",
                "shape": list(v_brt.shape),
                "axes": ["px", "py", "vx", "vy", "theta", "omega", "time"],
                "note": "exact 6D BRT at every time step; index 0 = full BRT, index -1 = target set",
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
            "time_axis_convention": "index 0 = t=tmax (full BRT), index -1 = t=0 (target set)",
            "independence_note": (
                "Translation [px,py,vx,vy] and rotation [theta,omega] share no state "
                "and no control.  The reconstruction max(V_trans, V_rot) is therefore "
                "exact, not a conservative bound.  The gap |V_trans - V_rot| is large "
                "on average but does not indicate any approximation error."
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
    """Recompute close_value_gap_all.npy from already-saved subsystem arrays.

    Use this when v_trans_brs.npy and v_rot_brs.npy already exist and you only
    need to (re-)generate close_value_gap_all.npy (e.g. after a grid change).
    Updates artifact_manifest.json in-place if it exists.
    """
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
        description="Run decomposed SpacecraftDocking6D BRT solver "
                    "(translation 4D + rotation 2D subsystems)"
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
