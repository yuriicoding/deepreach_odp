"""
Example: 4-D Dubins Car (bicycle model, no disturbance) — DubinsCar4D3

Solves the Backward Reachable Tube (BRT) for a box target set and saves
the value function arrays + JSON metadata, matching the artifact layout
described in artifact_manifest.json / metrics.json.

Output (written to --out_dir, default ./output_DubinsCar4D3/):
  v_direct_all.npy     — value function at every time step, shape (nx, ny, nv, nth, T)
  v_direct_final.npy   — value function at the final time step, shape (nx, ny, nv, nth)
  metrics.json         — full config + artifact pointer
  artifact_manifest.json
"""

import argparse
import json
import math
import os

import numpy as np

from odp.Grid import Grid
from odp.Shapes import Lower_Half_Space, Upper_Half_Space, Intersection
from odp.dynamics import DubinsCar4D3
from odp.solver import HJSolver


# ---------------------------------------------------------------------------
# Configuration — mirrors metrics.json exactly
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
    # target box (x-y only; v and theta are unconstrained)
    "x_center": 0.0, "y_center": 2.0,
    "box_radius_x": 0.8, "box_radius_y": 0.8,
    # control bounds
    "umin_a": -1.5,            "umax_a": 1.5,
    "umin_w": -math.pi / 18,   "umax_w":  math.pi / 18,
    # solver
    "accuracy": "medium",
    "target_set_mode": "minVWithV0",
}


def build_box_target(grid, x_c, y_c, rx, ry):
    """
    Implicit surface for max(|x - x_c| - rx, |y - y_c| - ry).
    Negative inside the box, positive outside — matches Dubins4D_new.boundary_fn.
    v and theta dimensions are unconstrained.
    """
    box_x = Intersection(
        Lower_Half_Space(grid, 0, x_c + rx),
        Upper_Half_Space(grid, 0, x_c - rx),
    )
    box_y = Intersection(
        Lower_Half_Space(grid, 1, y_c + ry),
        Upper_Half_Space(grid, 1, y_c - ry),
    )
    return Intersection(box_x, box_y)


def main(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    c = CFG

    # -----------------------------------------------------------------------
    # Grid  (dimension 3 = theta is periodic)
    # -----------------------------------------------------------------------
    g = Grid(
        np.array([c["x_min"], c["y_min"], c["v_min"], c["th_min"]]),
        np.array([c["x_max"], c["y_max"], c["v_max"], c["th_max"]]),
        4,
        np.array([c["nx"], c["ny"], c["nv"], c["nth"]]),
        [3],
    )

    # -----------------------------------------------------------------------
    # Dynamics
    # -----------------------------------------------------------------------
    my_car = DubinsCar4D3(
        uMin=[c["umin_a"], c["umin_w"]],
        uMax=[c["umax_a"], c["umax_w"]],
        uMode="max",    # avoid: maximise value function
    )

    # -----------------------------------------------------------------------
    # Target set
    # -----------------------------------------------------------------------
    target = build_box_target(
        g,
        c["x_center"], c["y_center"],
        c["box_radius_x"], c["box_radius_y"],
    )

    # -----------------------------------------------------------------------
    # Time vector
    # -----------------------------------------------------------------------
    tau = np.arange(
        start=0,
        stop=c["lookback_length"] + c["small_number"],
        step=c["dt"],
    )

    # -----------------------------------------------------------------------
    # Solve HJ PDE
    # -----------------------------------------------------------------------
    result = HJSolver(
        my_car, g, target, tau,
        {"TargetSetMode": c["target_set_mode"]},
        saveAllTimeSteps=True,
        accuracy=c["accuracy"],
    )
    # result shape: (nx, ny, nv, nth, T)  where T = len(tau)

    v_direct_all   = result                  # (60, 60, 20, 36, 31)
    v_direct_final = result[..., 0]          # (60, 60, 20, 36)  — t = lookback_length

    # -----------------------------------------------------------------------
    # Save arrays
    # -----------------------------------------------------------------------
    all_path   = os.path.join(out_dir, "v_direct_all.npy")
    final_path = os.path.join(out_dir, "v_direct_final.npy")
    np.save(all_path,   v_direct_all)
    np.save(final_path, v_direct_final)
    print(f"Saved v_direct_all   {v_direct_all.shape}  → {all_path}")
    print(f"Saved v_direct_final {v_direct_final.shape} → {final_path}")

    # -----------------------------------------------------------------------
    # artifact_manifest.json
    # -----------------------------------------------------------------------
    manifest = {
        "version": 1,
        "values": {
            "v_direct_all": {
                "path": "v_direct_all.npy",
                "shape": list(v_direct_all.shape),
            },
            "v_direct_final": {
                "path": "v_direct_final.npy",
                "shape": list(v_direct_final.shape),
            },
        },
        "masks": {},
        "training_inputs": {
            "evaluation_reference_all":   "v_direct_all.npy",
            "evaluation_reference_final": "v_direct_final.npy",
        },
        "notes": {
            "description": "raw 4D direct ODP solution — DubinsCar4D3 bicycle model, no disturbance",
        },
    }
    manifest_path = os.path.join(out_dir, "artifact_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved artifact_manifest.json → {manifest_path}")

    # -----------------------------------------------------------------------
    # metrics.json
    # -----------------------------------------------------------------------
    metrics = {
        "config": {k: v for k, v in c.items()},
        "artifact_manifest": "artifact_manifest.json",
    }
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics.json           → {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DubinsCar4D3 BRT solver")
    parser.add_argument(
        "--out_dir",
        default="output_DubinsCar4D3",
        help="Directory to write .npy and .json outputs (default: ./output_DubinsCar4D3)",
    )
    args = parser.parse_args()
    main(args.out_dir)
