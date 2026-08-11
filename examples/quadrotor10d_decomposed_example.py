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

BRS vs BRT — the one thing not to get wrong
-------------------------------------------
Each subsystem is solved with TargetSetMode="none", so the saved arrays are the
per-time-step BRS (value at exactly time t), NOT the running-min BRT. The
max-decomposition (Chen et al. 2018, Prop. 1 + 4) holds per time step; the
union-over-time does not commute with the intersection-over-subsystems:

    CORRECT :  V_BRT(x, t) = min_{s <= t} max_i V_i(x_i, s)
    WRONG   :  V_BRT(x, t) = max_i min_{s <= t} V_i(x_i, s)

The wrong ordering satisfies max_i min_s (...) <= min_s max_i (...), so it
under-estimates the value and over-estimates the BRT. Take the max over
subsystems FIRST, the min over time SECOND. This matches DeepReach's
loss_type='brt_hjivi' with set_mode='reach'.

Time-axis convention (same as HJSolver / the 6D example):
    index  0 (first) = t = tmax
    index -1 (last)  = t = 0  (the initial target function)

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
  v_x_brs.npy  — X BRS at all time steps, (npx, nvx, nth, nq, T)   ~690 MB
  v_y_brs.npy  — Y BRS at all time steps, (npy, nvy, nph, np_, T)  ~690 MB
  v_z_brs.npy  — Z BRS at all time steps, (npz, nvz, T)            ~1.6 MB
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
    # v_x_brs / v_y_brs shape: 41^4 x T x 4 bytes ~= 690 MB each (T = 61)
    # v_z_brs           shape: 81^2 x T x 4 bytes ~= 1.6 MB
    "px_min": -2.0,  "px_max": 2.0,
    "vx_min": -1.5,  "vx_max": 1.5,
    "th_min": -0.25, "th_max": 0.25,     # theta [rad]
    "q_min":  -1.5,  "q_max":  1.5,      # pitch rate [rad/s]
    "npx": 41, "nvx": 41, "nth": 41, "nq": 41,

    "py_min": -2.0,  "py_max": 2.0,
    "vy_min": -1.5,  "vy_max": 1.5,
    "ph_min": -0.25, "ph_max": 0.25,     # phi [rad]
    "p_min":  -1.5,  "p_max":  1.5,      # roll rate [rad/s]
    "npy": 41, "nvy": 41, "nph": 41, "np": 41,

    "pz_min": -2.0, "pz_max": 2.0,
    "vz_min": -1.5, "vz_max": 1.5,
    "npz": 81, "nvz": 81,

    # physics — gravity is NOT in DeepReach's state config; it is a constructor
    # arg there (default 9.81). Both sides must use this value.
    "gravity": 9.81,

    # control — shared_l2_control: u_theta^2 + u_phi^2 + u_z^2 <= u_max^2,
    # projected to the interval [-u_max, u_max] for each subsystem.
    "u_max": 2.0,

    # target box half-width, applied to all 10 components
    "target_radius": 0.2,

    # solver — "none" -> no running-min clamping -> pure BRS per step, which is
    # what the decomposition needs (see the BRS-vs-BRT note above).
    "accuracy": "medium",
    "target_set_mode": "none",
    "u_mode": "min",     # reach
}


# ---------------------------------------------------------------------------
# Subsystem solvers
# ---------------------------------------------------------------------------

def solve_x(c, tau):
    """Solve X subsystem BRS at every time step.

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
    """Solve Y subsystem BRS at every time step.

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
    """Solve Z subsystem BRS at every time step.

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

def mirror_symmetry_error(v_x_all, v_y_all):
    """Max |Vy(py,vy,phi,p) - Vx(py,vy,-phi,-p)| over the whole array.

    The Y dynamics map onto the X dynamics under (phi, p) -> (-phi, -p), and
    both the target box and the control interval are symmetric, so with
    identical grids and bounds the two value functions are exact mirrors. The
    grids ARE symmetric about 0 with an odd point count, so reversing the last
    two spatial axes realises that reflection exactly on grid nodes.

    This is a pure solver sanity check — a value far above discretisation noise
    means one of the two dynamics classes (or their grids) is wrong.
    """
    same_grid = v_x_all.shape == v_y_all.shape
    if not same_grid:
        return None
    return float(np.abs(v_y_all - v_x_all[:, :, ::-1, ::-1]).max())


# ---------------------------------------------------------------------------
# Main
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

    # -- X subsystem ---------------------------------------------------------
    print("\nSolving X subsystem  [px, vx, theta, q]  (BRS) ...")
    t0 = time.time()
    v_x_all = solve_x(c, tau)
    t_x = time.time() - t0
    print(f"  shape : {v_x_all.shape}   time : {t_x:.1f}s")
    print(f"  BRS volume at tmax (V<0): {(v_x_all[..., 0] < 0).mean():.4f}")

    # -- Y subsystem ---------------------------------------------------------
    print("\nSolving Y subsystem  [py, vy, phi, p]  (BRS) ...")
    t0 = time.time()
    v_y_all = solve_y(c, tau)
    t_y = time.time() - t0
    print(f"  shape : {v_y_all.shape}   time : {t_y:.1f}s")
    print(f"  BRS volume at tmax (V<0): {(v_y_all[..., 0] < 0).mean():.4f}")

    # -- Z subsystem ---------------------------------------------------------
    print("\nSolving Z subsystem  [pz, vz]  (BRS) ...")
    t0 = time.time()
    v_z_all = solve_z(c, tau)
    t_z = time.time() - t0
    print(f"  shape : {v_z_all.shape}   time : {t_z:.1f}s")
    print(f"  BRS volume at tmax (V<0): {(v_z_all[..., 0] < 0).mean():.4f}")

    # -- Sanity check: X/Y mirror symmetry -----------------------------------
    mirror_err = mirror_symmetry_error(v_x_all, v_y_all)
    if mirror_err is None:
        print("\nX/Y mirror check skipped (grids differ)")
    else:
        print(f"\nX/Y mirror check: max |Vy(phi,p) - Vx(-phi,-p)| = {mirror_err:.3e}")

    # -- Save subsystem arrays (no full 10D grid — see module docstring) -----
    v_x_f32 = v_x_all.astype(np.float32)
    v_y_f32 = v_y_all.astype(np.float32)
    v_z_f32 = v_z_all.astype(np.float32)
    for fname, arr in [("v_x_brs.npy", v_x_f32),
                       ("v_y_brs.npy", v_y_f32),
                       ("v_z_brs.npy", v_z_f32)]:
        p = os.path.join(out_dir, fname)
        np.save(p, arr)
        print(f"Saved {fname:14s} shape={str(arr.shape):28s} {arr.nbytes/1e6:.1f} MB  -> {p}")

    # -- Manifest ------------------------------------------------------------
    # Paths are absolute: the manifest is consumed from a different directory
    # than the one holding the arrays (see odp_10d.sh).
    root = os.path.abspath(out_dir)
    manifest = {
        "version": 1,
        "root": root,
        "system": "QuadrotorHover10D",
        "control_config": "shared_l2_control",
        "values": {
            "v_x_brs": {
                "path": os.path.join(root, "v_x_brs.npy"),
                "shape": list(v_x_f32.shape),
                "axes": ["px", "vx", "theta", "q", "time"],
                "state_idx": [0, 1, 2, 3],
                "grid_min": [c["px_min"], c["vx_min"], c["th_min"], c["q_min"]],
                "grid_max": [c["px_max"], c["vx_max"], c["th_max"], c["q_max"]],
                "note": "X subsystem BRS at each time step (no running-min clamping)",
            },
            "v_y_brs": {
                "path": os.path.join(root, "v_y_brs.npy"),
                "shape": list(v_y_f32.shape),
                "axes": ["py", "vy", "phi", "p", "time"],
                "state_idx": [4, 5, 6, 7],
                "grid_min": [c["py_min"], c["vy_min"], c["ph_min"], c["p_min"]],
                "grid_max": [c["py_max"], c["vy_max"], c["ph_max"], c["p_max"]],
                "note": "Y subsystem BRS at each time step (no running-min clamping)",
            },
            "v_z_brs": {
                "path": os.path.join(root, "v_z_brs.npy"),
                "shape": list(v_z_f32.shape),
                "axes": ["pz", "vz", "time"],
                "state_idx": [8, 9],
                "grid_min": [c["pz_min"], c["vz_min"]],
                "grid_max": [c["pz_max"], c["vz_max"]],
                "note": "Z subsystem BRS at each time step (no running-min clamping)",
            },
        },
        "time": {
            "tmax": c["tmax"],
            "dt": c["dt"],
            "num_steps": T,
            "index_0": "t = tmax",
            "index_-1": "t = 0 (target function)",
        },
        "reconstruction": {
            "full_grid_materialized": False,
            "reason": "full 10D grid is 41^4 * 41^4 * 81^2 ~= 5.2e16 cells per "
                      "time step; evaluate on demand instead",
            "method": "Proposition 1 + 4, Chen et al. 2018 (max over subsystems, per time step)",
            "formula_brs": "V10D(x, t) = max(Vx(x[0:4], t), Vy(x[4:8], t), Vz(x[8:10], t))",
            "formula_brt": "V10D_BRT(x, t) = min_{s <= t} max_i V_i(x_i, s)  "
                           "-- max over subsystems FIRST, min over time SECOND",
            "exact": False,
            "exactness_note": "shared_l2_control: each subsystem uses the projected "
                              "interval [-u_max, u_max], whose product box strictly "
                              "contains the L2 ball, so this reconstruction lower-bounds "
                              "the true coupled value (over-approximates the BRT).",
        },
        "validation": {
            "xy_mirror_max_abs_err": mirror_err,
            "xy_mirror_identity": "Vy(py,vy,phi,p,t) == Vx(py,vy,-phi,-p,t)",
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
        description="Run decomposed QuadrotorHover10D BRS solver "
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
