# DubinsCar4D3 Decomposed BRT — Usage Guide

This section documents the decomposed 3D subsystem classes for the 4D bicycle-model Dubins car
(`DubinsCar4D3`) and the example script that computes the full 4D Backward Reachable Tube (BRT)
via system decomposition.

## Background

`DubinsCar4D3` has state `[x, y, v, θ]` and control `[a, δ]` (acceleration, steering angle).
Computing its BRT directly in 4D is expensive. Using the decomposition method from
*"Decomposition of Reachable Sets and Tubes for a Class of Nonlinear Systems"* (Chen et al. 2018),
the 4D system is split into two 3D subsystems that share the common states `(v, θ)`:

| Class | File | State | Position axis dropped |
|---|---|---|---|
| `DubinsCar4D3Vx` | `odp/dynamics/DubinsCar4D3Vx.py` | `[x, v, θ]` | y |
| `DubinsCar4D3Vy` | `odp/dynamics/DubinsCar4D3Vy.py` | `[y, v, θ]` | x |

Each subsystem is a self-contained 3D HJ problem. The full 4D BRT is reconstructed from their
outputs with no additional approximation error (Proposition 4, Theorem 2 of the paper).

## Running the decomposed BRT example

```bash
cd /path/to/optimized_dp
python examples/dubins_car4d3_decomposed_example.py
# or specify a custom output directory:
python examples/dubins_car4d3_decomposed_example.py --out_dir my_output/
```

This runs two 3D solves (Vx then Vy) and reconstructs the full 4D BRT.

## Output files

All files are written to `output_DubinsCar4D3_decomposed/` (or the directory given by `--out_dir`).

### `v_vx_brs.npy` — shape `(nx, nv, nth, T)`

BRS of the Vx subsystem `[x, v, θ]` saved at every time step.
- Axis order: `[x, v, θ, time]`
- `[..., -1]` = initial target set (t = 0)
- `[..., 0]`  = BRS at the full lookback time

No running-minimum clamping is applied, so each time slice is the pure BRS at exactly that
backward time (not a cumulative tube).

### `v_vy_brs.npy` — shape `(ny, nv, nth, T)`

BRS of the Vy subsystem `[y, v, θ]` saved at every time step. Same conventions as above.

### `v_brt.npy` — shape `(nx, ny, nv, nth, T)`

Full reconstructed 4D BRT at every time step. Axis order: `[x, y, v, θ, time]`.
- `[..., 0]`  = full BRT at t = lookback\_length (the final result you typically use)
- `[..., -1]` = BRT at t = 0 (equals the initial target set)
- `[..., i]`  = BRT accumulated up to backward time step i

Matches the shape and axis convention of `v_direct_all.npy` produced by
`dubins_car4d3_example.py`, so slices can be compared directly:

```python
import numpy as np
v_direct = np.load("output_DubinsCar4D3/v_direct_all.npy")    # (60,60,20,36,31)
v_decomp  = np.load("output_DubinsCar4D3_decomposed/v_brt.npy")  # (60,60,20,36,31)
print("Max abs difference:", np.abs(v_direct - v_decomp).max())
# Expected: near zero (only floating-point noise from independent grid numerics)
```

### `artifact_manifest.json`

JSON file recording the shape, axis order, and reconstruction formula for each saved array.

### `metrics.json`

JSON file recording the full solver configuration (grid bounds, resolution, control limits,
target box, time horizon) and a pointer to `artifact_manifest.json`.

## How the reconstruction works

The full BRT cannot be obtained by simply intersecting the two subsystem BRTs — the paper
(Table II) shows this is invalid for an intersection target with shared controls, because
timing information is lost.

The correct method (Proposition 4) is:

1. **For each time step s**, compute the 4D BRS by back-projecting and intersecting the
   subsystem BRSs at that same time:
   ```
   BRS_full(s) = max(BRS_Vx_4D(s), BRS_Vy_4D(s))   ← intersection in level-set
   ```
2. **Union over all time steps** to form the BRT:
   ```
   BRT_full = min over all s of BRS_full(s)           ← union in level-set
   ```

In NumPy this is a running minimum over the time axis, implemented in `reconstruct_brt_4d()`
inside the example script.

## Running the direct 4D example (baseline)

```bash
cd /path/to/optimized_dp
python examples/dubins_car4d3_example.py
# or specify a custom output directory:
python examples/dubins_car4d3_example.py --out_dir my_output/
```

This solves the BRT for the full `DubinsCar4D3` system directly in 4D without any decomposition.
It is slower than the decomposed approach but serves as the ground-truth result to compare against.

### Output files

All files are written to `output_DubinsCar4D3/` (or the directory given by `--out_dir`).

#### `v_direct_all.npy` — shape `(nx, ny, nv, nth, T)`

Value function saved at every time step. Axis order: `[x, y, v, θ, time]`.
- `[..., 0]`  = full BRT at t = lookback\_length
- `[..., -1]` = initial target set (t = 0)

#### `v_direct_final.npy` — shape `(nx, ny, nv, nth)`

Value function at the final time step only — equivalent to `v_direct_all[..., 0]`.
Cells where the value is negative are inside the BRT (the car will enter the obstacle box
within the time horizon under worst-case control).

#### `artifact_manifest.json` / `metrics.json`

Same format as the decomposed example — shape metadata and full solver configuration.

### Comparing direct vs decomposed

```python
import numpy as np
v_direct = np.load("output_DubinsCar4D3/v_direct_all.npy")       # (60,60,20,36,31)
v_decomp  = np.load("output_DubinsCar4D3_decomposed/v_brt.npy")  # (60,60,20,36,31)
print("Max abs difference:", np.abs(v_direct - v_decomp).max())
# Expected: near zero — decomposition introduces no additional approximation error
```

---










# Optimized Dynamic Programming-Based Algorithms Solver (OptimizedDP)
The repo contains implementation of dynamic programming based algorithms in optimal control. Specifically, the solver supports 3 main classes of algorithms: level set based algorithm for solving Hamilton-Jacobi-Issac (HJI) partial differential equation (PDE) arising in reachability analysis and differential games [1], time-to-reach (TTR) computations of dynamical systems in reachability analysis [2], and value-iterations algorithm for solving continuous state-space action-space Markov Decision Process (MDP). All these algorithms share the property of being implemented on a multidimensional grid and hence, their computational complexities increase exponentially as a function of dimension. For all the aforementioned algorithms, our toolbox allows computation up to 6 dimensions, which we think is the limit of dynammic programming on most modern personal computers.
<div align="center">
    <img src="images/avoid.png" width="470" height="240">
</div>        

<!-- ![Avoid](images/avoid.png) -->

In comparison with previous works, our toolbox strives to be both efficient in implementation while being user-friendly. This is reflected in our choice of having Python as a language for initializing problems and having python-like HeteroCL [3] language for the core algorithms implementation and dynamical systems specification. Our implementation is 7-32x faster than the [Level Set Toolbox](https://github.com/risk-sensitive-reachability/ToolboxLS) and [HelperOC](https://github.com/HJReachability/helperOC) and 2-3x faster than [BEACLS](https://hjreachability.github.io/beacls/) implementation in C++. Please find more details about using the repo for solving your problems in this page, and should you have any questions/problems/requests please direct the messages to Minh Bui at buiminhb@sfu.ca 

# Quickstart (Ubuntu 18, 20, 22)
Please install the following:
* Install Anaconda (https://docs.anaconda.com/anaconda/install/linux/)
* Create a virtual environemnt using the provided `environment.yml`

    ``` conda env create -f environment.yml ```
    
* Install the `optimized_dp` repository

    ``` pip install -e . ```
* Note: If you're on Ubuntu 20.04 or 22.04, you may have encounter an error regarding ``` libtinfo5 ```. 
  To fix, please just run this command 

    ```sudo apt install libtinfo5 ``` 


# Solving the Hamilton-Jacobi-Issac (HJI) PDE
* We provide a running example of solving HJI PDE in the file [`examples/plotting_example.py`](https://github.com/SFU-MARS/optimized_dp/examples/examples.py):
```python
# STEP 1: Define grid
grid_min = np.array([-4.0, -4.0, -math.pi])
grid_max = np.array([4.0, 4.0, math.pi])
dims = 3
N = np.array([150, 150, 150])
pd=[2]
g = Grid(grid_min, grid_max, dims, N, pd)

# STEP 2: Generate initial values for grid using shape functions
center = np.zeros(dims)
radius = 1.0
ignore_dims = [2]
Initial_value_f = CylinderShape(g, ignore_dims, center, radius)

# STEP 3: Time length for computations
Lookback_length = 1.0
t_step = 0.05

small_number = 1e-5
tau = np.arange(start=0, stop=Lookback_length + small_number, step=t_step)

# STEP 4: System dynamics for computation
sys = DubinsCapture(uMode="max", dMode="min")

# STEP 5: Initialize plotting option
po1 = PlotOptions(do_plot=False, plot_type="set", plotDims=[0,1,2])

# STEP 6: Call HJSolver function
compMethod = { "TargetSetMode": "None"}
result_3 = HJSolver(sys, g, Initial_value_f, tau, compMethod, po1, saveAllTimeSteps=True)
```
* To run the example, execute the command `python3 examples/plotting_example.py`
* If the parameter `do_plot` is set to `True`, when initializing `PlotOptions`. The parameter `saveAllTimeSteps` is set to `False` in `HJSolver`, an static 3D plot will show on pop-up browser.
<!-- ![BallPic](images/ball_pic.png) -->
<div align="center">
<img src="images/3D_0_sublevel_set.png" width="500" height="400">
</div>        

* If interactive 3D animation needs to be visualized, and outputs need to be saved locally
```python
# While file needs to be saved locally, set save_fig=True and filename, recommend to set interactive_html=True for better visualization
po2 = PlotOptions(do_plot=False, plot_type="set", plotDims=[0,1,2],
                  slicesCut=[1], colorscale="Bluered", save_fig=True, filename="plots/3D_0_sublevel_set", interactive_html=True)

# STEP 6: Call Plotting function
plot_isosurface(g, result_3, po2)
```

<!-- * Check the interactive result [`images/3D_0_sublevel_set.html`](https://github.com/SFU-MARS/optimized_dp/images/3D_0_sublevel_set.html) -->


* For computing result higher than 6D, check [`examples/examples.py`](https://github.com/SFU-MARS/optimized_dp/examples/examples.py)
* Notes: For 6 dimensions, recommended grid size is 20-30 each dimension on system with 32Gbs of DRAM.
* Create a class file in folder dynamics/ to specify your own system dynamics. Remember to import the class in your running example.  

## System dynamics specification
It can noticed in ```user_definer.py``` that the class DubinsCapture is imported from the folder ```dynamics/```, where our example system dynamics. 
```python 
import heterocl as hcl

class DubinsCapture:
    def __init__(self, x=[0,0,0], wMax=1.0, speed=1.0, dMax=1.0, uMode="max", dMode="min"):
        self.x = x
        (...)
        self.dMode = dMode

    def opt_ctrl(self, t, state, spat_deriv):
        """
                :param  spat_deriv: tuple of spatial derivative in all dimensions
                        state: x1, x2, x3
                        t: time
                :return: a tuple of optimal disturbances
        """

        (...)
        return (opt_w[0], in3[0], in4[0])

    def opt_dstb(self, t, state, spat_deriv):
        """
            :param spat_deriv: tuple of spatial derivative in all dimensions
                    state: x1, x2, x3
                    t: time
            :return: a tuple of optimal disturbances
        """
        (...)        
        return (d1[0], d2[0], d3[0])

    def dynamics(self, t, state, uOpt, dOpt):
        (...)
        return (x_dot[0], y_dot[0], theta_dot[0])
``` 


# Time-to-Reach computation
* We have provided an example in [`examples/TTR_example.py`](https://github.com/SFU-MARS/optimized_dp/examples/TTR_example.py):
```python
# -------------------------------- ONE-SHOT TTR COMPUTATION ---------------------------------- #
g = Grid(minBounds=np.array([-3.0, -1.0, -math.pi]), maxBounds=np.array([3.0, 4.0, math.pi]),
         dims=3, pts_each_dim=np.array([50, 50, 50]), periodicDims=[2])
# Car is trying to reach the target
my_car = DubinsCar(uMode="min")

# Initialize target set as a cylinder
targetSet = CylinderShape(g, [2], np.array([0.0, 1.0, 0.0]), 0.70)
po = PlotOptions( "3d_plot", plotDims=[0,1,2], slicesCut=[],
                  min_isosurface=lookback_length, max_isosurface=lookback_length)

# Convergence threshold
epsilon = 0.001
V_0 = TTRSolver(my_car, g, targetSet, epsilon, po)
```
* To run the example : `python3 TTR_example.py`
# Current code structure
* solver.py: Containing python APIs to interact with the numerical solver
* dynamics/ : User's dynamical system specification
* Shapes/ShapesFunctions.py : Add-in functions for initializing different shapes/intial value functions
* computeGraphs/CustomGraphFunctions.py: Ready-to-user HeteroCL style utility functions

### Paper and Citation (Bibtex)  
Please cite our paper (https://arxiv.org/abs/2204.05520) if you use the toolbox in your research:
```
@misc{https://doi.org/10.48550/arxiv.2204.05520,
  doi = {10.48550/ARXIV.2204.05520},
  url = {https://arxiv.org/abs/2204.05520},
  author = {Bui, Minh and Hu, Hanyang and He, Chong and Lu, Michael and Giovanis, George and Shriraman, Arrvindh and Chen, Mo},
  keywords = {Systems and Control (eess.SY), FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering},
  title = {OptimizedDP: An Efficient, User-friendly Library For Optimal Control and Dynamic Programming},
  publisher = {arXiv},
  year = {2022}, 
  copyright = {Creative Commons Attribution 4.0 International}
}
```  


# Related Projects
### MATLAB
* [A Toolbox of Level Set Methods ](https://www.cs.ubc.ca/~mitchell/ToolboxLS/)
* [helperOC](https://github.com/HJReachability/helperOC)
### C++
* [BEACLS](https://hjreachability.github.io/beacls/)
### Python/JAX
* [hj_reachability](https://github.com/StanfordASL/hj_reachability)

# References
[1] "Hamilton–Jacobi Reachability: Some Recent Theoretical Advances and Applications in Unmanned Airspace Management" by Mo Chen and Claire J. Tomlin in 
Annual Review of Control, Robotics, and Autonomous Systems 2018 1:1, 333-358 [pdf](https://sfumars.com/wp-content/papers/2018_ar_hjreach.pdf)

[2] "One-Shot Computation of Reachable Sets for Differential Games" by Insoon Yang [pdf](https://dl.acm.org/doi/pdf/10.1145/2461328.2461359?casa_token=GmZ6JB2DhLwAAAAA:qRSxxQisIcNpNo6nJHWbi5lRSmxFWk_gL2dXxilkpPi3PsgwxwPSs5hCdcuV7Elx1PTQ84cAGFQ)

[3] "HeteroCL: A Multi-Paradigm Programming Infrastructure for Software-Defined Reconfigurable Computing" by Yi-Hsiang Lai [pdf](https://vast.cs.ucla.edu/~chiyuze/pub/fpga19-heterocl.pdf)

