#!/bin/bash
#SBATCH -J yha231_odp_10d
#SBATCH --nodelist=cs-venus-06
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=0-12:00
#SBATCH --output=odp_10d_%N-%j.out

source /project/mars-lab/yha231/miniconda3/etc/profile.d/conda.sh
conda activate odp

cd /project/mars-lab/yha231/deepreach_odp
export PYTHONPATH=/project/mars-lab/yha231/deepreach_odp:$PYTHONPATH

OUT_DIR=/localscratch/${USER:?USER is not set}/odp_10d_out/
DEEPREACH_DIR=/project/mars-lab/yha231/cmpt720_hybrid_hj/

# 0) Start clean, so a failed step can never leave arrays from an older run
#    sitting next to the new ones.
rm -rf "$OUT_DIR"

# 1) Solve the 3 decomposed subsystems (X 4D, Y 4D, Z 2D).
#    No full 10D grid is reconstructed — it would be ~5.2e16 cells per time
#    step. DeepReach evaluates max(Vx, Vy, Vz) on demand from these arrays.
#    Memory is modest (~690 MB per 4D array), hence 64G rather than the 6D
#    job's 350G.
srun python -u examples/quadrotor10d_decomposed_example.py \
    --out_dir $OUT_DIR \
&& \
# 2) Ship only the manifest to DeepReach; it carries absolute paths back to
#    the .npy arrays, which stay on node-local scratch.
rsync -azh --info=progress2 \
    $OUT_DIR/artifact_manifest.json \
    $DEEPREACH_DIR
