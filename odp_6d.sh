#!/bin/bash
#SBATCH -J yha231_odp_6d
#SBATCH --nodelist=cs-venus-06
#SBATCH --cpus-per-task=32
#SBATCH --mem=350G
#SBATCH --time=0-12:00
#SBATCH --output=odp_6d_%N-%j.out

source /project/mars-lab/yha231/miniconda3/etc/profile.d/conda.sh
conda activate odp

cd /project/mars-lab/yha231/deepreach_odp
export PYTHONPATH=/project/mars-lab/yha231/deepreach_odp:$PYTHONPATH

OUT_DIR=/localscratch/${USER:?USER is not set}/odp_6d_out/
DEEPREACH_DIR=/project/mars-lab/yha231/cmpt720_hybrid_hj/

# 0) Start clean, so a failed step can never leave arrays from an older run
#    sitting next to the new ones.
rm -rf "$OUT_DIR"

# 1) Solve the decomposed subsystems and reconstruct the full 6D BRAT.
srun python -u examples/spacecraft_docking6d_decomposed_example.py \
    --out_dir $OUT_DIR \
&& \
# 2) Compute close_value_gap_all.npy from the saved subsystem arrays.
srun python -u examples/spacecraft_docking6d_decomposed_example.py \
    --out_dir $OUT_DIR \
    --gap_only \
&& \
# 3) Ship only the manifest to DeepReach; it carries absolute paths back to
#    the .npy arrays, which stay on node-local scratch.
rsync -azh --info=progress2 \
    $OUT_DIR/artifact_manifest.json \
    $DEEPREACH_DIR
