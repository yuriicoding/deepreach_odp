#!/bin/bash
#SBATCH -J yha231_odp_6d
#SBATCH --nodelist=cs-venus-06
#SBATCH --cpus-per-task=32
#SBATCH --mem=380G
#SBATCH --time=3-00:00
#SBATCH --output=odp_6d_%N-%j.out

source /project/mars-lab/yha231/miniconda3/etc/profile.d/conda.sh
conda activate odp

cd /project/mars-lab/yha231/deepreach_odp
export PYTHONPATH=/project/mars-lab/yha231/deepreach_odp:$PYTHONPATH

srun python examples/spacecraft_docking6d_decomposed_example.py \
    --out_dir /localscratch/$USER/odp_6d_out/

rsync -azh --info=progress2 \
    /localscratch/$USER/odp_6d_out/ \
    /project/mars-lab/yha231/deepreach_odp/output_SpacecraftDocking6D_decomposed/ \
&& rm -rf /localscratch/$USER/odp_6d_out/
