#!/bin/bash
#SBATCH --nodes=1 
#SBATCH --gpus-per-node=v100l:1   
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1    
#SBATCH --mem=60G
#SBATCH --time=6:30:00

module load StdEnv/2023
module load python/3.12
module load cuda/12
module load gcc arrow scipy-stack
module load gcc opencv scipy-stack
module load cmake
module load clang
module load rust
module load flexiblas
python -c "import pyarrow"
python -c "import cv2"

srun -N $SLURM_NNODES -n $SLURM_NNODES bash << EOF
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install torch torchvision tqdm seaborn sklearn transformers wandb==0.19.6+computecanada argparse
EOF

echo "r$SLURM_NODEID master: $MASTER_ADDR"
echo "r$SLURM_NODEID Launching python script"

# The $((SLURM_NTASKS_PER_NODE * SLURM_JOB_NUM_NODES)) variable tells the script how many processes are available for this execution. “srun” executes the script <tasks-per-node * nodes> times

source $SLURM_TMPDIR/env/bin/activate

srun python /home/saahmed/scratch/projects/Image-segmentation/Different-Data/Scripts/retieval-script.py