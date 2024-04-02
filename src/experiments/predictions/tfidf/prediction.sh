#!/bin/bash
#SBATCH -N 1
#SBATCH --qos=normal
#SBATCH -p cpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=joana.amorim@dal.ca
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mem=15GB
#SBATCH --job-name=REDDIT
#SBATCH --output=REDDIT_%j.out

source /opt/lmod/lmod/init/profile
source /ssd005/projects/amorim_proj/envfiles/DS.sh

python prediction.py