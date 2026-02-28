#!/bin/bash
#SBATCH -J BG_1000
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=80:00:00
#SBATCH --mem=64GB
#SBATCH --partition=3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH -o /users/jdtoscan/data/jdtoscan/References/Variational-Residual-Based-Attention-vRBA-for-PINNs-and-Operator-Networks-/Results_log/Output/BG_1000-%j.out
#SBATCH -e /users/jdtoscan/data/jdtoscan/References/Variational-Residual-Based-Attention-vRBA-for-PINNs-and-Operator-Networks-/Results_log/Errors/BG_1000-%j.err

cd ///users/jdtoscan/data/jdtoscan/References/Variational-Residual-Based-Attention-vRBA-for-PINNs-and-Operator-Networks-/PINN/Potential_analysis/BG_1000/|| exit

nvidia-smi
source /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate JAX_311

rm -rf ./__pycache__/

# python3 -u BG_1000.py  --vrba_potential 'linear' --k_samp 0.0 --c_samp 1.0 
# python3 -u BG_1000.py  --vrba_potential 'exponential'
python3 -u BG_1000.py  --vrba_potential 'cosh'
# python3 -u BG_1000.py  --vrba_potential 'quadratic'
# python3 -u BG_1000.py  --vrba_potential 'lp'
#python3 -u BG_1000.py  --vrba_potential 'logarithmic'
python3 -u BG_1000.py  --vrba_potential 'superexp'
