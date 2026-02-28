#!/bin/bash
#SBATCH -J BG_100pi
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=80:00:00
#SBATCH --mem=64GB
#SBATCH --partition=3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH -o /users/jdtoscan/data/jdtoscan/References/Variational-Residual-Based-Attention-vRBA-for-PINNs-and-Operator-Networks-/Results_log/Output/BG_100pi-%j.out
#SBATCH -e /users/jdtoscan/data/jdtoscan/References/Variational-Residual-Based-Attention-vRBA-for-PINNs-and-Operator-Networks-/Results_log/Errors/BG_100pi-%j.err

cd /users/jdtoscan/data/jdtoscan/References/Variational-Residual-Based-Attention-vRBA-for-PINNs-and-Operator-Networks-/PINN/Sensitivity_Analysis/Burgers/|| exit

nvidia-smi
source /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate JAX_311

rm -rf ./__pycache__/

python3 -u BG_100pi.py  --vrba_potential 'quadratic' --gamma_bfgs 0.99
python3 -u BG_100pi.py  --vrba_potential 'quadratic' --gamma_bfgs 0.9
python3 -u BG_100pi.py  --vrba_potential 'quadratic' --gamma_bfgs 0.5
python3 -u BG_100pi.py  --vrba_potential 'quadratic' --gamma_bfgs 0.1

