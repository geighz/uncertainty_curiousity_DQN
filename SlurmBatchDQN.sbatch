#!/bin/bash
#SBATCH -JMontezuma_Plus                    
#SBATCH --account=gts-rkuske7                
#SBATCH -N1 -n1                                
#SBATCH -p cpu-large       
#SBATCH -t8:00:00                                   
#SBATCH -qembers                               
#SBATCH -oReport-%j.out                        
#SBATCH --mail-type=BEGIN,END,FAIL            
#SBATCH --mail-user=gzollicoffer3@gatech.edu 

cd /storage/home/hcoda1/6/gzollicoffer3/p-rkuske7-0/gzollicoffer3/uncertainty_rl/uncertainty_curiousity_DQN                          
eval "$(conda shell.bash hook)"
module load anaconda3/2022.05

conda activate uncertain_rl                   
srun python -u run.py  --env_id ALE/MontezumaRevenge-ram-v5  --exploitation_mode Plus --num_timesteps_per_save 25_000_000 --model_path models/DQN/ALE/MontezumaRevenge-ram-v5/Thompson-Plus-25000000.zip