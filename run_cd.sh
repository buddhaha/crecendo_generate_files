#!/bin/bash
#SBATCH -J crescendo  # sensible name for the job
#SBATCH --no-requeue
#SBATCH --wckey=P11J0:CRESCENDO
#SBATCH --nodes=1     
#SBATCH --output=out.%J
#SBATCH --error=err.%J
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00  ###12h queues

cres_ivs_bub='/gpfsgaia/home/g39677/Pour_Mirek/cresc_w_bubble/CRESCENDO_2020_05/codes/build/crescendo_ivs'
module load gcc/8.2.0
##for f in restart.he_in_void.*
##do
##      	echo "Runing MD for $f file....."
##	
##	name=${f/restart.he_in_void./''}	
##	$LAMPS_PATH -var name "$name" -var restart "$f" < in.run_md
##        echo "MD files dumped the for $name configuration!!"
##done

# now run this script in each folder with .cnf files
# ---> srun cresc_ivs_bub

#1) cd ${folder}
#2) sbatch ./run_cd.sh
$cres_ivs_bub
