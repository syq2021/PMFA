#!/bin/bash
#SBATCH -n 10
#SBATCH --gres=gpu:v100:1
#SBATCH --time=48:00:00

for trial in 1 2 3 4 5 6 7 8 9 10
do
	python train_1205pose_1.py \
	--dataset regdb  \
	--gpu 3 \
  --log_path log_0417_regdb_pose_caj_mmd_cc_0.5/ \
  --model_path save_model_0417_reddb_pose_caj_mmd_cc/ \
	--trial $trial
done
echo 'Done!'