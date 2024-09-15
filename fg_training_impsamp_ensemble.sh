#!bin/sh
#PBS -N INRSurro_Nyx_Evidential
#PBS -l walltime=36:00:00
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l mem=48GB

python3 -u fg_training_impsamp_ensemble.py --root /fs/ess/PAS0027/nyx_vdl/512/train/ --dir-weights /fs/ess/PAS0027/nyx_inr/model_weights/ --dir-outputs /fs/ess/PAS0027/nyx_inr/outputs/ --batch-size 262144 --sf-sr 0.10 --log-every 1 --check-every 2 --start-epoch 0 --epochs 500 --loss Evidential --dim3d 64 --dim2d 256 --dim1d 16 --spatial-fdim 32 --param-fdim 16