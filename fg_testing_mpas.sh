#!bin/sh
#PBS -N INRSurro_MPAS_eval
#PBS -l walltime=1:00:00
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l mem=48GB

python3 -u fg_testing_mpas.py --root /fs/ess/PAS0027/mpas_graph/ --dir-weights /fs/ess/PAS0027/mpas_inr/model_weights/ --dir-outputs /fs/ess/PAS0027/mpas_inr/outputs/ --batch-size 1974191 --log-every 1 --check-every 2 --start-epoch 70 --loss Evidential --dim3d 64 --dim2d 256 --dim1d 16 --spatial-fdim 32 --param-fdim 16 --dropout 0