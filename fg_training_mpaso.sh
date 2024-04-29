#!bin/sh
#PBS -N INRSurro_MPAS
#PBS -l walltime=36:00:00
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l mem=48GB

python3 -u fg_training_mpaso.py --root /fs/ess/PAS0027/mpas_graph/ --dir-weights /fs/ess/PAS0027/mpas_inr/model_weights/ --dir-outputs /fs/ess/PAS0027/mpas_inr/outputs/ --batch-size 262144 --sp-sr 1.0 --sf-sr 0.5 --log-every 1 --check-every 5 --epochs 500 --loss MSE --dim3d 64 --dim2d 256 --dim1d 16 --spatial-fdim 32 --param-fdim 16 --dropout 0 
# > fg_training_mpaso.log 2>&1 & 