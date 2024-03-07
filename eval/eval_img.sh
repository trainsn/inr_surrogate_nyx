#!/bin/sh
#PBS -N Nyx-eval_img
#PBS -l walltime=4:00:00
#PBS -l nodes=1:ppn=1

python -u eval_img.py --root1 /fs/ess/PAS0027/nyx_vdl/512/img --root2 /fs/ess/PAS0027/yitang_NyxImg --tf 3 --mode 64_256_16_32_16_v9_ensembleimp_