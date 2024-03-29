#!bin/sh

python3.9 -u fg_training.py --root /fs/ess/PAS0027/nyx_vdl/512/train/ --dir-weights /users/PAS2171/chen10522/Surrogate/model_weights/ --dir-outputs /users/PAS2171/chen10522/Surrogate/outputs/ --batch-size 262144 --sp-sr 1.0 --sf-sr 0.10 --log-every 1 --check-every 2 --epochs 50 --loss MSE --dim3d 64 --dim2d 256 --dim1d 16 --spatial-fdim 32 --param-fdim 16 --dropout 0 --fg-version 9