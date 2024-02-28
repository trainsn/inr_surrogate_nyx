#!bin/sh

python3.9 -u fg_extract_latent.py --root /fs/ess/PAS0027/nyx_vdl/512/test/ --dir-weights /users/PAS2171/chen10522/Surrogate/model_weights/ --dir-outputs /users/PAS2171/chen10522/Surrogate/outputs/ --batch-size 262144 --sp-sr 1.0 --sf-sr 0.10 --log-every 1 --check-every 2 --start-epoch 50 --loss MSE --dim3d 32 --dim2d 64 --dim1d 32 --feature_dim 16