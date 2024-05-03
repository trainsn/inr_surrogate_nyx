from models import *
from utils import *
import os
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import argparse
from tqdm import tqdm
import math
import itertools

import pdb

# parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Deep Learning Model")
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--root", required=True, type=str,
                        help="root of the dataset")
    parser.add_argument("--dir-weights", required=True, type=str,
                        help="model weights path")
    parser.add_argument("--dir-outputs", required=True, type=str,
                        help="directory for any outputs (ex: images)")
    parser.add_argument("--resume", type=str, default="",
                        help="path to the latest checkpoint (default: none)")
    parser.add_argument("--dsp", type=int, default=3,
                        help="dimensions of the simulation parameters (default: 3)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="learning rate (default: 1e-4)")
    parser.add_argument("--beta1", type=float, default=0.0,
                        help="beta1 of Adam (default: 0.0)")
    parser.add_argument("--beta2", type=float, default=0.999,
                        help="beta2 of Adam (default: 0.999)")
    parser.add_argument("--load-batch", type=int, default=1,
                        help="batch size for loading (default: 1)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch size for training (default: 1)")
    parser.add_argument("--weighted", action="store_true", default=False,
                        help="use weighted L1 Loss")
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="start epoch number (default: 0)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="number of epochs to train (default: 10000)")
    parser.add_argument("--log-every", type=int, default=1,
                        help="log training status every given number of batches (default: 1)")
    parser.add_argument("--check-every", type=int, default=2,
                        help="save checkpoint every given number of epochs (default: 2)")
    parser.add_argument("--loss", type=str, default='MSE',
                        help="loss function for training (default: MSE)")
    parser.add_argument("--dim3d", type=int, default=32,
                        help="dimension of 3D Grid for spatial domain")
    parser.add_argument("--dim2d", type=int, default=32,
                        help="dimension of 2D Plane for spatial domain")
    parser.add_argument("--dim1d", type=int, default=32,
                        help="dimension of 1D line for parameter domain")
    parser.add_argument("--spatial-fdim", type=int, default=8,
                        help="dimension of feature for spatial domain in feature grids")
    parser.add_argument("--param-fdim", type=int, default=8,
                        help="dimension of feature for parameter domain in feature grids")
    parser.add_argument("--dropout", type=int, default=0,
                        help="using dropout layer in MLP, 0: No, other: Yes (default: 0)")
    parser.add_argument("--equator", action="store_true", default=False,
                        help="compare the equator patch")
    parser.add_argument("--save", action="store_true", default=False,
                        help="save the npy file")
    return parser.parse_args()

def main(args):
    # log hyperparameters
    print(args)
    out_features = 4 if args.loss == 'Evidential' else 1
    network_str = str(args.dim3d) + '_' + str(args.dim2d) + '_' + str(args.dim1d) + '_' + str(args.spatial_fdim) + '_' + str(args.param_fdim)
    if args.loss == 'Evidential':
        network_str += '_Evidential'
    elif args.loss == 'MSE':
        network_str += '_MSE'
    else:
        network_str += '_L1'
    if args.dropout != 0:
        network_str += '_dp'
    network_str += '_mpaso'

    device = pytorch_device_config()

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Generate parameter combinations
    param_ranges = np.arange(0.0, 1.0 + 1e-4, 0.1)
    param_combinations = list(itertools.product(param_ranges, repeat=2))

    coords = np.load(os.path.join(args.root, "sphereCoord.npy"))
    coords = coords.astype(np.float32)

    lat_min, lat_max = -np.pi / 2, np.pi / 2
    coords[:,0] = (coords[:,0] - (lat_min + lat_max) / 2.0) / ((lat_max - lat_min) / 2.0)
    lon_min, lon_max = 0.0, np.pi * 2
    coords[:,1] = (coords[:,1] - (lon_min + lon_max) / 2.0) / ((lon_max - lon_min) / 2.0)
    depth_min, depth_max = 0.0, np.max(coords[:,2])
    coords[:,2] = (coords[:,2] - (depth_min + depth_max) / 2.0) / ((depth_max - depth_min) / 2.0)

    #####################################################################################

    feature_grid_shape = np.concatenate((np.ones(3, dtype=np.int32)*args.dim3d, np.ones(3, dtype=np.int32)*args.dim2d, np.ones(2, dtype=np.int32)*args.dim1d))
    if args.dropout != 0:
        inr_fg = INR_FG(feature_grid_shape, args.spatial_fdim, args.spatial_fdim, args.param_fdim, out_features, True)
    else:
        inr_fg = INR_FG(feature_grid_shape, args.spatial_fdim, args.spatial_fdim, args.param_fdim, out_features, False)
    inr_fg.load_state_dict(torch.load(os.path.join(args.dir_weights, "fg_model_" + network_str + '_'+ str(args.start_epoch) + ".pth")))
    inr_fg.eval()
    print(inr_fg)
    inr_fg.to(device)

    coords_torch = torch.from_numpy(coords)
    equator = np.load(os.path.join(args.root, "equator_patch.npy"))

    sum_pred_sigma = np.zeros((len(param_ranges), len(param_ranges)))
    sum_pred_var = np.zeros((len(param_ranges), len(param_ranges)))

    with torch.no_grad():
        for idx, params in enumerate(param_combinations):
            pred_mu, pred_sigma, pred_var = None, None, None
            
            params = np.array(params, dtype=np.float32).reshape(1, 2)
            params = torch.from_numpy(params)
            params_batch = params.repeat(args.batch_size, 1)
            params_batch = params_batch.to(device)

            tstart = time.time()
            num_batches = math.ceil(len(coords) / args.batch_size)

            for field_idx in range(num_batches):
                coord_batch = coords_torch[field_idx*args.batch_size:(field_idx+1)*args.batch_size]
                coord_batch = coord_batch.to(device)
                # ===================forward=====================
                model_output = inr_fg(torch.cat((coord_batch, params_batch), 1))
                model_output = model_output.cpu().numpy().astype(np.float32)
                gamma, v, alpha, beta = np.split(model_output, 4, axis=-1)
                mu = gamma[:, 0]
                sigma = np.sqrt(beta / (alpha - 1))[:, 0]
                var = np.sqrt(beta / (v * (alpha - 1)))[:, 0]
                if pred_mu is None:
                    pred_mu, pred_sigma, pred_var = mu, sigma, var
                else:
                    pred_mu = np.concatenate((pred_mu, mu), dtype=np.float32)
                    pred_sigma = np.concatenate((pred_sigma, sigma), dtype=np.float32)
                    pred_var = np.concatenate((pred_var, var), dtype=np.float32)
            i, j = idx // len(param_ranges), idx % len(param_ranges)
            sum_pred_sigma[i, j] = pred_sigma.sum()
            sum_pred_var[i, j] = pred_var.sum()
            tend = time.time()
            
            print('Parameters:', params, '\tpred_sigma', np.sum(pred_sigma), '\tpred_var:', np.sum(pred_var))
            print('Inference time: {0:.4f}'.format(tend-tstart))

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Plot heatmap for sum of predicted sigma
        im1 = axs[0].imshow(sum_pred_sigma, cmap='plasma', extent=[0, 1, 1, 0])
        axs[0].set_title('Aleatoric Uncertainty')
        axs[0].set_xlabel('CbrN')
        axs[0].set_ylabel('BwsA')
        axs[0].set_xticks(param_ranges)
        axs[0].set_yticks(param_ranges)
        fig.colorbar(im1, ax=axs[0])

        # Plot heatmap for sum of predicted variance
        im2 = axs[1].imshow(sum_pred_var, cmap='viridis', extent=[0, 1, 1, 0])
        axs[1].set_title('Epistemic Uncertainty')
        axs[1].set_xlabel('CbrN')
        axs[1].set_ylabel('BwsA')
        axs[1].set_xticks(param_ranges)
        axs[1].set_yticks(param_ranges)
        fig.colorbar(im2, ax=axs[1])

        # Show the plots
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    main(parse_args())
    