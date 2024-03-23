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
    parser.add_argument("--sp-sr", type=float, default=0.3,
                        help="simulation parameter sampling rate (default: 0.2)")
    parser.add_argument("--sf-sr", type=float, default=0.05,
                        help="scalar field sampling rate (default: 0.02)")
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
    parser.add_argument("--fg-version", type=int, default=2,
                        help="feature grid version")
    parser.add_argument("--equator", action="store_true", default=False,
                        help="compare the equator patch")
    parser.add_argument("--save", action="store_true", default=False,
                        help="save the npy file")
    return parser.parse_args()

def main(args):
    # log hyperparameters
    print(args)
    out_features = 1
    network_str = str(args.dim3d) + '_' + str(args.dim2d) + '_' + str(args.dim1d) + '_' + str(args.spatial_fdim) + '_' + str(args.param_fdim) + '_v' +str(args.fg_version)
    if args.loss == 'MSE':
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

    fh = open(os.path.join(args.root, "test", "names.txt"))
    filenames = []
    for line in fh:
        filenames.append(line.replace("\n", ""))

    params_arr = np.load(os.path.join(args.root, "test/params.npy"))
    coords = np.load(os.path.join(args.root, "sphereCoord.npy"))
    coords = coords.astype(np.float32)
    data_dicts = []
    for idx in range(len(filenames)):
        # params min [0.0, 300.0, 0.25, 100.0, 1]
        #        max [5.0, 1500.0, 1.0, 300.0, 384]
        params = np.array(params_arr[idx][1:])
        params = (params.astype(np.float32) - np.array([0.0, 300.0, 0.25, 100.0], dtype=np.float32)) / \
                 np.array([5.0, 1200.0, .75, 200.0], dtype=np.float32)
        d = {'file_src': os.path.join(args.root, "test", filenames[idx]), 'params': params}
        data_dicts.append(d)

    lat_min, lat_max = -np.pi / 2, np.pi / 2
    coords[:,0] = (coords[:,0] - (lat_min + lat_max) / 2.0) / ((lat_max - lat_min) / 2.0)
    lon_min, lon_max = 0.0, np.pi * 2
    coords[:,1] = (coords[:,1] - (lon_min + lon_max) / 2.0) / ((lon_max - lon_min) / 2.0)
    depth_min, depth_max = 0.0, np.max(coords[:,2])
    coords[:,2] = (coords[:,2] - (depth_min + depth_max) / 2.0) / ((depth_max - depth_min) / 2.0)

    #####################################################################################

    feature_grid_shape = np.concatenate((np.ones(3, dtype=np.int32)*args.dim3d, np.ones(3, dtype=np.int32)*args.dim2d, np.ones(3, dtype=np.int32)*args.dim1d))
    if args.dropout != 0:
        inr_fg = INR_FG(feature_grid_shape, args.spatial_fdim, args.spatial_fdim, args.param_fdim, out_features, args.fg_version, True)
    else:
        inr_fg = INR_FG(feature_grid_shape, args.spatial_fdim, args.spatial_fdim, args.param_fdim, out_features, args.fg_version, False)
    inr_fg.load_state_dict(torch.load(os.path.join(args.dir_weights, "fg_model_" + network_str + '_'+ str(args.start_epoch) + ".pth")))
    inr_fg.eval()
    print(inr_fg)
    inr_fg.to(device)

    if args.loss == 'MSE':
        print('Use MSE Loss')
        criterion = torch.nn.MSELoss()
    elif args.loss == 'L1':
        print('Use L1 Loss')
        criterion = torch.nn.L1Loss()
    else:
        print('Use L1 Loss')
        criterion = torch.nn.L1Loss()

    dmin = -1.93
    dmax = 30.36
    psnrs = []
    max_diff = np.zeros(len(data_dicts))
    coords_torch = torch.from_numpy(coords)
    equator = np.load(os.path.join(args.root, "equator_patch.npy"))
    total_mse = 0.

    with torch.no_grad():
        for param_idx in range(len(data_dicts)):
            pred = None
            
            params = data_dicts[param_idx]['params'].reshape(1,4)
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
                model_output = model_output.cpu().numpy().flatten().astype(np.float32)
                if pred is None:
                    pred = model_output
                else:
                    pred = np.concatenate((pred, model_output), dtype=np.float32)
            tend = time.time()

            gt = ReadMPASOScalar(data_dicts[param_idx]['file_src'])
            pred = pred * (dmax-dmin) + dmin
            if args.equator:
                gt = gt * equator
                gt = gt[abs(gt) > 0]
                pred = pred * equator
                pred = pred[abs(pred) > 0]
            diff = abs(gt - pred)
            max_diff[param_idx] = diff.max()
            mse = np.mean((gt - pred)**2)
            total_mse += mse
            if args.equator: 
                psnr = 20. * np.log10(29.50 - 11.00) - 10. * np.log10(mse)
            else:
                psnr = 20. * np.log10(dmax - dmin) - 10. * np.log10(mse)
            psnrs.append(psnr)
            print('Inference time: {0:.4f} , data: {1}'.format(tend-tstart, data_dicts[param_idx]['file_src']))
            print('PSNR = {0:.4f}, MSE = {1:.4f}'.format(psnr, mse))
            if args.save:
                pred.astype(np.float64).tofile(args.dir_outputs + network_str + '_' + filenames[param_idx][:filenames[param_idx].rfind('.')] + '.bin')
        if args.equator: 
            print('<<<<<<<  PSNR = {0:.4f} >>>>>>>>>>'.format(20. * np.log10(29.50 - 11.00) -
                    10. * np.log10(total_mse / len(data_dicts))))
            print('<<<<<<<  Max Diff = {0:.4f} >>>>>>>>>>'.format(max_diff.mean() / (29.50 - 11.00)))
        else:
            print('<<<<<<<  PSNR = {0:.4f} >>>>>>>>>>'.format(20. * np.log10(dmax - dmin) -
                    10. * np.log10(total_mse / len(data_dicts))))
            print('<<<<<<<  Max Diff = {0:.4f} >>>>>>>>>>'.format(max_diff.mean() / (dmax - dmin)))
    

if __name__ == '__main__':
    main(parse_args())
    