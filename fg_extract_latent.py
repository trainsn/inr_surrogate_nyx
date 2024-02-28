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
    parser.add_argument("--feature_dim", type=int, default=8,
                        help="dimension of feature in feature grids")
    parser.add_argument("--dropout", type=int, default=0,
                        help="using dropout layer in MLP, 0: No, other: Yes (default: 0)")
    return parser.parse_args()

def main(args):
    # log hyperparameters
    print(args)
    scalar_kwargs = {"num_workers": 2, "pin_memory": True}
    out_features = 1
    network_str = str(args.dim3d) + '_' + str(args.dim2d) + '_' + str(args.dim1d) + '_' + str(args.feature_dim)
    if args.dropout != 0:
        network_str += '_dp'

    device = pytorch_device_config()

    #####################################################################################

    feature_grid_shape = np.concatenate((np.ones(3, dtype=np.int32)*args.dim3d, np.ones(3, dtype=np.int32)*args.dim2d, np.ones(3, dtype=np.int32)*args.dim1d))
    if args.dropout != 0:
        inr_fg = INR_FG_DP(feature_grid_shape, args.feature_dim, args.feature_dim, args.feature_dim, out_features)
    else:
        inr_fg = INR_FG(feature_grid_shape, args.feature_dim, args.feature_dim, args.feature_dim, out_features)
    inr_fg.load_state_dict(torch.load(os.path.join(args.dir_weights, "fg_model_" + network_str + '_'+ str(args.start_epoch) + ".pth")))
    inr_fg.eval()
    print(inr_fg)
    inr_fg.to(device)

    test_coord = np.array([-0.1, 0.7, -0.5], dtype=np.float32)
    test_param = np.array([0.2, 0.3, 0.4], dtype=np.float32)
    test_coord = torch.autograd.Variable(torch.from_numpy(test_coord)).to(device)
    test_param = torch.autograd.Variable(torch.from_numpy(test_param)).to(device)

    inputs = torch.cat((test_coord, test_param)).reshape(1, len(test_coord)+len(test_param))

    with torch.no_grad():
        fg_output, coord_features, param_features = inr_fg.forwardFGOnly(inputs)
    print(fg_output)
    print(coord_features)
    print(param_features)
    

if __name__ == '__main__':
    main(parse_args())
    