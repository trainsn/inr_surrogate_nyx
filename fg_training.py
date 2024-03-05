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
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="learning rate (default: 5e-5)")
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
    return parser.parse_args()

def main(args):
    # log hyperparameters
    print(args)
    scalar_kwargs = {"num_workers": 2, "pin_memory": True}
    feature_dim = 64
    out_features = 1
    nEnsemble = 4
    data_size = 512**3
    num_sf_batches = math.ceil(nEnsemble * data_size * args.sf_sr / args.batch_size)
    num_sp_sampling = math.ceil(100 * args.sp_sr)
    network_str = str(args.dim3d) + '_' + str(args.dim2d) + '_' + str(args.dim1d)

    device = pytorch_device_config()

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    files = os.listdir(args.root)
    files = sorted(files)
    training_dicts = []
    for fidx in range(len(files)):
        if files[fidx].endswith('bin'):
            sps = files[fidx].split('_')
            # params min [0.12, 0.0215, 0.55]
            #        max [0.155, 0.0235, 0.85]
            params_np = np.array([float(sps[1]), float(sps[2]), float(sps[3][:-4])], dtype=np.float32)
            params_np = (params_np - np.array([0.1375, 0.0225, 0.7], dtype=np.float32)) / np.array([0.0175, 0.001, 0.15], dtype=np.float32)
            d = {'file_src': os.path.join(args.root, files[fidx]), 'params': params_np}
            training_dicts.append(d)

    xcoords, ycoords, zcoords = np.linspace(-1,1,512), np.linspace(-1,1,512), np.linspace(-1,1,512)
    xv, yv, zv = np.meshgrid(xcoords, ycoords, zcoords, indexing='ij')
    xv, yv, zv = xv.flatten().reshape(-1,1), yv.flatten().reshape(-1,1), zv.flatten().reshape(-1,1)
    coords = np.hstack((xv, yv, zv)).astype(np.float32)

    ensembleParam_dataset = EnsumbleParamDataset(training_dicts)
    ensembleParam_dataloader = DataLoader(ensembleParam_dataset, batch_size=nEnsemble, shuffle=True, num_workers=2)

    #####################################################################################

    feature_grid_shape = np.concatenate((np.ones(3, dtype=np.int32)*args.dim3d, np.ones(3, dtype=np.int32)*args.dim2d, np.ones(3, dtype=np.int32)*args.dim1d))
    inr_fg = INR_FG(feature_grid_shape, feature_dim, feature_dim, feature_dim, out_features)
    if args.start_epoch > 0:
        inr_fg.load_state_dict(torch.load(os.path.join(args.dir_weights, "fg_model_" + network_str + '_'+ str(args.start_epoch) + ".pth")))
    print(inr_fg)
    inr_fg.to(device)

    optimizer = torch.optim.Adam(inr_fg.parameters(), lr=args.lr)
    if args.loss == 'MSE':
        print('Use MSE Loss')
        criterion = torch.nn.MSELoss()
    elif args.loss == 'L1':
        print('Use L1 Loss')
        criterion = torch.nn.L1Loss()
    else:
        print('Use L1 Loss')
        criterion = torch.nn.L1Loss()

    pdb.set_trace()
    dmin_vdl = 8.6
    dmax_vdl = 13.6
    losses = []
    num_bins = 10
    cluster_size = data_size // np.power(2, 24) # for multinomial, number of categories cannot exceed 2^24
    coords_torch = torch.from_numpy(coords)
    batch_size_per_field = args.batch_size // nEnsemble

    def imp_func(data):
        freq = torch.histc(data, bins=num_bins, min=-1., max=1.)
        importance = 1. / freq
        importance_idx = ((data + 1) / 2. * num_bins).type(torch.long)
        return importance, importance_idx

    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        print('epoch {0}'.format(epoch+1))
        total_loss = 0
        for param_idx, ensembleParam_dict in enumerate(ensembleParam_dataloader):
            if param_idx*nEnsemble >= num_sp_sampling:
                break
            tstart = time.time()
            scalar_fields = []
            sample_weights_arr = []
            params_batch = None
            curr_batch_loss = 0
            for eidx in range(nEnsemble):
                curr_scalar_field = ReadScalarBinary(ensembleParam_dict['file_src'][eidx])
                curr_scalar_field = (curr_scalar_field - (dmax_vdl + dmin_vdl) / 2.) / ((dmax_vdl - dmin_vdl) / 2.)
                curr_scalar_field = torch.from_numpy(curr_scalar_field)
                scalar_fields.append(curr_scalar_field)
                curr_params = ensembleParam_dict['params'][eidx].reshape(1,3)
                curr_params_batch = curr_params.repeat(batch_size_per_field, 1)
                if params_batch is None:
                    params_batch = curr_params_batch
                else:
                    params_batch = torch.cat((params_batch, curr_params_batch), 0)
                curr_imp, curr_impidx = imp_func(curr_scalar_field)
                curr_sample_weights = curr_imp[curr_impidx].reshape(-1, cluster_size).sum(1)
                sample_weights_arr.append(curr_sample_weights)
            params_batch = torch.autograd.Variable(params_batch).to(device)
            for field_idx in range(num_sf_batches):
                coord_batch = None
                value_batch = None
                for eidx in range(nEnsemble):
                    #####
                    rnd_idx = torch.multinomial(sample_weights_arr[eidx], batch_size_per_field, replacement=True)
                    rnd_idx = rnd_idx * cluster_size + torch.randint(high=cluster_size, size=rnd_idx.shape)
                    ######
                    if coord_batch is None:
                        coord_batch, value_batch = coords_torch[rnd_idx], scalar_fields[eidx][rnd_idx]
                    else:
                        coord_batch, value_batch = torch.cat((coord_batch, coords_torch[rnd_idx]), 0), torch.cat((value_batch, scalar_fields[eidx][rnd_idx]), 0)
                value_batch = value_batch.reshape(len(value_batch), 1)
                coord_batch = torch.autograd.Variable(coord_batch).to(device)
                value_batch = torch.autograd.Variable(value_batch).to(device)
                # ===================forward=====================
                pdb.set_trace()
                model_output = inr_fg(torch.cat((coord_batch, params_batch), 1))
                loss = criterion(model_output, value_batch)
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.data
                curr_batch_loss += loss.data
            tend = time.time()
            print('Training time: {0:.4f} for {1} x {2} data points, loss = {3:.4f}'\
                  .format(tend-tstart, args.batch_size, field_idx, curr_batch_loss))
        total_loss = total_loss.cpu().numpy()
        losses.append(total_loss)
        if (epoch+1) % args.log_every == 0:
            print('epoch {0}, loss = {1}'.format(epoch+1, total_loss))
            print("====> Epoch: {0} Average {1} loss: {2:.4f}".format(epoch+1, args.loss, total_loss / num_sp_sampling))
            plt.plot(losses)

            plt.savefig(args.dir_outputs + 'fg_inr_loss_' + network_str + '.jpg')
            plt.clf()

        if (epoch+1) % args.check_every == 0:
            print("=> saving checkpoint at epoch {}".format(epoch + 1))
            torch.save(inr_fg.state_dict(),
                       os.path.join(args.dir_weights, "fg_model_" + network_str + '_'+ str(epoch+1) + ".pth"))

if __name__ == '__main__':
    main(parse_args())
    