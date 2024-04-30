from models import *
from utils import *
import loss_helper
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
    return parser.parse_args()

def main(args):
    # log hyperparameters
    print(args)
    out_features = 4 if args.loss == 'Evidential' else 1
    nEnsemble = 4
    data_size = 512 ** 3
    num_sf_batches = math.ceil(nEnsemble * data_size * args.sf_sr / args.batch_size)
    network_str = str(args.dim3d) + '_' + str(args.dim2d) + '_' + str(args.dim1d) + '_' + str(args.spatial_fdim) + '_' + str(args.param_fdim)
    if args.loss == 'Evidential':
        network_str += '_Evidential'
    elif args.loss == 'MSE':
        network_str += '_MSE'
    else:
        network_str += '_L1'

    if args.dropout != 0:
        network_str += '_dp'

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
            params_np = (params_np - np.array([0.12, 0.0215, 0.55], dtype=np.float32)) / np.array([0.035, 0.002, 0.3], dtype=np.float32)
            d = {'file_src': os.path.join(args.root, files[fidx]), 'params': params_np}
            training_dicts.append(d)

    xcoords, ycoords, zcoords = np.linspace(-1,1,512), np.linspace(-1,1,512), np.linspace(-1,1,512)
    xv, yv, zv = np.meshgrid(xcoords, ycoords, zcoords, indexing='ij')
    xv, yv, zv = xv.flatten().reshape(-1,1), yv.flatten().reshape(-1,1), zv.flatten().reshape(-1,1)
    coords = np.hstack((xv, yv, zv)).astype(np.float32)

    #####################################################################################

    feature_grid_shape = np.concatenate((np.ones(3, dtype=np.int32)*args.dim3d, np.ones(3, dtype=np.int32)*args.dim2d, np.ones(3, dtype=np.int32)*args.dim1d))
    if args.dropout != 0:
        inr_fg = INR_FG(feature_grid_shape, args.spatial_fdim, args.spatial_fdim, args.param_fdim, out_features, True)
    else:
        inr_fg = INR_FG(feature_grid_shape, args.spatial_fdim, args.spatial_fdim, args.param_fdim, out_features, False)
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

    #####################################################################################

    losses = []

    dmin = 8.6
    dmax = 13.6
    num_bins = 10
    cluster_size = data_size // np.power(2, 24) # for multinomial, number of categories cannot exceed 2^24
    bin_width = 1.0 / num_bins
    max_binidx_f = float(num_bins-1)
    batch_size_per_field = args.batch_size // nEnsemble
    nEnsembleGroups_per_epoch = len(training_dicts) // nEnsemble

    coords_torch = torch.from_numpy(coords)

    # preprocessing: load all data and compute importance
    # Only compute once and data is already saved
    # nBlocks = 16
    # block_size = data_size // nBlocks
    # all_freq = None
    # for tdidx, td in enumerate(training_dicts):
    #     sf = ReadScalarBinary(td['file_src'])
    #     sf = (sf-dmin) / (dmax-dmin)
    #     sf = torch.from_numpy(sf)
    #     curr_freq = None
    #     for bidx in range(nBlocks):
    #         block_freq = torch.histc(sf[bidx*block_size:(bidx+1)*block_size], bins=num_bins, min=0.0, max=1.0).type(torch.long)
    #         if curr_freq is None:
    #             curr_freq = block_freq
    #         else:
    #             curr_freq += block_freq
    #     if all_freq is None:
    #         all_freq = curr_freq
    #     else:
    #         all_freq += curr_freq
    #     print('tdidx: ', tdidx, '  ', torch.sum(all_freq) / 512**3)
    # all_freq = all_freq.type(torch.double)
    # importance = 1. / all_freq
    # sfimps = torch.zeros(len(training_dicts))
    # for tdidx, td in enumerate(training_dicts):
    #     sf = ReadScalarBinary(td['file_src'])
    #     sf = (sf-dmin) / (dmax-dmin)
    #     sf = torch.from_numpy(sf)
    #     curr_impidx = torch.clamp(sf / bin_width, min=0.0, max=max_binidx_f).type(torch.long)
    #     curr_sfimp = importance[curr_impidx].sum()
    #     sfimps[tdidx] = curr_sfimp
    #     print('tdidx: ', tdidx, '  ', curr_sfimp)
    # sfimps = sfimps / sfimps.sum()
    # sfimps_np = sfimps.numpy()
    # np.save(args.dir_outputs + 'ensemble_member_importances', sfimps_np)

    sfimps_np = np.load('ensemble_member_importances.npy')
    sfimps = torch.from_numpy(sfimps_np)

    #####################################################################################

    def imp_func(data, minval, maxval, bw, maxidx):
        freq = None
        nBlocks = 16
        block_size = data_size // nBlocks
        for bidx in range(nBlocks):
            block_freq = torch.histc(data[bidx*block_size:(bidx+1)*block_size], bins=num_bins, min=minval, max=maxval).type(torch.long)
            if freq is None:
                freq = block_freq
            else:
                freq += block_freq
        freq = freq.type(torch.double)
        importance = 1. / freq
        importance_idx = torch.clamp((data - minval) / bw, min=0.0, max=maxidx).type(torch.long)
        return importance, importance_idx

    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        print('epoch {0}'.format(epoch+1))
        total_loss = 0
        total_mse = 0
        e_rndidx = torch.multinomial(sfimps, nEnsembleGroups_per_epoch * nEnsemble, replacement=True)
        for egidx in range(nEnsembleGroups_per_epoch):
            tstart = time.time()
            scalar_fields = []
            sample_weights_arr = []
            params_batch = None
            errsum = 0
            # Load and compute importance map
            for eidx in range(nEnsemble):
                curr_scalar_field = ReadScalarBinary(training_dicts[e_rndidx[egidx*nEnsemble + eidx]]['file_src'])
                curr_scalar_field = (curr_scalar_field-dmin) / (dmax-dmin)
                curr_scalar_field = torch.from_numpy(curr_scalar_field)
                curr_params = training_dicts[e_rndidx[egidx*nEnsemble + eidx]]['params'].reshape(1,3)
                curr_params = torch.from_numpy(curr_params)
                curr_params_batch = curr_params.repeat(batch_size_per_field, 1)
                if params_batch is None:
                    params_batch = curr_params_batch
                else:
                    params_batch = torch.cat((params_batch, curr_params_batch), 0)
                curr_imp, curr_impidx = imp_func(curr_scalar_field, 0.0, 1.0, bin_width, max_binidx_f)
                curr_sample_weights = curr_imp[curr_impidx].reshape(-1, cluster_size).sum(1)
                
                scalar_fields.append(curr_scalar_field)
                sample_weights_arr.append(curr_sample_weights)
            params_batch = params_batch.to(device)
            # Train
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
                coord_batch = coord_batch.to(device)
                value_batch = value_batch.to(device)
                # ===================forward=====================
                model_output = inr_fg(torch.cat((coord_batch, params_batch), 1))
                if args.loss == 'Evidential':
                    loss = loss_helper.EvidentialRegression(value_batch, model_output, coeff=1e-2)
                    gamma, _, _, _ = torch.chunk(model_output, 4, dim=-1) 
                    mse = torch.mean((gamma - value_batch) ** 2)
                else:
                    loss = criterion(model_output, value_batch)
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_mse = mse.item() if args.loss == 'Evidential' else loss.data.cpu().numpy()
                errsum += batch_mse * nEnsemble * batch_size_per_field
                total_loss += loss.data.cpu().numpy()
                total_mse += batch_mse
            tend = time.time()
            curr_mse = errsum / (nEnsemble * batch_size_per_field * num_sf_batches)
            curr_psnr = - 10. * np.log10(curr_mse)
            print('Training time: {0:.4f} for {1} data points x {2} batches, approx PSNR = {3:.4f}'\
                  .format(tend-tstart, nEnsemble * batch_size_per_field, num_sf_batches, curr_psnr))
        losses.append(total_loss)
        if (epoch+1) % args.log_every == 0:
            print('epoch {0}, loss = {1}'.format(epoch+1, total_loss))
            print("====> Epoch: {0} Average {1} loss: {2:.4f}".format(epoch+1, args.loss, total_loss / len(training_dicts)))
            plt.plot(losses)

            plt.savefig(args.dir_outputs + 'fg_inr_loss_' + network_str + '.jpg')
            plt.clf()

        if (epoch+1) % args.check_every == 0:
            print("=> saving checkpoint at epoch {}".format(epoch + 1))
            torch.save(inr_fg.state_dict(),
                       os.path.join(args.dir_weights, "fg_model_" + network_str + '_'+ str(epoch+1) + ".pth"))

if __name__ == '__main__':
    main(parse_args())
    