import torch
import numpy as np
from torch.utils.data import Dataset
from utils import *
from itertools import combinations

class EnsumbleParamDataset(Dataset):
    def __init__(self, params:list):
        ''' Input: param 1,2,3 and x,y,z '''
        self.params = params

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        return self.params[idx]
    

class ScalarDataset(Dataset):
    def __init__(self, coords:np.array, scalar_field_src:str):
        ''' Input: param 1,2,3 and x,y,z '''
        self.coords = coords
        self.scalar_field = torch.from_numpy(ReadScalarBinary(scalar_field_src))

    def getScalarField(self):
        return self.scalar_field

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.coords[idx], self.scalar_field[idx]
    


###########################################################################################################


class SineLayer(torch.nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate
    
    
class Siren(torch.nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = torch.nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = torch.nn.Sequential(*self.net)
    
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords

class ResidualSineLayer(torch.nn.Module):
    def __init__(self, features, bias=True, ave_first=False, ave_second=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0

        self.features = features
        self.linear_1 = torch.nn.Linear(features, features, bias=bias)
        self.linear_2 = torch.nn.Linear(features, features, bias=bias)

        self.weight_1 = .5 if ave_first else 1
        self.weight_2 = .5 if ave_second else 1

        self.init_weights()
    #

    def init_weights(self):
        with torch.no_grad():
            self.linear_1.weight.uniform_(-np.sqrt(6 / self.features) / self.omega_0, 
                                           np.sqrt(6 / self.features) / self.omega_0)
            self.linear_2.weight.uniform_(-np.sqrt(6 / self.features) / self.omega_0, 
                                           np.sqrt(6 / self.features) / self.omega_0)

    def forward(self, input):
        sine_1 = torch.sin(self.omega_0 * self.linear_1(self.weight_1*input))
        sine_2 = torch.sin(self.omega_0 * self.linear_2(sine_1))
        return self.weight_2*(input+sine_2)



###############################################################################################################
    
class Siren_Residual_Surrogate(torch.nn.Module):
    '''
    in_features are 6 dimensions: hyper parameter * 3 and x y z
    '''
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, dropout=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.fourier_feature = []
        self.fourier_feature.append(SineLayer(3, hidden_features-3, is_first=True, omega_0=first_omega_0))
        self.fourier_feature = torch.nn.Sequential(*self.fourier_feature)

        for i in range(hidden_layers):
            self.net.append(ResidualSineLayer(hidden_features, bias=True, ave_first=(i>0), ave_second=(i==(hidden_layers-1))))

        if dropout:
            self.net.append(torch.nn.Dropout(p=0.2))
        if outermost_linear:
            final_linear = torch.nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = torch.nn.Sequential(*self.net)
    
    def forward(self, params, coords):
        # params = params.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        # coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        coords = self.fourier_feature(coords)
        inputs = torch.cat((params, coords), 1)
        output = self.net(inputs)
        return output

###############################################################################################################

class Siren_Surrogate(torch.nn.Module):
    '''
    in_features are 6 dimensions: hyper parameter * 3 and x y z
    '''
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.fourier_feature = []
        self.fourier_feature.append(SineLayer(3, hidden_features-3, is_first=True, omega_0=first_omega_0))
        self.fourier_feature = torch.nn.Sequential(*self.fourier_feature)

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = torch.nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = torch.nn.Sequential(*self.net)
    
    def forward(self, params, coords):
        # params = params.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        # coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        coords = self.fourier_feature(coords)
        inputs = torch.cat((params, coords), 1)
        output = self.net(inputs)
        return output
    
class PosEncoding(torch.nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''
    def __init__(self, num_frequencies=10):
        super().__init__()
        self.num_frequencies = num_frequencies

    def forward(self, coords):
        coords_pos_enc = coords
        in_features = coords.shape[-1]

        for i in range(self.num_frequencies):
            for j in range(in_features):
                c = coords[..., j]
                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)
                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)
        return coords_pos_enc

class SnakeAlt(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return (input + 1. - torch.cos(2. * input)) / 2.


class Inr_Surrogate(torch.nn.Module):
    '''
    in_features are 6 dimensions: hyper parameter * 3 and x y z
    '''
    def __init__(self, dsp=3, ch=64, num_frequencies=10):
        super().__init__()
        # dsp  - dimensions of the simulation parameters
        # ch   - channel multiplier
        # num_frequencies  -  number of frequencies for positional encoding
        self.dsp = dsp
        self.num_frequencies = num_frequencies
        self.in_features = 3
        self.ch = ch

        self.pos_encoding = PosEncoding(self.num_frequencies)
        self.pos_subnet = torch.nn.Sequential(
            torch.nn.Linear(self.in_features + 2 * self.in_features * self.num_frequencies + self.dsp, ch * 2), SnakeAlt(),
            torch.nn.Linear(ch * 2, ch * 2), SnakeAlt(),
            torch.nn.Linear(ch * 2, ch * 2), SnakeAlt(),
            torch.nn.Linear(ch * 2, ch * 2), SnakeAlt(),
            torch.nn.Linear(ch * 2, 1)
        )
    
    def forward(self, sp, pos):
        pos = self.pos_encoding(pos)
        fc_input = torch.cat((sp, pos), 1)
        output = self.pos_subnet(fc_input)
        return output

###############################################################################################################

# dense feature grid initialize with Uniform(-1e-4, 1e-4) as instant-NGP paper
class FeatureGrid_SP(torch.nn.Module):
    '''
    a regular feature grid with grid_shape (dim0, dim1, ..., dimn)
    '''
    def __init__(self, grid_shape, num_feat:int) -> None:
        super().__init__()
        self.grid_shape = grid_shape
        self.num_feat = num_feat
        self.feature_grid = torch.nn.Parameter(
            torch.Tensor(1, num_feat, *reversed(grid_shape)),
            requires_grad=True
        )
        # initialize with Uniform(-1e-4, 1e-4)
        torch.nn.init.uniform_(self.feature_grid, a=-0.0001, b=0.0001)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        input: (Batch, 3)
        output: (Batch, num_feat)
        '''
        grid = self.feature_grid
        feats = torch.nn.functional.grid_sample(torch.nn.functional.dropout(grid, self.dropout),
                            x.reshape(([1]*x.shape[-1]) + list(x.shape)),
                            mode='bilinear', align_corners=True)
        return feats.squeeze().permute(1, 0)
    
###############################################################################################################

class DecompGrid(torch.nn.Module):
    '''
    grid_shape: [x_3d, y_3d, z_3d, x_2d, y_2d, z_2d, ..._2d]
    '''
    def __init__(self, grid_shape, num_feat_3d, num_feat_2d,
                 a_3d=-0.001, b_3d=0.001,
                 a_2d=0.9, b_2d=1.1) -> None:
        super().__init__()
        assert num_feat_2d == num_feat_3d
        
        self.grid_shape = grid_shape
        self.num_feat_3d = num_feat_3d
        self.num_feat_2d = num_feat_2d
        self.feature_grid_3d = torch.nn.Parameter(
            torch.Tensor(1, num_feat_3d, *reversed(grid_shape[:3])),
            requires_grad=True
        )
        torch.nn.init.uniform_(self.feature_grid_3d, a=-0.001, b=0.001)
        
        self.plane_dimid = list(combinations(range(len(grid_shape[3:])), 2))
        self.plane_dims = list(combinations(grid_shape[3:], 2))
        self.planes = []
        print(self.plane_dimid, self.plane_dims)
        for i, dims in enumerate(self.plane_dims):
            plane = torch.nn.Parameter(
                torch.Tensor(1, num_feat_2d, *reversed(dims)),
                requires_grad=True
            )
            torch.nn.init.uniform_(plane, a=0.9, b=1.1)
            self.planes.append(plane)
        self.planes = torch.nn.ParameterList(self.planes)
        print('DecompGrid 2d shapes:', grid_shape[3:], 'plane_dimid', self.plane_dimid, 'dims', self.plane_dims)
        
        # initialize with Uniform(-1e-4, 1e-4)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        input: (Batch, Ndim)
        output: (Batch, num_feat_3d/2d)
        '''
        coords = x[..., :3]
        feats_3d = torch.nn.functional.grid_sample(self.feature_grid_3d,
                            coords.reshape(([1]*coords.shape[-1]) + list(coords.shape)),
                            mode='bilinear', align_corners=True)
        # print(feats_3d.shape)
        # feats_3d = feats_3d.squeeze().permute(1, 0)
        feats_3d = feats_3d.squeeze()
        # feats_3d = torch.ones((x.shape[0], self.num_feat_3d), device='cuda')
        for i, dimids in enumerate(self.plane_dimid):
            x2d = x[:,dimids]
            # print(i, dimids, x[:2], x2d[:2])
            f2d = torch.nn.functional.grid_sample(self.planes[i],
                            x2d.reshape(([1]*x2d.shape[-1]) + list(x2d.shape)),
                            mode='bilinear', align_corners=True)
            # print(dimids, x2d.shape, self.planes[i].shape, x2d.reshape(([1]*x2d.shape[-1]) + list(x2d.shape)).shape, f2d.shape)
            # f2d = f2d.squeeze().permute(1, 0)
            f2d = f2d.squeeze()
            # feats_2d.append(f2d.squeeze().permute(1, 0))
            feats_3d = feats_3d * f2d
        # feats_3d = feats_3d + sum(feats_2d)
        return feats_3d



class DecompGridv2(torch.nn.Module):
    '''
    grid_shape: [x_3d, y_3d, z_3d, x_2d, y_2d, z_2d, ..._2d]
    '''
    def __init__(self, grid_shape, num_feat_3d, num_feat_2d, num_feat_1d,
                 a_3d=-0.001, b_3d=0.001,
                 a_2d=0.9, b_2d=1.1) -> None:
        super().__init__()
        assert num_feat_2d == num_feat_3d
        assert num_feat_1d == num_feat_3d
        
        self.grid_shape = grid_shape
        self.num_feat_3d = num_feat_3d
        self.num_feat_2d = num_feat_2d
        self.num_feat_1d = num_feat_1d
        self.feature_grid_3d = torch.nn.Parameter(
            torch.Tensor(1, num_feat_3d, *reversed(grid_shape[:3])),
            requires_grad=True
        )
        torch.nn.init.uniform_(self.feature_grid_3d, a=-0.001, b=0.001)
        
        self.plane_dimid = list(combinations(range(len(grid_shape[3:6])), 2))
        self.plane_dims = list(combinations(grid_shape[3:6], 2))
        self.line_dimid = list(range(3, 3+len(grid_shape[6:])))
        self.line_dims = grid_shape[6:]
        self.planes = []
        self.lines = []
        print('plane dimid', self.plane_dimid)
        print('plane dims', self.plane_dims)
        print('line dimid', self.line_dimid)
        print('line dims', self.line_dims)
        for i, dims in enumerate(self.plane_dims):
            plane = torch.nn.Parameter(
                torch.Tensor(1, num_feat_2d, *reversed(dims)),
                requires_grad=True
            )
            torch.nn.init.uniform_(plane, a=0.9, b=1.1)
            self.planes.append(plane)

        for i, dim in enumerate(self.line_dims):
            line = torch.nn.Parameter(
                torch.Tensor(num_feat_1d, dim),
                requires_grad=True
            )
            torch.nn.init.uniform_(line, a=0.9, b=1.1)
            self.lines.append(line)
        self.planes = torch.nn.ParameterList(self.planes)
        self.lines = torch.nn.ParameterList(self.lines)
        # print('DecompGrid 2d shapes:', grid_shape[3:], 'plane_dimid', self.plane_dimid, 'dims', self.plane_dims)
        
        # initialize with Uniform(-1e-4, 1e-4)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        input: (Batch, Ndim)
        output: (Batch, num_feat_3d/2d)
        '''
        coords = x[..., :3]
        feats_3d = torch.nn.functional.grid_sample(self.feature_grid_3d,
                            coords.reshape(([1]*coords.shape[-1]) + list(coords.shape)),
                            mode='bilinear', align_corners=True)
        # print(feats_3d.shape)
        # feats_3d = feats_3d.squeeze().permute(1, 0)
        feats_3d = feats_3d.squeeze()
        # feats_3d = torch.ones((x.shape[0], self.num_feat_3d), device='cuda')
        for i, dimids in enumerate(self.plane_dimid):
            x2d = x[:,dimids]
            # print(i, dimids, x, x2d)
            x2d = x2d.reshape(([1]*x2d.shape[-1]) + list(x2d.shape))
            # print(x2d.shape)
            f2d = torch.nn.functional.grid_sample(self.planes[i],
                            x2d,
                            mode='bilinear', align_corners=True)
            # print(dimids, x2d.shape, self.planes[i].shape, x2d.reshape(([1]*x2d.shape[-1]) + list(x2d.shape)).shape, f2d.shape)
            # f2d = f2d.squeeze().permute(1, 0)
            f2d = f2d.squeeze()
            # feats_2d.append(f2d.squeeze().permute(1, 0))
            feats_3d = feats_3d * f2d
        for i, dimids in enumerate(self.line_dimid):
            x1d = x[:,dimids]
            x1dn = x1d*self.line_dims[i]
            x1d_f = torch.floor(x1dn)
            weights = x1dn-x1d_f
            x1d_f = x1d_f
            f1d = torch.lerp(self.lines[i][:,x1d_f.type(torch.int)], self.lines[i][:,torch.clamp(x1d_f+1.0, min=0.0, max=self.line_dims[i]-1).type(torch.int)], weights)
            f1d = f1d.squeeze()
            feats_3d = feats_3d * f1d
        return feats_3d.T
    
    def forwardWithIntermediates(self, x: torch.Tensor) -> torch.Tensor:
        coords = x[..., :3]
        feats_3d = torch.nn.functional.grid_sample(self.feature_grid_3d,
                            coords.reshape(([1]*coords.shape[-1]) + list(coords.shape)),
                            mode='bilinear', align_corners=True)
        feats_3d = feats_3d.squeeze()
        for i, dimids in enumerate(self.plane_dimid):
            x2d = x[:,dimids]
            x2d = x2d.reshape(([1]*x2d.shape[-1]) + list(x2d.shape))
            f2d = torch.nn.functional.grid_sample(self.planes[i],
                            x2d,
                            mode='bilinear', align_corners=True)
            f2d = f2d.squeeze()
            feats_3d = feats_3d * f2d
        feats_coords = feats_3d
        f1ds = []
        for i, dimids in enumerate(self.line_dimid):
            x1d = x[:,dimids]
            x1dn = x1d*self.line_dims[i]
            x1d_f = torch.floor(x1dn)
            weights = x1dn-x1d_f
            x1d_f = x1d_f
            f1d = torch.lerp(self.lines[i][:,x1d_f.type(torch.int)], self.lines[i][:,torch.clamp(x1d_f+1.0, min=0.0, max=self.line_dims[i]-1).type(torch.int)], weights)
            f1d = f1d.squeeze()
            feats_3d = feats_3d * f1d
            f1ds.append(f1d.T)
        return feats_3d.T, feats_coords.T, f1ds

class DecompGridv3(torch.nn.Module):
    '''
    grid_shape: [x_3d, y_3d, z_3d, x_2d, y_2d, z_2d, ..._2d]
    '''
    def __init__(self, grid_shape, num_feat_3d, num_feat_2d, num_feat_1d,
                 a_3d=-0.001, b_3d=0.001,
                 a_2d=0.9, b_2d=1.1) -> None:
        super().__init__()
        assert num_feat_2d == num_feat_3d
        assert num_feat_1d == num_feat_3d
        init_bound = np.power(0.001, 1.0/(1+3+3))
        
        self.grid_shape = grid_shape
        self.num_feat_3d = num_feat_3d
        self.num_feat_2d = num_feat_2d
        self.num_feat_1d = num_feat_1d
        self.feature_grid_3d = torch.nn.Parameter(
            torch.Tensor(1, num_feat_3d, *reversed(grid_shape[:3])),
            requires_grad=True
        )
        torch.nn.init.uniform_(self.feature_grid_3d, a=-init_bound, b=init_bound)
        
        self.plane_dimid = list(combinations(range(len(grid_shape[3:6])), 2))
        self.plane_dims = list(combinations(grid_shape[3:6], 2))
        self.line_dimid = list(range(3, 3+len(grid_shape[6:])))
        self.line_dims = grid_shape[6:]
        self.planes = []
        self.lines = []
        print('plane dimid', self.plane_dimid)
        print('plane dims', self.plane_dims)
        print('line dimid', self.line_dimid)
        print('line dims', self.line_dims)
        for i, dims in enumerate(self.plane_dims):
            plane = torch.nn.Parameter(
                torch.Tensor(1, num_feat_2d, *reversed(dims)),
                requires_grad=True
            )
            torch.nn.init.uniform_(plane, a=-init_bound, b=init_bound)
            self.planes.append(plane)

        for i, dim in enumerate(self.line_dims):
            line = torch.nn.Parameter(
                torch.Tensor(num_feat_1d, dim),
                requires_grad=True
            )
            torch.nn.init.uniform_(line, a=-init_bound, b=init_bound)
            self.lines.append(line)
        self.planes = torch.nn.ParameterList(self.planes)
        self.lines = torch.nn.ParameterList(self.lines)
        # print('DecompGrid 2d shapes:', grid_shape[3:], 'plane_dimid', self.plane_dimid, 'dims', self.plane_dims)
        
        # initialize with Uniform(-1e-4, 1e-4)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        input: (Batch, Ndim)
        output: (Batch, num_feat_3d/2d)
        '''
        coords = x[..., :3]
        feats_3d = torch.nn.functional.grid_sample(self.feature_grid_3d,
                            coords.reshape(([1]*coords.shape[-1]) + list(coords.shape)),
                            mode='bilinear', align_corners=True)
        # print(feats_3d.shape)
        # feats_3d = feats_3d.squeeze().permute(1, 0)
        feats_3d = feats_3d.squeeze()
        # feats_3d = torch.ones((x.shape[0], self.num_feat_3d), device='cuda')
        for i, dimids in enumerate(self.plane_dimid):
            x2d = x[:,dimids]
            # print(i, dimids, x, x2d)
            x2d = x2d.reshape(([1]*x2d.shape[-1]) + list(x2d.shape))
            # print(x2d.shape)
            f2d = torch.nn.functional.grid_sample(self.planes[i],
                            x2d,
                            mode='bilinear', align_corners=True)
            # print(dimids, x2d.shape, self.planes[i].shape, x2d.reshape(([1]*x2d.shape[-1]) + list(x2d.shape)).shape, f2d.shape)
            # f2d = f2d.squeeze().permute(1, 0)
            f2d = f2d.squeeze()
            # feats_2d.append(f2d.squeeze().permute(1, 0))
            feats_3d = feats_3d * f2d
        for i, dimids in enumerate(self.line_dimid):
            x1d = x[:,dimids]
            x1dn = x1d*self.line_dims[i]
            x1d_f = torch.floor(x1dn)
            weights = x1dn-x1d_f
            x1d_f = x1d_f
            f1d = torch.lerp(self.lines[i][:,x1d_f.type(torch.int)], self.lines[i][:,torch.clamp(x1d_f+1.0, min=0.0, max=self.line_dims[i]-1).type(torch.int)], weights)
            f1d = f1d.squeeze()
            feats_3d = feats_3d * f1d
        return feats_3d.T
    
    

class DecompGridv4(torch.nn.Module):
    '''
    grid_shape: [x_3d, y_3d, z_3d, x_2d, y_2d, z_2d, ..._2d]
    '''
    def __init__(self, grid_shape, num_feat_3d, num_feat_2d, num_feat_1d,
                 a_3d=-0.001, b_3d=0.001,
                 a_2d=0.9, b_2d=1.1) -> None:
        super().__init__()
        assert num_feat_2d == num_feat_3d
        assert num_feat_1d == num_feat_3d
        init_bound = np.power(0.001, 1.0/3.0)
        
        self.grid_shape = grid_shape
        self.num_feat_3d = num_feat_3d
        self.num_feat_2d = num_feat_2d
        self.num_feat_1d = num_feat_1d
        # self.feature_grid_3d = torch.nn.Parameter(
        #     torch.Tensor(1, num_feat_3d, *reversed(grid_shape[:3])),
        #     requires_grad=True
        # )
        # torch.nn.init.uniform_(self.feature_grid_3d, a=-init_bound, b=init_bound)
        
        self.plane_dimid = list(combinations(range(len(grid_shape[3:6])), 2))
        self.plane_dims = list(combinations(grid_shape[3:6], 2))
        self.line_dimid = list(range(3, 3+len(grid_shape[6:])))
        self.line_dims = grid_shape[6:]
        self.planes = []
        self.lines = []
        print('plane dimid', self.plane_dimid)
        print('plane dims', self.plane_dims)
        print('line dimid', self.line_dimid)
        print('line dims', self.line_dims)
        for i, dims in enumerate(self.plane_dims):
            plane = torch.nn.Parameter(
                torch.Tensor(1, num_feat_2d, *reversed(dims)),
                requires_grad=True
            )
            torch.nn.init.uniform_(plane, a=-init_bound, b=init_bound)
            self.planes.append(plane)

        for i, dim in enumerate(self.line_dims):
            line = torch.nn.Parameter(
                torch.Tensor(num_feat_1d, dim),
                requires_grad=True
            )
            torch.nn.init.uniform_(line, a=0.9, b=1.1)
            self.lines.append(line)
        self.planes = torch.nn.ParameterList(self.planes)
        self.lines = torch.nn.ParameterList(self.lines)
        # print('DecompGrid 2d shapes:', grid_shape[3:], 'plane_dimid', self.plane_dimid, 'dims', self.plane_dims)
        
        # initialize with Uniform(-1e-4, 1e-4)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        input: (Batch, Ndim)
        output: (Batch, num_feat_3d/2d)
        '''
        feats = None
        for i, dimids in enumerate(self.plane_dimid):
            x2d = x[:,dimids]
            x2d = x2d.reshape(([1]*x2d.shape[-1]) + list(x2d.shape))
            f2d = torch.nn.functional.grid_sample(self.planes[i],
                            x2d,
                            mode='bilinear', align_corners=True)
            f2d = f2d.squeeze()
            if feats is None:
                feats = f2d
            else:
                feats = feats * f2d
        return feats.T


class DecompGridv5(torch.nn.Module):
    '''
    grid_shape: [x_3d, y_3d, z_3d, x_2d, y_2d, z_2d, ..._2d]
    '''
    def __init__(self, grid_shape, num_feat_3d, num_feat_2d, num_feat_1d) -> None:
        super().__init__()
        assert num_feat_2d == num_feat_3d
        assert num_feat_1d == num_feat_3d
        
        self.grid_shape = grid_shape
        self.num_feat_3d = num_feat_3d
        self.num_feat_2d = num_feat_2d
        self.num_feat_1d = num_feat_1d
        self.feature_grid_3d = torch.nn.Parameter(
            torch.Tensor(1, num_feat_3d, *reversed(grid_shape[:3])),
            requires_grad=True
        )
        torch.nn.init.uniform_(self.feature_grid_3d, a=-0.001, b=0.001)
        
        self.plane_dimid = list(combinations(range(len(grid_shape[3:6])), 2))
        self.plane_dims = list(combinations(grid_shape[3:6], 2))
        self.planes = []
        self.lines = []
        print('plane dimid', self.plane_dimid)
        print('plane dims', self.plane_dims)
        for i, dims in enumerate(self.plane_dims):
            plane = torch.nn.Parameter(
                torch.Tensor(1, num_feat_2d, *reversed(dims)),
                requires_grad=True
            )
            torch.nn.init.uniform_(plane, a=0.999, b=1.001)
            self.planes.append(plane)
        self.planes = torch.nn.ParameterList(self.planes)
        
        # initialize with Uniform(-1e-4, 1e-4)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        input: (Batch, Ndim)
        output: (Batch, num_feat_3d/2d)
        '''
        coords = x[..., :3]
        feats_3d = torch.nn.functional.grid_sample(self.feature_grid_3d,
                            coords.reshape(([1]*coords.shape[-1]) + list(coords.shape)),
                            mode='bilinear', align_corners=True)
        feats_3d = feats_3d.squeeze()
        for i, dimids in enumerate(self.plane_dimid):
            x2d = x[:,dimids]
            x2d = x2d.reshape(([1]*x2d.shape[-1]) + list(x2d.shape))
            f2d = torch.nn.functional.grid_sample(self.planes[i],
                            x2d,
                            mode='bilinear', align_corners=True)
            f2d = f2d.squeeze()
            feats_3d = feats_3d * f2d
        return feats_3d.T
    

class DecompGridv6(torch.nn.Module):
    '''
    grid_shape: [x_3d, y_3d, z_3d, x_2d, y_2d, z_2d, ..._2d]
    '''
    def __init__(self, grid_shape, num_feat_3d, num_feat_2d, num_feat_1d,
                 a_3d=-0.001, b_3d=0.001,
                 a_2d=0.9, b_2d=1.1) -> None:
        super().__init__()
        assert num_feat_2d == num_feat_3d
        assert num_feat_1d == num_feat_3d
        
        self.grid_shape = grid_shape
        self.num_feat_3d = num_feat_3d
        self.num_feat_2d = num_feat_2d
        self.num_feat_1d = num_feat_1d
        
        self.plane_dimid = list(combinations(range(len(grid_shape[3:6])), 2))
        self.plane_dims = list(combinations(grid_shape[3:6], 2))
        self.planes = []
        print('plane dimid', self.plane_dimid)
        print('plane dims', self.plane_dims)
        for i, dims in enumerate(self.plane_dims):
            plane = torch.nn.Parameter(
                torch.Tensor(1, num_feat_2d, *reversed(dims)),
                requires_grad=True
            )
            torch.nn.init.uniform_(plane, a=0.01, b=0.25)
            self.planes.append(plane)

        self.planes = torch.nn.ParameterList(self.planes)
        # print('DecompGrid 2d shapes:', grid_shape[3:], 'plane_dimid', self.plane_dimid, 'dims', self.plane_dims)
        
        # initialize with Uniform(-1e-4, 1e-4)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        input: (Batch, Ndim)
        output: (Batch, num_feat_3d/2d)
        '''
        feats = None
        for i, dimids in enumerate(self.plane_dimid):
            x2d = x[:,dimids]
            x2d = x2d.reshape(([1]*x2d.shape[-1]) + list(x2d.shape))
            f2d = torch.nn.functional.grid_sample(self.planes[i],
                            x2d,
                            mode='bilinear', align_corners=True)
            f2d = f2d.squeeze()
            if feats is None:
                feats = f2d
            else:
                feats = feats * f2d
        return feats.T

class DecompGridv7(torch.nn.Module):
    '''
    grid_shape: [x_3d, y_3d, z_3d, x_2d, y_2d, z_2d, ..._2d]
    '''
    def __init__(self, grid_shape, num_feat_3d, num_feat_2d, num_feat_1d,
                 a_3d=-0.001, b_3d=0.001,
                 a_2d=0.9, b_2d=1.1) -> None:
        super().__init__()
        assert num_feat_2d == num_feat_3d
        assert num_feat_1d == num_feat_3d
        
        # self.grid_shape = grid_shape
        self.num_feat_3d = num_feat_3d
        self.num_feat_2d = num_feat_2d
        self.num_feat_1d = num_feat_1d
        

        self.plane_dimid = list(combinations(range(3), 2))
        self.lowResPlane_dims = list(combinations([64, 64, 64], 2))
        self.midResPlane_dims = list(combinations([128, 128, 128], 2))
        self.highResPlane_dims = list(combinations([256, 256, 256], 2))
        self.lowResPlanes = []
        self.midResPlanes = []
        self.highResPlanes = []
        print('plane dimid', self.plane_dimid)
        print('low res plane dims', self.lowResPlane_dims)
        print('mid res plane dims', self.midResPlane_dims)
        print('high res plane dims', self.highResPlane_dims)
        for i, dims in enumerate(self.lowResPlane_dims):
            plane = torch.nn.Parameter(
                torch.Tensor(1, num_feat_2d, *reversed(dims)),
                requires_grad=True
            )
            torch.nn.init.uniform_(plane, a=0.01, b=0.25)
            self.lowResPlanes.append(plane)

        for i, dims in enumerate(self.midResPlane_dims):
            plane = torch.nn.Parameter(
                torch.Tensor(1, num_feat_2d, *reversed(dims)),
                requires_grad=True
            )
            torch.nn.init.uniform_(plane, a=0.01, b=0.25)
            self.midResPlanes.append(plane)

        for i, dims in enumerate(self.highResPlane_dims):
            plane = torch.nn.Parameter(
                torch.Tensor(1, num_feat_2d, *reversed(dims)),
                requires_grad=True
            )
            torch.nn.init.uniform_(plane, a=0.01, b=0.25)
            self.highResPlanes.append(plane)

        self.lowResPlanes = torch.nn.ParameterList(self.lowResPlanes)
        self.midResPlanes = torch.nn.ParameterList(self.midResPlanes)
        self.highResPlanes = torch.nn.ParameterList(self.highResPlanes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        input: (Batch, Ndim)
        output: (Batch, num_feat_3d/2d * 3)
        '''
        lowResfeats = 1.
        midResfeats = 1.
        highResfeats = 1.
        for i, dimids in enumerate(self.plane_dimid):
            x2d = x[:,dimids]
            x2d = x2d.reshape(([1]*x2d.shape[-1]) + list(x2d.shape))
            lowResf2d = torch.nn.functional.grid_sample(self.lowResPlanes[i],
                            x2d,
                            mode='bilinear', align_corners=True)
            lowResf2d = lowResf2d.squeeze()
            midResf2d = torch.nn.functional.grid_sample(self.midResPlanes[i],
                            x2d,
                            mode='bilinear', align_corners=True)
            midResf2d = midResf2d.squeeze()
            highResf2d = torch.nn.functional.grid_sample(self.highResPlanes[i],
                            x2d,
                            mode='bilinear', align_corners=True)
            highResf2d = highResf2d.squeeze()
            lowResfeats = lowResfeats * lowResf2d
            midResfeats = midResfeats * midResf2d
            highResfeats = highResfeats * highResf2d
            assert lowResfeats.shape == lowResf2d.shape
        feats = torch.cat((lowResfeats.T, midResfeats.T, highResfeats.T), 1)
        return feats

class DecompGridv8(torch.nn.Module):
    '''
    grid_shape: [x_3d, y_3d, z_3d, x_2d, y_2d, z_2d, ..._2d]
    '''
    def __init__(self, grid_shape, num_feat_3d, num_feat_2d, num_feat_1d) -> None:
        super().__init__()
        assert num_feat_2d == num_feat_3d
        assert num_feat_1d == num_feat_3d
        
        self.grid_shape = grid_shape
        self.num_feat_3d = num_feat_3d
        self.num_feat_2d = num_feat_2d
        self.num_feat_1d = num_feat_1d
        self.feature_grid_3d = torch.nn.Parameter(
            torch.Tensor(1, num_feat_3d, *reversed(grid_shape[:3])),
            requires_grad=True
        )
        torch.nn.init.uniform_(self.feature_grid_3d, a=0.1, b=0.5)
        
        self.plane_dimid = list(combinations(range(len(grid_shape[3:6])), 2))
        self.plane_dims = list(combinations(grid_shape[3:6], 2))
        self.planes = []
        self.lines = []
        print('plane dimid', self.plane_dimid)
        print('plane dims', self.plane_dims)
        for i, dims in enumerate(self.plane_dims):
            plane = torch.nn.Parameter(
                torch.Tensor(1, num_feat_2d, *reversed(dims)),
                requires_grad=True
            )
            torch.nn.init.uniform_(plane, a=0.1, b=0.5)
            self.planes.append(plane)
        self.planes = torch.nn.ParameterList(self.planes)
        
        # initialize with Uniform(-1e-4, 1e-4)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        input: (Batch, Ndim)
        output: (Batch, num_feat_3d/2d)
        '''
        coords = x[..., :3]
        feats_3d = torch.nn.functional.grid_sample(self.feature_grid_3d,
                            coords.reshape(([1]*coords.shape[-1]) + list(coords.shape)),
                            mode='bilinear', align_corners=True)
        feats_3d = feats_3d.squeeze()
        for i, dimids in enumerate(self.plane_dimid):
            x2d = x[:,dimids]
            x2d = x2d.reshape(([1]*x2d.shape[-1]) + list(x2d.shape))
            f2d = torch.nn.functional.grid_sample(self.planes[i],
                            x2d,
                            mode='bilinear', align_corners=True)
            f2d = f2d.squeeze()
            feats_3d = feats_3d * f2d
        return feats_3d.T
    

class DecompGridv9(torch.nn.Module):
    '''
    grid_shape: [x_3d, y_3d, z_3d, x_2d, y_2d, z_2d, ..._2d]
    '''
    def __init__(self, grid_shape, num_feat_3d, num_feat_2d, num_feat_1d) -> None:
        super().__init__()
        assert num_feat_2d == num_feat_3d
        
        self.grid_shape = grid_shape
        self.num_feat_3d = num_feat_3d
        self.num_feat_2d = num_feat_2d
        self.num_feat_1d = num_feat_1d
        self.feature_grid_3d = torch.nn.Parameter(
            torch.Tensor(1, num_feat_3d, *reversed(grid_shape[:3])),
            requires_grad=True
        )
        torch.nn.init.uniform_(self.feature_grid_3d, a=-0.001, b=0.001)
        
        self.plane_dimid = list(combinations(range(len(grid_shape[3:6])), 2))
        self.plane_dims = list(combinations(grid_shape[3:6], 2))
        self.line_dimid = list(range(3, 3+len(grid_shape[6:])))
        self.line_dims = grid_shape[6:]
        self.planes = []
        self.lines = []
        print('plane dimid', self.plane_dimid)
        print('plane dims', self.plane_dims)
        print('line dimid', self.line_dimid)
        print('line dims', self.line_dims)
        for i, dims in enumerate(self.plane_dims):
            plane = torch.nn.Parameter(
                torch.Tensor(1, num_feat_2d, *reversed(dims)),
                requires_grad=True
            )
            torch.nn.init.uniform_(plane, a=0.999, b=1.001)
            self.planes.append(plane)
        self.planes = torch.nn.ParameterList(self.planes)

        for i, dim in enumerate(self.line_dims):
            line = torch.nn.Parameter(
                torch.Tensor(num_feat_1d, dim),
                requires_grad=True
            )
            torch.nn.init.uniform_(line, a=0.01, b=0.25)
            self.lines.append(line)
        self.lines = torch.nn.ParameterList(self.lines)
        
        # initialize with Uniform(-1e-4, 1e-4)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        input: (Batch, Ndim)
        output: (Batch, num_feat_3d/2d)
        '''
        coords = x[..., :3]
        spatial_feats = torch.nn.functional.grid_sample(self.feature_grid_3d,
                            coords.reshape(([1]*coords.shape[-1]) + list(coords.shape)),
                            mode='bilinear', align_corners=True)
        spatial_feats = spatial_feats.squeeze()
        for i, dimids in enumerate(self.plane_dimid):
            x2d = x[:,dimids]
            x2d = x2d.reshape(([1]*x2d.shape[-1]) + list(x2d.shape))
            f2d = torch.nn.functional.grid_sample(self.planes[i],
                            x2d,
                            mode='bilinear', align_corners=True)
            f2d = f2d.squeeze()
            spatial_feats = spatial_feats * f2d
        param_feats = 1.
        for i, dimids in enumerate(self.line_dimid):
            p1d = x[:,dimids]
            p1dn = p1d*(self.line_dims[i]-1)
            p1d_f = torch.floor(p1dn)
            weights = p1dn-p1d_f
            p1d_f = p1d_f
            f1d = torch.lerp(self.lines[i][:,p1d_f.type(torch.int)], self.lines[i][:,torch.clamp(p1d_f+1.0, min=0.0, max=self.line_dims[i]-1).type(torch.int)], weights)
            f1d = f1d.squeeze()
            param_feats = param_feats * f1d
        feats = torch.cat((spatial_feats.T, param_feats.T), 1)
        return feats

    def forwardWithIntermediates(self, x: torch.Tensor) -> torch.Tensor:
        coords = x[..., :3]
        spatial_feats = torch.nn.functional.grid_sample(self.feature_grid_3d,
                            coords.reshape(([1]*coords.shape[-1]) + list(coords.shape)),
                            mode='bilinear', align_corners=True)
        spatial_feats = spatial_feats.squeeze()
        for i, dimids in enumerate(self.plane_dimid):
            x2d = x[:,dimids]
            x2d = x2d.reshape(([1]*x2d.shape[-1]) + list(x2d.shape))
            f2d = torch.nn.functional.grid_sample(self.planes[i],
                            x2d,
                            mode='bilinear', align_corners=True)
            f2d = f2d.squeeze()
            spatial_feats = spatial_feats * f2d
        param_feats = 1.
        for i, dimids in enumerate(self.line_dimid):
            p1d = x[:,dimids]
            p1dn = p1d*(self.line_dims[i]-1)
            p1d_f = torch.floor(p1dn)
            weights = p1dn-p1d_f
            p1d_f = p1d_f
            f1d = torch.lerp(self.lines[i][:,p1d_f.type(torch.int)], self.lines[i][:,torch.clamp(p1d_f+1.0, min=0.0, max=self.line_dims[i]-1).type(torch.int)], weights)
            f1d = f1d.squeeze()
            param_feats = param_feats * f1d
        feats = torch.cat((spatial_feats.T, param_feats.T), 1)
        return feats, spatial_feats.T, param_feats.T

###############################################################################################################
    
class INR_FG(torch.nn.Module):
    def __init__(self, grid_shape, num_feat_3d, num_feat_2d, num_feat_1d, out_features:int, version:int, dropout_layer:bool=False) -> None:
        super().__init__()
        if version == 2:
            self.dg = DecompGridv2(grid_shape=grid_shape, num_feat_3d=num_feat_3d, num_feat_2d=num_feat_2d, num_feat_1d=num_feat_1d)
        elif version == 3:
            self.dg = DecompGridv3(grid_shape=grid_shape, num_feat_3d=num_feat_3d, num_feat_2d=num_feat_2d, num_feat_1d=num_feat_1d)
        elif version == 4:
            self.dg = DecompGridv4(grid_shape=grid_shape, num_feat_3d=num_feat_3d, num_feat_2d=num_feat_2d, num_feat_1d=num_feat_1d)
        elif version == 5:
            self.dg = DecompGridv5(grid_shape=grid_shape, num_feat_3d=num_feat_3d, num_feat_2d=num_feat_2d, num_feat_1d=num_feat_1d)
        elif version == 6:
            self.dg = DecompGridv6(grid_shape=grid_shape, num_feat_3d=num_feat_3d, num_feat_2d=num_feat_2d, num_feat_1d=num_feat_1d)
        elif version == 7:
            self.dg = DecompGridv7(grid_shape=grid_shape, num_feat_3d=num_feat_3d, num_feat_2d=num_feat_2d, num_feat_1d=num_feat_1d)
        elif version == 8:
            self.dg = DecompGridv8(grid_shape=grid_shape, num_feat_3d=num_feat_3d, num_feat_2d=num_feat_2d, num_feat_1d=num_feat_1d)
        elif version == 9:
            self.dg = DecompGridv9(grid_shape=grid_shape, num_feat_3d=num_feat_3d, num_feat_2d=num_feat_2d, num_feat_1d=num_feat_1d)
        else:
            self.dg = DecompGridv2(grid_shape=grid_shape, num_feat_3d=num_feat_3d, num_feat_2d=num_feat_2d, num_feat_1d=num_feat_1d)
        
        self.hidden_nodes = 128
        self.hasDP = dropout_layer
        if version == 7:
            self.fc1 = torch.nn.Linear(num_feat_3d*3, self.hidden_nodes)
        elif version == 9:
            self.fc1 = torch.nn.Linear(num_feat_3d+num_feat_1d, self.hidden_nodes)
        else:
            self.fc1 = torch.nn.Linear(num_feat_3d, self.hidden_nodes)
        self.fc2 = torch.nn.Linear(self.hidden_nodes, self.hidden_nodes)
        self.fc3 = torch.nn.Linear(self.hidden_nodes, self.hidden_nodes)
        self.fc4 = torch.nn.Linear(self.hidden_nodes, out_features)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        if self.hasDP:
            self.dp = torch.nn.Dropout(p=0.125)
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.normal_(self.fc1.bias, 0, 0.001)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        torch.nn.init.normal_(self.fc2.bias, 0, 0.001)
        torch.nn.init.xavier_normal_(self.fc3.weight)
        torch.nn.init.normal_(self.fc3.bias, 0, 0.001)
        torch.nn.init.xavier_normal_(self.fc4.weight)
        torch.nn.init.normal_(self.fc4.bias, 0, 0.001)

    def forward(self, x):
        x = self.dg(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        if self.hasDP:
            x = self.dp(x)
        x = self.sigmoid(self.fc4(x))
        return x
    
    def forwardFGOnly(self, x):
        res, coord_features, param_features = self.dg.forwardWithIntermediates(x)
        return res, coord_features, param_features
