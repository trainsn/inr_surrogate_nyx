import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from utils import *
from itertools import combinations

import pdb

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

###############################################################################################################

class DecompGrid(torch.nn.Module):
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
            f1d = torch.lerp(self.lines[i][:,p1d_f.type(torch.long)], self.lines[i][:,torch.clamp(p1d_f+1.0, min=0.0, max=self.line_dims[i]-1).type(torch.long)], weights)
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
    def __init__(self, grid_shape, num_feat_3d, num_feat_2d, num_feat_1d, out_features:int, dropout_layer:bool=False) -> None:
        super().__init__()
        self.dg = DecompGrid(grid_shape=grid_shape, num_feat_3d=num_feat_3d, num_feat_2d=num_feat_2d, num_feat_1d=num_feat_1d)
        
        self.hidden_nodes = 128
        self.hasDP = dropout_layer
        self.out_features = out_features
        self.fc1 = torch.nn.Linear(num_feat_3d + num_feat_1d, self.hidden_nodes)
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
    
    def evidence(self, x):
        # Using softplus as the activation function for evidence
        return F.softplus(x)

    def DenseNormalGamma(self, x):
        mu, logv, logalpha, logbeta = x.chunk(4, dim=-1)
        mu = F.sigmoid(mu)
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        # Concatenating the tensors along the last dimension
        return torch.cat([mu, v, alpha, beta], dim=-1)

    def forward(self, x):
        x = self.dg(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        if self.hasDP:
            x = self.dp(x)
        x = self.fc4(x)
        if self.out_features == 4:
            x = self.DenseNormalGamma(x)
        else:
            x = self.sigmoid(x)
        return x
