import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data

from module.dataset import ModelNet40
from module.utils import *

import os, sys
from collections import OrderedDict

import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BatchNorm(nn.Module):
    '''
        Perform batch normalization.
        Input: A tensor of size (N, M, feature_dim), or (N, feature_dim, M) 
                                               ( it would be the former case if feature_dim == M ), 
                or (N, feature_dim)
        Output: A tensor of the same size as input.
    '''
    def __init__(self, feature_dim):
        super(BatchNorm, self).__init__()
        self.feature_dim = feature_dim
        self.batchnorm = nn.BatchNorm1d(feature_dim)
        self.permute = Permute((0, 2, 1))
        
    def forward(self, x, _ = None):
        if (len(x.shape) == 3) and (x.shape[-1] == self.feature_dim):
            return self.permute(self.batchnorm(self.permute(x)))
        else:
            return self.batchnorm(x)


class Permute(nn.Module):
    def __init__(self, param):
        super(Permute, self).__init__()
        self.param = param
    def forward(self, x):
        return x.permute(self.param)


class MLP(nn.Module):
    def __init__(self, hidden_size, batchnorm = True, last_activation = True):
        super(MLP, self).__init__()
        q = []
        for i in range(len(hidden_size)-1):
            in_dim = hidden_size[i]
            out_dim = hidden_size[i+1]
            q.append(("Linear_%d" % i, nn.Linear(in_dim, out_dim)))
            if (i < len(hidden_size) - 2) or ((i == len(hidden_size) - 2) and (last_activation)):
                if (batchnorm):
                    q.append(("Batchnorm_%d" % i, BatchNorm(out_dim)))
                q.append(("ReLU_%d" % i, nn.ReLU(inplace=True)))
        self.mlp = nn.Sequential(OrderedDict(q))
    def forward(self, x):
        return self.mlp(x)


class MaxPooling(nn.Module):
    def __init__(self):
        super(MaxPooling, self).__init__()
    def forward(self, x, dim=1, keepdim = False):
        res, _ = torch.max(x, dim=dim, keepdim = keepdim)
        return res


class TNet(nn.Module):
    def __init__(self, nfeat, dropout = 0):
        super(TNet, self).__init__()
        self.nfeat = nfeat
        self.encoder = MLP((nfeat, 64, 128, 128))
        self.gcn = GCN(128, 128, 256, 512)
        self.decoder = nn.Sequential(MaxPooling(), BatchNorm(1024), 
                                     MLP((1024, 512, 256)), nn.Dropout(dropout), MLP((256, nfeat*nfeat)))
        
    def forward(self, x, adjs):
        batch_size = x.shape[0]
        x = self.decoder(self.gcn(self.encoder(x), adjs))
        return x.view(batch_size, self.nfeat, self.nfeat)


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class DynamicGraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907

    Build graph structure from a 3D point cloud using k-nearest neighbor(KNN)
    """

    def __init__(self, in_features, out_features, k, bias=True):
        super(DynamicGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.k = k

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj=None):
        if adj is None:
            adj = self.build_knn(input, self.k)

        support = torch.matmul(input, self.weight)
        output = torch.bmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'       

    def build_knn(self, input, k):
        batch_size, num_points, _ = input.size()

        inner = torch.matmul(input, input.transpose(1,2)) # (B,N,D) @ (B,D,N) > (B,N,N)
        xx = torch.sum(input**2, dim=2, keepdim=True) # (B,N,D) > (B,N,1)
        pairwise_distance = -1 * (xx + xx.transpose(2, 1) - 2*inner) # (B,N,1) + (B,1,N) -2*(B,N,N) > (B,N,N)
    
        _, topk_indices = pairwise_distance.topk(k=k, dim=-1) # (B,N,K)
        base_indices = torch.arange(num_points)[None, :, None].repeat(batch_size, 1, k).to(topk_indices.device) # (B,N,K)
        indices = torch.stack([base_indices.view(batch_size, -1), topk_indices.view(batch_size, -1)], dim=1) # (B, 2, N*K)
        
        adj = torch.stack([torch.sparse.FloatTensor(indices[b], (1/k) * torch.ones_like(indices[b][0], dtype=torch.float), torch.Size([num_points, num_points])) for b in range(batch_size)])
        return adj


class GCN(nn.Module):
    def __init__(self, n_in, n_hid1, n_hid2, n_out):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(n_in, n_hid1)
        self.batchnorm1 = BatchNorm(n_hid1)
        self.gc2 = GraphConvolution(n_in+n_hid1, n_hid2)
        self.batchnorm2 = BatchNorm(n_hid2)
        self.gc3 = GraphConvolution(n_in+n_hid1+n_hid2, n_out)
        self.batchnorm3 = BatchNorm(n_out)
    
    def forward(self, xs, adjs = None):
        if (adjs is None):
            xs, adjs = xs
        
        num_points = xs.shape[1]
        
        xs = torch.cat(tuple(xs), dim=0)
        xs = xs.to(device)
        adjs = adjs.to(device)
        
        xs1 = torch.cat( (xs, F.relu(self.batchnorm1(self.gc1(xs, adjs)))), dim = 1)
        del xs
        xs2 = torch.cat( (xs1, F.relu(self.batchnorm2(self.gc2(xs1, adjs)))), dim = 1)
        del xs1
        xs3 = torch.cat( (xs2, F.relu(self.batchnorm3(self.gc3(xs2, adjs)))), dim = 1)
        del xs2
        
        res = xs3
        ys = torch.stack(torch.split(res, num_points, dim=0)).to(device)
        return ys


class PointNetGCN(nn.Module):
    def __init__(self, nfeat, nclass, dropout = 0):
        super(PointNetGCN, self).__init__()

        self.input_transform = TNet(nfeat, 0.1)
        self.encoder = nn.Sequential(BatchNorm(3), MLP((nfeat, 64, 64)))
        self.feature_transform = TNet(64, 0.1)
        self.batchnorm = BatchNorm(64)
        self.mlp = MLP((64, 64, 128, 128))
        self.gcn = GCN(128, 128, 256, 512)
        self.maxpooling = MaxPooling()
        self.decoder = nn.Sequential(BatchNorm(1024), MLP((1024, 512, 256)), nn.Dropout(dropout), nn.Linear(256, nclass))

        self.eye64 = torch.eye(64).to(device)
        
    def forward(self, xs, adjs):
        batch_size = xs.shape[0]
        
        transform = self.input_transform(xs, adjs)
        xs = torch.stack([torch.mm(xs[i],transform[i]) for i in range(batch_size)])
        xs = self.encoder(xs)
        
        transform = self.feature_transform(xs, adjs)
        xs = torch.stack([torch.mm(xs[i],transform[i]) for i in range(batch_size)])
        
        xs = self.gcn(self.mlp(self.batchnorm(xs)), adjs)
        xs = self.decoder(self.maxpooling(xs))
        
        if (self.training):
            transform_transpose = transform.transpose(1, 2)
            tmp = torch.stack([torch.mm(transform[i], transform_transpose[i]) for i in range(batch_size)])
            L_reg = ((tmp - self.eye64) ** 2).sum() / batch_size
            
        return (F.log_softmax(xs, dim=1), L_reg) if self.training else F.log_softmax(xs, dim=1)


lr = 0.001
num_points = 128
save_name = "PointNet.pt"

########### loading data ###########
train_data = ModelNet40(num_points)
test_data = ModelNet40(num_points, 'test')

train_size = int(0.9 * len(train_data))
valid_size = len(train_data) - train_size
train_data, valid_data = Data.random_split(train_data, [train_size, valid_size])
valid_data.partition = 'valid'
train_data.partition = 'train'

print("train data size: ", len(train_data))
print("valid data size: ", len(valid_data))
print("test data size: ", len(test_data))

def collate_fn(batch):
    Xs = torch.stack([X for X, _, _ in batch])
    #adjs = [adj for _, adj, _ in batch]
    
    global num_points
    batch_size = len(batch)
    edges = torch.cat( tuple(batch[i][1][0] + i*num_points for i in range(batch_size)), dim=0)
    values = torch.cat( tuple(batch[i][1][1] for i in range(batch_size)), dim=0)
    N = num_points * batch_size
    adjs = torch.sparse.FloatTensor(edges.t(), values, torch.Size([N,N]))
    
    Ys = torch.tensor([Y for _,_, Y in batch], dtype = torch.long)
    return Xs, adjs, Ys


train_iter  = Data.DataLoader(train_data, shuffle = True, batch_size = 32, collate_fn = collate_fn)
valid_iter = Data.DataLoader(valid_data, batch_size = 32, collate_fn = collate_fn)
test_iter = Data.DataLoader(test_data, batch_size = 32, collate_fn = collate_fn)
############### loading model ####################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = PointNetGCN(nfeat=3, nclass=40, dropout=0.3)
net.to(device)
print(net)
############### training #########################
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay = 0.0001)
loss = nn.NLLLoss()

def adjust_lr(optimizer, decay_rate=0.95):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_rate

retrain = True
if os.path.exists(save_name):
    print("Model parameters have already been trained before. Retrain (y) or tune (n) ?")
    ans = input()
    if not (ans == 'y'):
        checkpoint = torch.load(save_name, map_location = device)
        net.load_state_dict(checkpoint["net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for g in optimizer.param_groups:
            g['lr'] = lr
        
train_model(train_iter, valid_iter, net, loss, optimizer, device = device, max_epochs = 1000, adjust_lr = adjust_lr,
            early_stop = EarlyStop(patience = 20, save_name = save_name))
    

############### testing ##########################

loss, acc = evaluate_model(test_iter, net, loss)
print('test acc = %.6f' % (acc))
