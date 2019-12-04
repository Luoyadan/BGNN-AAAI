from torchtools import *
from collections import OrderedDict
import math
#import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import init
from util import sample_normal

class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, userelu=True, momentum=0.1, affine=True, track_running_stats=True):
        super(ConvBlock, self).__init__()
        self.layers = nn.Sequential()
        self.layers.add_module('Conv', nn.Conv2d(in_planes, out_planes,
            kernel_size=3, stride=1, padding=1, bias=False))

        if tt.arg.normtype == 'batch':
            self.layers.add_module('Norm', nn.BatchNorm2d(out_planes, momentum=momentum, affine=affine, track_running_stats=track_running_stats))
        elif tt.arg.normtype == 'instance':
            self.layers.add_module('Norm', nn.InstanceNorm2d(out_planes))

        if userelu:
            self.layers.add_module('ReLU', nn.ReLU(inplace=True))

        self.layers.add_module(
            'MaxPool', nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

    def forward(self, x):
        out = self.layers(x)
        return out

class ConvNet(nn.Module):
    def __init__(self, opt, momentum=0.1, affine=True, track_running_stats=True):
        super(ConvNet, self).__init__()
        self.in_planes  = opt['in_planes']
        self.out_planes = opt['out_planes']
        self.num_stages = opt['num_stages']
        if type(self.out_planes) == int:
            self.out_planes = [self.out_planes for i in range(self.num_stages)]
        assert(type(self.out_planes)==list and len(self.out_planes)==self.num_stages)

        num_planes = [self.in_planes,] + self.out_planes
        userelu = opt['userelu'] if ('userelu' in opt) else True

        conv_blocks = []
        for i in range(self.num_stages):
            if i == (self.num_stages-1):
                conv_blocks.append(
                    ConvBlock(num_planes[i], num_planes[i+1], userelu=userelu))
            else:
                conv_blocks.append(
                    ConvBlock(num_planes[i], num_planes[i+1]))
        self.conv_blocks = nn.Sequential(*conv_blocks)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv_blocks(x)
        out = out.view(out.size(0),-1)
        return out



# encoder for imagenet dataset
class EmbeddingImagenet(nn.Module):
    def __init__(self,
                 emb_size):
        super(EmbeddingImagenet, self).__init__()
        # set size
        self.hidden = 64
        self.last_hidden = self.hidden * 25
        self.emb_size = emb_size

        # set layers
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels=3,
                                              out_channels=self.hidden,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels=self.hidden,
                                              out_channels=int(self.hidden*1.5),
                                              kernel_size=3,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=int(self.hidden*1.5)),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv_3 = nn.Sequential(nn.Conv2d(in_channels=int(self.hidden*1.5),
                                              out_channels=self.hidden*2,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden * 2),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Dropout2d(0.4))
        self.conv_4 = nn.Sequential(nn.Conv2d(in_channels=self.hidden*2,
                                              out_channels=self.hidden*4,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden * 4),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Dropout2d(0.5))
        self.layer_last = nn.Sequential(nn.Linear(in_features=self.last_hidden * 4,
                                              out_features=self.emb_size, bias=True),
                                        nn.BatchNorm1d(self.emb_size))

    def forward(self, input_data):
        output_data = self.conv_4(self.conv_3(self.conv_2(self.conv_1(input_data))))
        return self.layer_last(output_data.view(output_data.size(0), -1))




class NodeUpdateNetwork(nn.Module):
    def __init__(self,
                 in_features,
                 num_features,
                 ratio=[2, 1],
                 dropout=0.0):
        super(NodeUpdateNetwork, self).__init__()
        # set size
        self.in_features = in_features
        self.num_features_list = [num_features * r for r in ratio]
        self.dropout = dropout

        # layers
        layer_list = OrderedDict()
        for l in range(len(self.num_features_list)):
            # Dimension Reduction
            layer_list['conv{}'.format(l)] = nn.Conv2d(
                in_channels=self.num_features_list[l - 1] if l > 0 else self.in_features * 3,
                out_channels=self.num_features_list[l],
                kernel_size=1,
                bias=False)
            layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=self.num_features_list[l],
                                                            )
            layer_list['relu{}'.format(l)] = nn.LeakyReLU()

            if self.dropout > 0 and l == (len(self.num_features_list) - 1):
                layer_list['drop{}'.format(l)] = nn.Dropout2d(p=self.dropout)

        self.network = nn.Sequential(layer_list)

    def forward(self, node_feat, edge_feat):
        # get size
        num_tasks = node_feat.size(0)
        num_data = node_feat.size(1)

        # get eye matrix (batch_size x 2 x node_size x node_size)
        diag_mask = 1.0 - torch.eye(num_data).unsqueeze(0).unsqueeze(0).repeat(num_tasks, 2, 1, 1).to(tt.arg.device)

        # set diagonal as zero and normalize
        edge_feat = F.normalize(edge_feat * diag_mask, p=1, dim=-1)

        # compute attention and aggregate
        aggr_feat = torch.bmm(torch.cat(torch.split(edge_feat, 1, 1), 2).squeeze(1), node_feat)

        node_feat = torch.cat([node_feat, torch.cat(aggr_feat.split(num_data, 1), -1)], -1).transpose(1, 2)

        # non-linear transform
        node_feat = self.network(node_feat.unsqueeze(-1)).transpose(1, 2).squeeze(-1)
        return node_feat


class EdgeUpdateNetwork(nn.Module):
    def __init__(self,
                 in_features,
                 num_features,
                 ratio=[2, 2, 1, 1],
                 separate_dissimilarity=False,
                 dropout=0.0, arch=None):
        super(EdgeUpdateNetwork, self).__init__()
        # set size
        self.in_features = in_features
        self.num_features = num_features
        self.num_features_list = [num_features * r for r in ratio]
        self.separate_dissimilarity = separate_dissimilarity
        self.dropout = dropout
        self.arch = arch
        # layers
        def creat_network(self, name):
            layer_list = OrderedDict()
            for l in range(len(self.num_features_list)):
                # set layer
                layer_list[name + 'conv{}'.format(l)] = nn.Conv2d(in_channels=self.num_features_list[l-1] if l > 0 else self.in_features,
                                                           out_channels=self.num_features_list[l],
                                                           kernel_size=1,
                                                           bias=False)
                layer_list[name + 'norm{}'.format(l)] = nn.BatchNorm2d(num_features=self.num_features_list[l],
                                                                )
                layer_list[name + 'relu{}'.format(l)] = nn.LeakyReLU()

                if self.dropout > 0:
                    layer_list[name + 'drop{}'.format(l)] = nn.Dropout2d(p=self.dropout)

            layer_list[name + 'conv_out'] = nn.Conv2d(in_channels=self.num_features_list[-1],
                                               out_channels=1,
                                               kernel_size=1)
            return layer_list
        self.sim_network = nn.Sequential(creat_network(self, 'sim_val'))


        if self.arch is 'edge':
            mod_self = self
            mod_self.num_features_list = [num_features]
            self.num_samples = 1
            self.W_mean = nn.Sequential(creat_network(mod_self, 'W_mean'))
            self.W_bias = nn.Sequential(creat_network(mod_self, 'W_bias'))
            self.B_mean = nn.Sequential(creat_network(mod_self, 'B_mean'))
            self.B_bias = nn.Sequential(creat_network(mod_self, 'B_bias'))

        if self.arch is 'att':
            self.attention = nn.Sequential(
                nn.Linear(num_features, int(num_features / 4)),
                nn.LeakyReLU(),
                nn.Linear(int(num_features / 4), 1)
            )

            def init_weights(m):
                if type(m) == nn.Linear:
                    init.xavier_normal_(m.weight.data)
                    init.normal_(m.bias.data)

            self.attention.apply(init_weights)





    def forward(self, node_feat, edge_feat):
        if self.arch is 'att':
            # task temperature
            alpha = self.attention(node_feat.view(-1, self.num_features)).view(node_feat.size(0), -1)# find attention for each task then expand to multi-dim
            alpha = F.softmax(alpha, dim=1).unsqueeze(2)
            node_feat = node_feat * alpha

        # Update adjacency matrix
        x_i = node_feat.unsqueeze(2)
        x_j = torch.transpose(x_i, 1, 2)
        x_ij = torch.abs(x_i - x_j)
        x_ij = torch.transpose(x_ij, 1, 3)

        # Bayes by Backprop
        if self.arch is 'edge':
            sim_val = self.sim_network(x_ij)
            w_mean = self.W_mean(x_ij)
            w_bias = self.W_bias(x_ij)
            b_mean = self.B_mean(x_ij)
            b_bias = self.B_bias(x_ij)
            logit_mean = w_mean * sim_val + b_mean
            logit_var = torch.log((sim_val ** 2)*torch.exp(w_bias) + torch.exp(b_bias))
            sim_val = F.sigmoid(sample_normal(logit_mean, logit_var, self.num_samples)) # batch * num_samples * node * node
        else:
            sim_val = F.sigmoid(self.sim_network(x_ij))

        if self.separate_dissimilarity:
            dsim_val = F.sigmoid(self.dsim_network(x_ij))
        else:
            dsim_val = 1.0 - sim_val


        diag_mask = 1.0 - torch.eye(node_feat.size(1)).unsqueeze(0).unsqueeze(0).repeat(node_feat.size(0), 2, 1, 1).to(tt.arg.device)
        edge_feat = edge_feat * diag_mask
        merge_sum = torch.sum(edge_feat, -1, True)
        # set diagonal as zero and normalize

        # edge_feat = F.normalize(sim_val * edge_feat, p=1, dim=-1) * merge_sum
        edge_feat = F.normalize(torch.cat([sim_val, dsim_val], 1) * edge_feat, p=1, dim=-1) * merge_sum
        force_edge_feat = torch.cat((torch.eye(node_feat.size(1)).unsqueeze(0), torch.zeros(node_feat.size(1), node_feat.size(1)).unsqueeze(0)), 0).unsqueeze(0).repeat(node_feat.size(0), 1, 1, 1).to(tt.arg.device)
        edge_feat = edge_feat + force_edge_feat
        edge_feat = edge_feat + 1e-6
        edge_feat = edge_feat / torch.sum(edge_feat, dim=1).unsqueeze(1).repeat(1, 2, 1, 1)

        return edge_feat


class GraphNetwork(nn.Module):
    def __init__(self,
                 in_features,
                 node_features,
                 edge_features,
                 num_layers,
                 num_cell,
                 dropout=0.0, arch = None):
        super(GraphNetwork, self).__init__()
        # set size
        self.in_features = in_features
        self.node_features = node_features
        self.edge_features = edge_features
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_cell = num_cell
        self.rnn = nn.GRUCell(self.node_features, self.node_features, bias=True)
        self.arch = arch

        # # build Mixture of Experts layer
        # self.MOE_conv = nn.ModuleList()
        # self.MOE_relu = nn.ModuleList()
        #
        #
        # for _ in range(self.num_cell):
        #     m = nn.Linear(self.node_features, self.node_features)
        #     if isinstance(m, nn.Linear):
        #         init.xavier_normal_(m.weight.data)
        #         init.normal_(m.bias.data)
        #     self.MOE_conv.append(m)
        #     self.MOE_relu.append(nn.LeakyReLU())
        #
        # self.gates = nn.Sequential(
        #     nn.Linear(self.node_features, self.num_cell, bias=False),
        #     nn.Softmax()
        # )
        #
        # # initialize Gating function
        # for layer in self.gates:
        #     if isinstance(layer, nn.Linear):
        #         init.xavier_normal_(layer.weight.data)


        # initialize GRU cells
        if isinstance(self.rnn, nn.GRUCell):
            for param in self.rnn.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)

        # for each layer
        for l in range(self.num_layers):
            # set edge to node
            edge2node_net = NodeUpdateNetwork(in_features=self.in_features if l == 0 else self.node_features,
                                              num_features=self.node_features,
                                              dropout=self.dropout if l < self.num_layers-1 else 0.0)

            # set node to edge
            node2edge_net = EdgeUpdateNetwork(in_features=self.node_features,
                                              num_features=self.edge_features,
                                              separate_dissimilarity=False,
                                              dropout=self.dropout if l < self.num_layers-1 else 0.0, arch = self.arch)

            self.add_module('edge2node_net{}'.format(l), edge2node_net)
            self.add_module('node2edge_net{}'.format(l), node2edge_net)

    # forward
    def forward(self, node_feat, edge_feat):
        # for each layer
        node_feat_list = []
        edge_feat_list = []
        edge_feat_temp_list = []
        node_feat_temp_list = []

        batch_size = node_feat.size(0)
        idx = list(range(batch_size+1))[::int(batch_size/self.num_cell)]
        for l in range(self.num_layers):
            for i in range(self.num_cell):
               # (1) edge to node
                node_feat_temp = self._modules['edge2node_net{}'.format(l)](node_feat[idx[i]: idx[i+1], :], edge_feat[idx[i]: idx[i+1], :])
                if i == 0:
                    hidden = torch.zeros_like(node_feat_temp.contiguous().view(-1, self.node_features))

                hidden = self.rnn(node_feat_temp.contiguous().view(-1, self.node_features), hidden)

                # (2) node to edge
                edge_feat_temp = self._modules['node2edge_net{}'.format(l)](hidden.view(node_feat_temp.size(0), -1, self.node_features), edge_feat[idx[i]: idx[i+1], :])

                # save node and edge feature (in each cell)
                edge_feat_temp_list.append(edge_feat_temp)
                node_feat_temp_list.append(node_feat_temp)

            # update node and edge feat
            edge_feat = torch.cat(edge_feat_temp_list)
            node_feat = torch.cat(node_feat_temp_list)
            # clear cache
            edge_feat_temp_list = []
            node_feat_temp_list = []

            edge_feat_list.append(edge_feat)
            node_feat_list.append(node_feat)



        return edge_feat_list, node_feat_list

