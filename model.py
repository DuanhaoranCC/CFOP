import torch
from torch.nn.init import ones_, zeros_
from torch.nn import Parameter
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv, GATConv, SAGEConv
from torch_geometric.nn import BatchNorm, LayerNorm
import copy
import numpy as np


class CosineDecayScheduler:
    def __init__(self, max_val, warmup_steps, total_steps):
        self.max_val = max_val
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get(self, step):
        if step < self.warmup_steps:
            return self.max_val * step / self.warmup_steps
        elif self.warmup_steps <= step <= self.total_steps:
            return self.max_val * (1 + np.cos((step - self.warmup_steps) * np.pi /
                                              (self.total_steps - self.warmup_steps))) / 2
        else:
            raise ValueError('Step ({}) > total number of steps ({}).'.format(step, self.total_steps))


def sce_loss(x, y, alpha=1):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss = - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


def mask(g, x, mask_rate=0.5):
    num_nodes = g.num_nodes()
    perm = torch.randperm(num_nodes, device=x.device)
    num_mask_nodes = int(mask_rate * num_nodes)
    mask_nodes = perm[: num_mask_nodes]

    return mask_nodes


class GraphSAGE_GCN(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size):
        super().__init__()

        self.convs = nn.ModuleList([
            SAGEConv(input_size, hidden_size, 'mean'),
            SAGEConv(hidden_size, hidden_size, 'mean'),
            SAGEConv(hidden_size, embedding_size, 'mean')
        ])

        self.skip_lins = nn.ModuleList([
            nn.Linear(input_size, hidden_size, bias=False),
            nn.Linear(input_size, hidden_size, bias=False),
        ])

        self.layer_norms = nn.ModuleList([
            LayerNorm(hidden_size),
            LayerNorm(hidden_size),
            LayerNorm(embedding_size),
        ])

        self.activations = nn.ModuleList([
            nn.PReLU(),
            nn.PReLU(),
            nn.PReLU(),
        ])

        self.dp = nn.Dropout(p=0.2)

    def forward(self, g, x):
        # x = g.ndata['feat']
        if 'batch' in g.ndata.keys():
            batch = g.ndata['batch']
        else:
            batch = None

        h1 = self.convs[0](g, self.dp(x))
        h1 = self.layer_norms[0](h1, batch)
        h1 = self.activations[0](h1)

        x_skip_1 = self.skip_lins[0](x)
        h2 = self.convs[1](g, h1 + x_skip_1)
        h2 = self.layer_norms[1](h2, batch)
        h2 = self.activations[1](h2)

        x_skip_2 = self.skip_lins[1](x)
        ret = self.convs[2](g, h1 + h2 + x_skip_2)
        ret = self.layer_norms[2](ret, batch)
        ret = self.activations[2](ret)
        return ret

    def reset_parameters(self):
        for m in self.convs:
            m.reset_parameters()
        for m in self.skip_lins:
            m.reset_parameters()
        for m in self.activations:
            m.weight.data.fill_(0.25)
        for m in self.layer_norms:
            m.reset_parameters()


class Encoder1(nn.Module):
    def __init__(self, in_dim, out_dim, p1, hidden, num_layers):
        super(Encoder1, self).__init__()
        self.num_layers = num_layers
        self.conv = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.act = nn.ModuleList()
        for layer in range(num_layers):  # excluding the input layer
            self.act.append(nn.PReLU())
            if layer == 0 and num_layers == 1:
                self.conv.append(GraphConv(in_dim, out_dim))
                self.bn.append(BatchNorm(out_dim))
            elif layer == 0:
                self.conv.append(GraphConv(in_dim, hidden))
                self.bn.append(BatchNorm(hidden))
            else:
                self.conv.append(GraphConv(hidden, out_dim))
                self.bn.append(BatchNorm(out_dim))

        self.dp = nn.Dropout(p1)

    def forward(self, graph, feat):
        h = self.dp(feat)
        for i, layer in enumerate(self.conv):
            h = layer(graph, h)
            h = self.bn[i](h)
            if self.num_layers > 1 and i == 0:
                h = self.act[i](h)

        # h = self.dp(feat)
        # h = self.conv1(graph, x)
        # h = self.bn(h)
        # h = self.act(h)
        # h = self.conv2(graph, h)
        # h = self.bn2(h)

        return h

    def reset_parameters(self):
        for i in range(self.num_layers):
            self.conv[i].reset_parameters()
            self.bn[i].reset_parameters()
        # self.conv1.reset_parameters()
        # self.conv2.reset_parameters()
        # self.bn.reset_parameters()
        # self.bn2.reset_parameters()


class CG(nn.Module):
    def __init__(self, in_dim, out_dim, p1, rate, hidden, layers):
        super(CG, self).__init__()
        self.online_encoder = Encoder1(in_dim, out_dim, p1, hidden, layers)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_encoder.reset_parameters()
        self.rate = rate
        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        self.criterion = self.setup_loss_fn("sce", 1)

        # stop gradient
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def trainable_parameters(self):
        r"""Returns the parameters that will be updated via an optimizer."""
        return list(self.online_encoder.parameters())

    @torch.no_grad()
    def update_target_network(self, mm):
        r"""Performs a momentum update of the target network's weights.
        Args:
            mm (float): Momentum used in moving average update.
        """
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)

    def forward(self, graph, feat):

        mask_nodes = mask(graph, feat, mask_rate=self.rate)
        x = feat.clone()
        x[mask_nodes] = 0.0
        x[mask_nodes] += self.enc_mask_token

        h1 = self.online_encoder(graph, x)
        with torch.no_grad():
            h2 = self.target_encoder(graph, feat)
        loss = self.criterion(h1[mask_nodes], h2[mask_nodes].detach())

        return loss

    def get_embed(self, graph, feat):
        h1 = self.online_encoder(graph, feat)

        return h1.detach()
