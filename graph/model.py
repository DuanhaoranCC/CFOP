import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch.glob import SumPooling, MaxPooling, AvgPooling
from dgl.nn.pytorch import GINConv
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


def sce_loss(x, y, alpha=3):
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


class MLP(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d((hidden_dim))

    def forward(self, x):
        h = x
        h = F.relu(self.batch_norm(self.linears[0](h)))
        return self.linears[1](h)

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


class Encoder1(nn.Module):
    def __init__(self, in_hidden, out_hidden, hidden, num_layers):
        super(Encoder1, self).__init__()
        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.act = nn.ModuleList()
        # five-layer GCN with two-layer MLP aggregator and sum-neighbor-pooling scheme
        for layer in range(num_layers):  # excluding the input layer
            if layer == 0:
                mlp = MLP(in_hidden, out_hidden, hidden)
            else:
                mlp = MLP(hidden, out_hidden, hidden)
            self.ginlayers.append(
                GINConv(mlp, learn_eps=False)
            )  # set to True if learning epsilon
            self.batch_norms.append(BatchNorm(hidden))
            self.act.append(nn.PReLU())
        self.pool = SumPooling()

    def forward(self, graph, h):
        output = []
        for i, layer in enumerate(self.ginlayers):
            h = layer(graph, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            output.append(self.pool(graph, h))

        return h, torch.cat(output, dim=1)

    # def reset_parameters(self):
    #     self.conv1.reset_parameters()
    #     self.conv2.reset_parameters()
    #     self.conv3.reset_parameters()
    #     self.bn.reset_parameters()
    #     self.bn2.reset_parameters()
    #     self.bn3.reset_parameters()


class CG(nn.Module):
    def __init__(self, in_hidden, out_hidden, rate, hidden, alpha, layers):
        super(CG, self).__init__()
        self.online_encoder = Encoder1(in_hidden, out_hidden, hidden, layers)
        self.target_encoder = Encoder1(in_hidden, out_hidden, hidden, layers)
        # self.decoder = Encoder1(hidden, out_hidden, in_hidden, 1)
        self.rate = rate
        self.alpha = alpha
        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_hidden))
        self.criterion = self.setup_loss_fn("sce")

        # stop gradient
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # self.proj_head = nn.Sequential(nn.Linear(out_hidden * layers, 64 * layers), nn.ReLU(inplace=True),
        #                                nn.Linear(64 * layers, 128))

    def setup_loss_fn(self, loss_fn):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=1)
        else:
            raise NotImplementedError
        return criterion

    def cl_loss(self, x, x_aug, t=0.2):

        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / t)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        return loss

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

        h1, gh1 = self.online_encoder(graph, x)
        with torch.no_grad():
            h2, gh2 = self.target_encoder(graph, feat)

        # h = h1.clone()
        # h[mask_nodes] = 0.0
        # h2, _ = self.decoder(graph, h)
        # gh1 = self.proj_head(gh1)
        # gh2 = self.proj_head(gh2)
        # loss1 = self.criterion(h1[mask_nodes], h2[mask_nodes].detach())
        # loss2 = self.cl_loss(gh1, gh2, t=0.2)

        loss = self.criterion(h1[mask_nodes], h2[mask_nodes].detach()) * self.alpha + \
              self.criterion(gh1, gh2.detach()) * (1 - self.alpha)
        # loss = self.criterion(h2[mask_nodes], feat[mask_nodes])
        # loss = self.criterion(h1[mask_nodes], h2[mask_nodes].detach())
        # loss = self.criterion(gh1, gh2.detach())
        # loss = self.criterion(h1[mask_nodes], h2[mask_nodes].detach()) + \
        #        self.criterion(gh1, gh2.detach())

        return loss

    def get_embed(self, graph, feat):
        _, h = self.online_encoder(graph, feat)

        return h.detach()


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret
