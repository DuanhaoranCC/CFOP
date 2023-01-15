import torch
import dgl
from torch.optim import AdamW, Adam
from model import CG, CosineDecayScheduler
from torch_geometric import seed_everything
import numpy as np
import warnings
import yaml
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from eval import label_classification, fit_ppi_linear
from dataset import load_data, get_ppi

warnings.filterwarnings('ignore')


def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    configs = configs[args.dataname]

    for k, v in configs.items():
        if "lr" in k or "w" in k:
            v = float(v)
        setattr(args, k, v)
    return args


seed_everything(35536)
parser = argparse.ArgumentParser(description="SimGOP")
parser.add_argument("--dataname", type=str, default="Cora")
parser.add_argument("--cuda", type=int, default=0)
args = parser.parse_args()
args = load_best_configs(args, "config.yaml")
dataname = args.dataname
device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

label_type = args.label
graph, feat, label, train_mask, val_mask, test_mask = load_data(dataname)
n_feat = feat.shape[1]
n_classes = np.unique(label).shape[0]
graph = graph.to(device)
label = label.to(device)
feat = feat.to(device)


def train():
    model = CG(n_feat, args.dim, args.p1, args.rate, args.hidden, args.layer).to(device)
    # total_params = sum(p.numel() for p in model.parameters())
    # print("Number of parameter: %.2fM" % (total_params/1e6))
    # scheduler
    lr_scheduler = CosineDecayScheduler(args.lr, args.warmup, args.epochs)
    mm_scheduler = CosineDecayScheduler(1 - 0.99, 0, args.epochs)
    # optimizer
    optimizer = Adam(model.trainable_parameters(), lr=args.lr, weight_decay=args.w)

    for epoch in range(1, args.epochs + 1):
        model.train()

        # update learning rate
        lr = lr_scheduler.get(epoch - 1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # update momentum
        mm = 1 - mm_scheduler.get(epoch - 1)
        # mm = 0.99
        loss = model(graph, feat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.update_target_network(mm)
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

    model.eval()
    z1 = model.get_embed(graph, feat)
    acc = label_classification(z1, train_mask, val_mask, test_mask,
                               label, label_type, name=dataname, device=device)
    # print(acc)
    print(f" Acc: {acc['Acc']['mean']}, Std: {round(acc['Acc']['std'], 4)}")


# train()

# TT({'dim': 512, 'epochs': 7000, 'lr': 8e-5, 'p1': 0.1, 'rate': 0.5, 'warmup': 100, 'w': 5e-4, 'acc': 0.6193, 'beta': 1})

data = [73.93, 10.99, 10.82, 1.87, 1.15, 1.24]
data1 = [30, 21, 13, 13, 12, 8, 3]
labels = ['C', 'N', 'O', 'S', 'F', 'I, P, B...']
labels1 = ['1', '2', '3', '4', '5', '6', '7']
# explode = [0.05, 0.05, 0.1, 0.2, 0.3, 0.45]
explode = [0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1]
colors = sns.color_palette('Dark2')
plt.pie(data1, colors=colors, autopct='%0.0f%%', explode=explode)
# plt.pie(data, colors=colors, labels=labels, autopct='%0.0f%%', explode=explode, wedgeprops={'width': 0.4})
plt.savefig("./饼图2.svg", dpi=800)
plt.show()

# # Cora = np.array([[82.9, 82.9, 84.2, 81.5]])
# # CiteSeer = np.array([[72.5, 71.1, 73.1]])
#
# NCI1 = np.array([[78.4, 79.9, 81, 82.7]])
# DD = np.array([[76, 78.1, 78.6, 79]])
# ENZYMES = np.array([[44, 30, 58.66, 59.6]])
# PTC_MR = np.array([[58.71, 59.3, 63.36, 64.6]])
# z = pd.DataFrame(np.concatenate([CiteSeer, Cora], axis=0),
#                  columns=['GraphMAE', 'LaGraph', 'SimGOP'], index=["CiteSeer", "Cora"])
# colors = sns.color_palette('pastel')
# z = pd.DataFrame(np.concatenate([ENZYMES, PTC_MR, NCI1], axis=0),
#                  columns=['GraphMAE', 'LaGraph', 'SimGOP', 'Supervised'], index=["ENZYMES", 'PTC-MR', "NCI1"])
# sns.set_palette("pastel")
# # sns.set(style='ticks')
# sns.lineplot(data=z, markers=True, linewidth=5)
# # plt.xlabel("Datasets")
# plt.ylabel("Accuracy")
# # plt.title("Graph Classification")
# plt.grid()
# sns.despine()
# plt.savefig("./折线图.svg", dpi=800)
# plt.show()

# Cora = np.array([[83, 82.7, 83.5, 83.3, 83.4, 82.9, 84.2, 83.8, 82.5]])
# Wiki = np.array([[78.6, 78.8, 79, 79.27, 79.4, 79.5, 79.6, 79.75, 79.8]])
# PubMed = np.array([[77.7, 77.9, 80.2, 80.1, 80.6, 77.9, 78.89, 78.5, 79.2]])
# z = pd.DataFrame(np.concatenate([Cora, Wiki, PubMed], axis=0),
#                  columns=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
#                  index=["Cora", "WikiCS", "PubMed"])
# sns.set_palette("pastel")
# # sns.set(style='darkgrid')
# sns.lineplot(data=z.transpose(), markers=True, linewidth=6)
# plt.xlabel("Masking Ratio")
# plt.ylabel("Accuracy")
# # sns.despine()
# # plt.title("Graph Classification")
# # plt.grid()
# plt.savefig("./M2.svg", dpi=800)
# plt.show()
