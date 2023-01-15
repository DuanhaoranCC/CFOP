import numpy as np
import torch
from torch_geometric.datasets import WikiCS
import torch_geometric.transforms as T
import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset
from dgl.data import CoauthorCSDataset, CoauthorPhysicsDataset, PPIDataset
from dgl.transforms import RowFeatNormalizer
from dgl.dataloading import GraphDataLoader
from ogb.nodeproppred import DglNodePropPredDataset
from sklearn.preprocessing import StandardScaler


def get_ppi(root, transform=None):
    train_dataset = PPIDataset(mode='train', raw_dir=root)
    val_dataset = PPIDataset(mode='valid', raw_dir=root)
    test_dataset = PPIDataset(mode='test', raw_dir=root)
    train_val_dataset = [i for i in train_dataset] + [i for i in val_dataset]
    for idx, data in enumerate(train_val_dataset):
        data.ndata['batch'] = torch.zeros(data.number_of_nodes()) + idx
        data.ndata['batch'] = data.ndata['batch'].long()

    g = list(GraphDataLoader(train_val_dataset, batch_size=22, shuffle=True))

    return g, train_dataset, val_dataset, test_dataset


def preprocess(graph):
    feat = graph.ndata["feat"]
    graph = dgl.to_bidirected(graph)
    graph.ndata["feat"] = feat

    graph = graph.remove_self_loop()
    graph.create_formats_()
    return graph


def scale_feats(x):
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats


def load_data(name):
    if name == 'Cora':
        dataset = CoraGraphDataset(raw_dir='./')
    elif name == 'CiteSeer':
        dataset = CiteseerGraphDataset(raw_dir='./')
    elif name == 'PubMed':
        dataset = PubmedGraphDataset(raw_dir='./')
    elif name == 'Com':
        dataset = AmazonCoBuyComputerDataset(raw_dir='./')
    elif name == 'Photo':
        dataset = AmazonCoBuyPhotoDataset(raw_dir='./')
    elif name == 'CS':
        dataset = CoauthorCSDataset(raw_dir='./')
    elif name == 'Phy':
        dataset = CoauthorPhysicsDataset(raw_dir='./')
    elif name == 'WikiCS':
        dataset = WikiCS(root='./Wiki')
        data = dataset[0]
        graph = dgl.graph((data.edge_index[0], data.edge_index[1]))
        graph.ndata['feat'] = data.x
        graph.ndata['label'] = data.y
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
        feat = data.x
        label = data.y
        graph = graph.add_self_loop()
        return graph, feat, label, train_mask, val_mask, test_mask
    elif name == 'ppi':
        dataset, train_data, val_data, test_data = get_ppi('./', transform=RowFeatNormalizer())
        return dataset, train_data, val_data, test_data
    if name != 'arxiv':
        graph = dataset[0]
        feat = graph.ndata.pop('feat')
        label = graph.ndata.pop('label')
        co = ['Photo', 'Com', 'CS', 'Phy']
        if name in co:
            train_mask, val_mask, test_mask = None, None, None
        else:
            train_mask = graph.ndata.pop('train_mask')
            val_mask = graph.ndata.pop('val_mask')
            test_mask = graph.ndata.pop('test_mask')
    elif name == 'arxiv':
        graph, labels = dataset[0]
        num_nodes = graph.num_nodes()

        split_idx = dataset.get_idx_split()
        train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph = preprocess(graph)

        if not torch.is_tensor(train_idx):
            train_idx = torch.as_tensor(train_idx)
            val_idx = torch.as_tensor(val_idx)
            test_idx = torch.as_tensor(test_idx)

        feat = graph.ndata["feat"]
        feat = scale_feats(feat)
        graph.ndata["feat"] = feat

        train_mask = torch.full((num_nodes,), False).index_fill_(0, train_idx, True)
        val_mask = torch.full((num_nodes,), False).index_fill_(0, val_idx, True)
        test_mask = torch.full((num_nodes,), False).index_fill_(0, test_idx, True)
        graph.ndata["label"] = labels.view(-1)
        graph.ndata["train_mask"], graph.ndata["val_mask"], graph.ndata["test_mask"] = train_mask, val_mask, test_mask
        label = labels.view(-1)

    # train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
    # val_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
    # test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()

    graph = graph.add_self_loop()

    return graph, feat, label, train_mask, val_mask, test_mask
