import numpy as np
import functools
import copy
import torch
import torch.nn as nn
from torch.optim import Adam
import sklearn.metrics as metrics
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder


# Code reference resources https://github.com/flyingtango/DiGCL,
#                          https://github.com/CRIPAC-DIG/GRACE
def sample_per_class(random_state, labels, num_examples_per_class, forbidden_indices=None):
    num_samples = labels.shape[0]
    num_classes = labels.max() + 1
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index] == class_index:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [random_state.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])


def repeat(n_times):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            results = [f(*args, **kwargs) for _ in range(n_times)]
            statistics = {}
            for key in results[0].keys():
                values = [r[key] for r in results]
                statistics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values)}
            # print_statistics(statistics, f.__name__)
            return statistics

        return wrapper

    return decorator


def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, np.bool)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret


def print_statistics(statistics, function_name):
    print(f'(E) | {function_name}:', end=' ')
    for i, key in enumerate(statistics.keys()):
        mean = statistics[key]['mean']
        std = statistics[key]['std']
        print(f'{key}={mean:.4f}+-{std:.4f}', end='')
        if i != len(statistics.keys()) - 1:
            print(',', end=' ')
        else:
            print()


def get_train_val_test_split(random_state,
                             labels,
                             train_examples_per_class=None, val_examples_per_class=None,
                             test_examples_per_class=None,
                             train_size=None, val_size=None, test_size=None):
    num_samples = labels.shape[0]
    num_classes = labels.max() + 1
    remaining_indices = list(range(num_samples))

    if train_examples_per_class is not None:
        train_indices = sample_per_class(
            random_state, labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices = random_state.choice(
            remaining_indices, train_size, replace=False)

    if val_examples_per_class is not None:
        val_indices = sample_per_class(
            random_state, labels, val_examples_per_class, forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(
            remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(random_state, labels, test_examples_per_class,
                                        forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(
            remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)
               ) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)
               ) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert len(np.concatenate(
            (train_indices, val_indices, test_indices))) == num_samples

    if train_examples_per_class is not None:
        train_labels = labels[train_indices]
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices]
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices]
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1

    return train_indices, val_indices, test_indices


# def train_test_split(labels, seed, train_examples_per_class=None, val_examples_per_class=None,
#                      test_examples_per_class=None, train_size=None, val_size=None, test_size=None):
#     random_state = np.random.RandomState(seed)
#     train_indices, val_indices, test_indices = get_train_val_test_split(
#         random_state, labels, train_examples_per_class, val_examples_per_class, test_examples_per_class, train_size,
#         val_size, test_size)
#
#     # print('number of training: {}'.format(len(train_indices)))
#     # print('number of validation: {}'.format(len(val_indices)))
#     # print('number of testing: {}'.format(len(test_indices)))
#
#     train_mask = np.zeros((labels.shape[0], 1), dtype=int)
#     train_mask[train_indices, 0] = 1
#     train_mask = np.squeeze(train_mask, 1)
#     val_mask = np.zeros((labels.shape[0], 1), dtype=int)
#     val_mask[val_indices, 0] = 1
#     val_mask = np.squeeze(val_mask, 1)
#     test_mask = np.zeros((labels.shape[0], 1), dtype=int)
#     test_mask[test_indices, 0] = 1
#     test_mask = np.squeeze(test_mask, 1)
#     mask = {}
#     mask['train'] = train_mask
#     mask['val'] = val_mask
#     mask['test'] = test_mask
#     return mask

def test(X_train, y_train, X_test, y_test):
    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)

    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                       verbose=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)
    y_pred = prob_to_one_hot(y_pred)
    acc = accuracy_score(y_test, y_pred)
    micro = f1_score(y_test, y_pred, average="micro")
    macro = f1_score(y_test, y_pred, average="macro")

    return acc, micro, macro


class Logistic(nn.Module):
    def __init__(self, num_dim, num_class):
        super(Logistic, self).__init__()
        self.linear = nn.Linear(num_dim, num_class)

    def forward(self, x):
        logits = self.linear(x)
        return logits


def accuracy(y_pred, y_true):
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double()
    correct = correct.sum().item()
    return correct / len(y_true)


def linear_probing_for_transductive_node_classiifcation(model, x, optimizer, labels,
                                                        max_epoch, device, train_mask,
                                                        val_mask, test_mask):

    criterion = torch.nn.CrossEntropyLoss()
    best_val_acc = 0
    best_val_epoch = 0
    best_test = 0
    best_model = None

    epoch_iter = range(max_epoch)

    for epoch in epoch_iter:
        model.train()
        out = model(x)
        loss = criterion(out[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
        optimizer.step()

        with torch.no_grad():
            model.eval()
            pred = model(x)
            val_acc = accuracy(pred[val_mask], labels[val_mask])
            # val_loss = criterion(pred[val_mask], labels[val_mask])
            # test_acc = accuracy(pred[test_mask], labels[test_mask])
            # test_loss = criterion(pred[test_mask], labels[test_mask])

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model)

        best_model.eval()
        with torch.no_grad():
            pred = best_model(x)
            estp_test_acc = accuracy(pred[test_mask], labels[test_mask])

    return estp_test_acc


@repeat(10)
def label_classification(embeddings, train_mask, val_mask, test_mask, label, type, device, name='common'):
    X = embeddings.detach().cpu().numpy()
    Y = label.detach().cpu().numpy()
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(np.bool)

    if np.isinf(X).any() == True or np.isnan(X).any() == True:
        return {
            'F1Mi': 0,
            'F1Ma': 0,
            'Acc': 0
        }
    X = normalize(X, norm='l2')
    if type == 1:
        # mask = train_test_split(
        #     label.detach().cpu().numpy(), seed=np.random.randint(0, 35456, size=1), train_examples_per_class=20,
        #     val_size=500, test_size=None)
        #
        # X_train = X[mask['train'].astype(bool)]
        # X_val = X[mask['val'].astype(bool)]
        # X_test = X[mask['test'].astype(bool)]
        # y_train = Y[mask['train'].astype(bool)]
        # y_val = Y[mask['val'].astype(bool)]
        # y_test = Y[mask['test'].astype(bool)]
        X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                            test_size=1 - 0.1)
    elif name == 'WikiCS':
        a = []
        for i in range(20):
            X_train = X[train_mask[:, i].cpu().numpy()]
            X_val = X[val_mask[:, i].cpu().numpy()]
            X_test = X[test_mask.cpu().numpy()]
            y_train = Y[train_mask[:, i].cpu().numpy()]
            y_val = Y[val_mask[:, i].cpu().numpy()]
            y_test = Y[test_mask.cpu().numpy()]
            acc, micro, macro = test(X_train, y_train, X_test, y_test)
            a.append(acc)
        return {
            'Acc': np.array(a).mean(),
            'std': np.array(a).std()
        }
    elif name == 'arxiv':
        n_classes = np.unique(label.detach().cpu().numpy()).shape[0]
        encoder = Logistic(X.shape[1], n_classes)
        encoder.to(device)
        optimizer_f = Adam(encoder.parameters(), lr=0.005, weight_decay=1e-5)
        test_acc = linear_probing_for_transductive_node_classiifcation(encoder, embeddings.detach(), optimizer_f,
                                                                       label.detach(), 1000,
                                                                       device, train_mask, val_mask, test_mask)
        return {'Acc': test_acc}

    else:
        X_train = X[train_mask.cpu().numpy()]
        X_val = X[val_mask.cpu().numpy()]
        X_test = X[test_mask.cpu().numpy()]
        y_train = Y[train_mask.cpu().numpy()]
        y_val = Y[val_mask.cpu().numpy()]
        y_test = Y[test_mask.cpu().numpy()]

    acc, micro, macro = test(X_train, y_train, X_test, y_test)

    return {
        'F1Mi': micro,
        'F1Ma': macro,
        'Acc': acc
    }


def fit_ppi_linear(num_classes, train_data, val_data, test_data, device, repeat=1):
    r"""
        Trains a linear layer on top of the representations. This function is specific to the PPI dataset,
        which has multiple labels.
        """

    def train(classifier, train_data, optimizer):
        classifier.train()

        x, label = train_data
        x, label = x.to(device), label.to(device)
        for step in range(100):
            # forward
            optimizer.zero_grad()
            pred_logits = classifier(x)

            # loss and backprop
            loss = criterion(pred_logits, label)
            loss.backward()
            optimizer.step()

    def test(classifier, data):
        classifier.eval()
        x, label = data
        label = label.cpu().numpy().squeeze()

        # feed to network and classifier
        with torch.no_grad():
            pred_logits = classifier(x.to(device))
            pred_class = (pred_logits > 0).float().cpu().numpy()

        return metrics.f1_score(label, pred_class, average='micro') if pred_class.sum() > 0 else 0

    num_feats = train_data[0].size(1)
    criterion = torch.nn.BCEWithLogitsLoss()

    # normalization
    mean, std = train_data[0].mean(0, keepdim=True), train_data[0].std(0, unbiased=False, keepdim=True)
    train_data[0] = (train_data[0] - mean) / std
    val_data[0] = (val_data[0] - mean) / std
    test_data[0] = (test_data[0] - mean) / std

    best_val_f1 = []
    test_f1 = []
    for _ in range(repeat):
        tmp_best_val_f1 = 0
        tmp_test_f1 = 0
        for weight_decay in 2.0 ** np.arange(-10, 11, 2):
            classifier = torch.nn.Linear(num_feats, num_classes).to(device)
            optimizer = torch.optim.AdamW(params=classifier.parameters(), lr=0.01, weight_decay=weight_decay)

            train(classifier, train_data, optimizer)
            val_f1 = test(classifier, val_data)
            if val_f1 > tmp_best_val_f1:
                tmp_best_val_f1 = val_f1
                tmp_test_f1 = test(classifier, test_data)
        best_val_f1.append(tmp_best_val_f1)
        test_f1.append(tmp_test_f1)

    return [best_val_f1], [test_f1]
