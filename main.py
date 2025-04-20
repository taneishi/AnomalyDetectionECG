import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.io.arff import loadarff
import argparse
import timeit
import os

import torch
from torch import nn, optim

from models import LSTMAutoencoder

CLASS_NAMES = ['Normal', 'R on T', 'PVC', 'SP', 'UB']
CLASS_NORMAL = 1

def create_dataset(df):
    dataset = torch.FloatTensor(df.values)
    dataset = dataset.reshape(df.shape[0], df.shape[1], 1)
    n_seq, seq_len, n_features = dataset.shape

    return dataset, seq_len, n_features

def load_dataset(args):
    train = loadarff(args.train_path)
    train = pd.DataFrame(train[0])

    test = loadarff(args.test_path)
    test = pd.DataFrame(test[0])

    df = pd.concat([train, test])
    df['target'] = df['target'].astype(int)

    df = df.sample(frac=1.0)

    return df

def main(args):
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = load_dataset(args)

    counts = df.groupby('target').aggregate(count=('target', 'size'))
    counts.insert(0, 'name', counts.index.map(lambda x: CLASS_NAMES[x - 1]))
    print(counts)

    print('Training and test dataset {}'.format(df.shape))

    normal_df = df[df.target == CLASS_NORMAL].drop(labels='target', axis=1)
    print('Normal heatbeat data {}'.format(normal_df.shape))

    train_df, val_df = train_test_split(normal_df, test_size=0.15, random_state=args.random_seed)
    val_df, test_df = train_test_split(val_df, test_size=0.33, random_state=args.random_seed)

    train_dataset, seq_len, n_features = create_dataset(train_df)
    val_dataset, seq_len, n_features = create_dataset(val_df)

    net = LSTMAutoencoder(seq_len, n_features, args.embedding_dim)
    net = net.to(device)
    print(net)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    criterion = nn.L1Loss(reduction='sum').to(device)

    train_losses = []
    val_losses = []
    for epoch in range(1, args.n_epochs + 1):
        epoch_start = timeit.default_timer()

        net = net.train()
        train_loss = []
        for seq_true in train_dataset:
            seq_true = seq_true.to(device)
            seq_pred = net(seq_true)

            loss = criterion(seq_pred, seq_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        train_losses.append(np.mean(train_loss))
        print('epoch {:3d} train loss {:6.3f}'.format(epoch, np.mean(train_loss)), end='')

        net = net.eval()
        val_loss = []
        with torch.no_grad():
            for seq_true in val_dataset:
                seq_true = seq_true.to(device)
                seq_pred = net(seq_true)

                loss = criterion(seq_pred, seq_true)
                val_loss.append(loss.item())

        val_losses.append(np.mean(val_loss))
        print(' val loss {:6.3f} {:5.1f}sec'.format(np.mean(val_loss), timeit.default_timer() - epoch_start))

    torch.save(net.state_dict(), args.model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', default=123, type=int)
    parser.add_argument('--train_path', default='data/ECG5000_TRAIN.arff', type=str)
    parser.add_argument('--test_path', default='data/ECG5000_TEST.arff', type=str)
    parser.add_argument('--model_path', default='model.pth', type=str)
    parser.add_argument('--n_epochs', default=150, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--embedding_dim', default=128, type=int)
    args = parser.parse_args()
    print(vars(args))

    main(args)
