import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
import argparse
import os

import torch
from torch import nn

from main import load_dataset, create_dataset, CLASS_NORMAL
from models import LSTMAutoencoder

COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

def plot_prediction(data, predictions, losses, title, ax):
    ax.plot(data, label='true', color=COLORS_PALETTE[0])
    ax.plot(predictions[0], label='reconstructed', color=COLORS_PALETTE[1])
    ax.set_title('{} (loss: {:.2f})'.format(title, losses[0]))
    ax.grid(True, linestyle='--')
    ax.legend()

def plot_predictions(net, test_normal_dataset, test_anomaly_dataset, ncols=6):
    fig, axs = plt.subplots(nrows=2, ncols=ncols, sharey=True, sharex=True, figsize=(18, 6))

    for i, data in enumerate(test_normal_dataset[:ncols]):
        predictions, losses = predict(net, [data])
        plot_prediction(data, predictions, losses, title='Normal', ax=axs[0, i])

    for i, data in enumerate(test_anomaly_dataset[:ncols]):
        predictions, losses = predict(net, [data])
        plot_prediction(data, predictions, losses, title='Anomaly', ax=axs[1, i])

    fig.tight_layout()
    plt.savefig('figure/predictions.png')

def plot_losses(normal_losses, anomaly_losses, threshold):
    plt.figure(figsize=(6, 4))

    plt.hist(normal_losses, bins=50, rwidth=0.9, color=COLORS_PALETTE[0], alpha=0.8)
    plt.hist(anomaly_losses, bins=50, rwidth=0.9, color=COLORS_PALETTE[1], alpha=0.8)
    plt.axline([threshold, 0], [threshold, 250], linestyle='--', color=COLORS_PALETTE[2])

    plt.xlim(0, 180)
    plt.ylim(0, 250)
    plt.grid(True, linestyle='--')
    plt.xlabel('Loss')
    plt.ylabel('Count')
    plt.savefig('figure/reconstruction_error.png', dpi=100)

def predict(net, dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    predictions, losses = [], []
    criterion = nn.L1Loss(reduction='sum').to(device)
    net = net.eval()
    with torch.no_grad():
        for seq_true in dataset:
            seq_true = seq_true.to(device)
            seq_pred = net(seq_true)

            loss = criterion(seq_pred, seq_true)

            predictions.append(seq_pred.cpu().numpy().flatten())
            losses.append(loss.item())

    return predictions, losses

def main(args):
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = load_dataset(args)

    normal_df = df[df.target == CLASS_NORMAL].drop(labels='target', axis=1)
    print('Normal heartbeat data {}'.format(normal_df.shape))

    anomaly_df = df[df.target != CLASS_NORMAL].drop(labels='target', axis=1)
    print('Anomaly heatbeat data', anomaly_df.shape)

    train_df, val_df = train_test_split(normal_df, test_size=0.15, random_state=args.random_seed)
    val_df, test_df = train_test_split(val_df, test_size=0.33, random_state=args.random_seed)

    train_dataset, seq_len, n_features = create_dataset(train_df)

    test_normal_dataset, seq_len, n_features = create_dataset(test_df)
    test_anomaly_dataset, seq_len, n_features = create_dataset(anomaly_df)

    net = LSTMAutoencoder(seq_len, n_features, args.embedding_dim)
    net = net.to(device)
    print(net)

    net.load_state_dict(torch.load(args.model_path, weights_only=True))

    _, train_losses = predict(net, train_dataset)

    threshold = pd.Series(train_losses).quantile(0.985)
    print('Set threshold = {:.3f}'.format(threshold))

    _, normal_losses = predict(net, test_normal_dataset)
    _, anomaly_losses = predict(net, test_anomaly_dataset)

    y_true = [0 for loss in test_normal_dataset] + [1 for loss in test_anomaly_dataset]
    y_pred = [loss > threshold for loss in normal_losses] + [loss > threshold for loss in anomaly_losses]

    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)

    print('precision {:5.3f} recall {:5.3f}'.format(precision, recall))

    plot_losses(normal_losses, anomaly_losses, threshold)

    plot_predictions(net, test_normal_dataset, test_anomaly_dataset)

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
