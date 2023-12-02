import numpy as np
import argparse

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import load_data, plot_confusion_matrix, plot_roc_curve, plot_training_curve
from models import BiLSTM
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='lstm',
                    help='model type')
# 完整训练次数
parser.add_argument('--epochs', type=int, default=50,
                    help='Number of epochs to train.')
# 学习率
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
# batch_size 
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch_size')
args = parser.parse_args()

train_data, test_data, vocab = load_data()
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = BiLSTM(vocab).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

writer_path = './logs'
def train():
    writer = SummaryWriter(writer_path)

    train_losses = []
    for epoch in range(1, args.epochs + 1):
        train_loss = 0
        loop = tqdm((train_loader), total=len(train_loader))
        for batch_idx, (X, y) in enumerate(loop):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            train_loss += loss
            loss.backward()
            optimizer.step()

            loop.set_description(f'Epoch {epoch:2}:[{batch_idx + 1}/{len(train_loader)}] train loss: {loss:.4f}')
        
        avg_train_loss = train_loss / (batch_idx + 1)
        train_losses.append(avg_train_loss)
        
        # # tensorboard visualization
        # writer.add_scalars(main_tag=args.model + '/loss',
        #                        tag_scalar_dict={'train loss': avg_train_loss, 'test loss': test_loss},
        #                        global_step=epoch)
        # writer.add_scalars(main_tag=args.model + '/accuracy',
        #                        tag_scalar_dict={'train accuracy': train_acc, 'test accuracy': test_acc},
        #                        global_step=epoch)

    torch.save({'model': model.state_dict()}, 'pth/' + args.model + '.pth')
    plot_training_curve(train_losses)
    return train_losses

def test():
    state_dict = torch.load('pth/' + args.model + '.pth')
    model.load_state_dict(state_dict['model'])

    acc = 0
    y_true, y_scores = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            y_true.extend(y.cpu().numpy())
            y_scores.extend(pred.cpu().numpy()[:, 1])
            acc += (pred.argmax(1) == y).sum().item()

    print(f"Accuracy: {acc / len(test_loader.dataset):.4f}")

    # Plot ROC curve
    plot_roc_curve(y_true, y_scores)

    # Calculate and print additional metrics
    y_pred = (np.array(y_scores) > 0.5).astype(int)  # Adjust threshold as needed

    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, classes=['Class 0', 'Class 1'])

if __name__ == '__main__':
    train()
    test()
