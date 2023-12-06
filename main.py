import numpy as np
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from utils import *
from models import *
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='LSTM', help='model type')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
parser.add_argument('--freeze', type=bool, default=False, help='freeze layer')
args = parser.parse_args()

train_loader, valid_loader, test_loader, vocab = load_data(args.batch_size)
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model_name = 'LSTM'
# model_name = 'Glove_LSTM'
# model_name = 'Freeze_LSTM'
# model_name = 'Freeze_Glove_LSTM'
model_savePath='pth/' + model_name + '.pth'
model = BiLSTM(vocab, use_glove=False).to(device)
# state_dict = torch.load('pth/LSTM.pth')
# model.load_state_dict(state_dict['model'])
# for param in model.embedding.parameters():
# 	param.requires_grad = False
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
criterion = nn.CrossEntropyLoss()

writer_path = './logs'
def train():
    writer = SummaryWriter(writer_path)

    train_losses = []
    valid_accuracies = []
    for epoch in range(1, args.epochs + 1):
        if args.freeze==True and epoch==args.epochs/2:
            print("Unfreeze")
            for param in model.embedding.parameters():
                param.requires_grad = True
                optimizer.add_param_group({'params': param})
        model.train()
        train_loss = 0
        loop = tqdm((train_loader), total=len(train_loader))
        for batch_idx, (X, y) in enumerate(loop):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            train_loss += loss.cpu().item()
            loss.backward()
            optimizer.step()

            loop.set_description(f'Epoch {epoch:2}:[{batch_idx + 1:3}/{len(train_loader)}]')

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # evaluation
        model.eval()
        correct = 0
        with torch.no_grad():
            for X, y in valid_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                correct += (pred.argmax(1) == y).sum().int().cpu()
            
            valid_acc = correct / len(valid_loader.dataset)  
            valid_accuracies.append(valid_acc)
            
            # tensorboard visualization
            writer.add_scalars(main_tag=model_name + '/loss',
                                tag_scalar_dict={'train loss': avg_train_loss},
                                global_step=epoch)
            writer.add_scalars(main_tag=model_name + '/accuracy',
                                tag_scalar_dict={'valid accuracy': valid_acc},
                                global_step=epoch)
    
    torch.save({
        'model': model.state_dict(),
        "loss": train_losses,
        "acc": valid_accuracies
        }, model_savePath)
    print("End")

def test():
    state_dict = torch.load(model_savePath)
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
    plot_roc_curve(model_name, y_true, y_scores)

    # Calculate and print additional metrics
    y_pred = (np.array(y_scores) > 0.5).astype(int)  # Adjust threshold as needed

    # Plot confusion matrix
    plot_confusion_matrix(model_name, y_true, y_pred)
    # F1-score/Recall/Precision
    report(model_name, y_true, y_pred)

if __name__ == '__main__':
    train()
    test()
