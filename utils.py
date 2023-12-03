import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score, recall_score, precision_score
from torchtext.data import get_tokenizer
from torchtext import transforms as T
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

def load_data(batch_size):
    reviews_train, labels_train = read_imdb(is_train=True)
    reviews_test, labels_test = read_imdb(is_train=False)
    vocab = build_vocab_from_iterator(reviews_train, min_freq=20, specials=['<pad>', '<unk>', '<cls>', '<sep>'])
    vocab.set_default_index(vocab['<unk>'])
    train_data = build_dataset(reviews_train, labels_train, vocab)
    test_data = build_dataset(reviews_test, labels_test, vocab)

    # 将数据集分割成训练集和验证集
    train_size = int(0.8 * len(train_data))
    valid_size = len(train_data) - train_size
    train_data, valid_data = torch.utils.data.random_split(train_data, [train_size, valid_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader, vocab

def build_dataset(reviews, labels, vocab, max_len=500):
    text_transform = T.Sequential(
        T.VocabTransform(vocab=vocab),
        T.Truncate(max_seq_len=max_len),
        T.ToTensor(padding_value=vocab['<pad>']),
        T.PadTransform(max_length=max_len, pad_value=vocab['<pad>']),
    )
    dataset = TensorDataset(text_transform(reviews), torch.tensor(labels))
    return dataset

def read_imdb(path='./data', is_train=True):
    data, labels = [], []
    tokenizer = get_tokenizer('basic_english')
    for label in ('pos', 'neg'):
        folder_name = os.path.join(path, 'train' if is_train else 'test', label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), mode='r', encoding='utf-8') as f:
                review = f.read().replace('\n', '')
                data.append(tokenizer(review))
                labels.append(1 if label == 'pos' else 0)
    return data, labels

def plot_training_curve(train_losses, valid_accuracies):
    assert len(train_losses)==len(valid_accuracies), "train_losses and valid_accuracies must have the same length"
    epochs = len(train_losses)
    plt.figure(figsize=(10, 6))
    # Plot training loss
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss', marker='o')
    # Plot validation accuracy
    plt.plot(range(1, epochs + 1), valid_accuracies, label='Validation Accuracy', marker='o')

    plt.xlabel('Epoch')
    plt.ylabel('Loss / Accuracy')
    plt.title('Training Loss and Validation Accuracy Curve')
    plt.legend()
    plt.savefig('figs/Train_loss&Valid_acc.jpg', dpi=500)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes):
    plt.cla()
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks(range(len(classes)), classes)
    plt.yticks(range(len(classes)), classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('figs/Confusion_matrix.jpg', dpi=500)
    plt.show()

def plot_roc_curve(y_true, y_scores):
    plt.cla()
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig('figs/Roc_curve.jpg', dpi=500)
    plt.show()

def report(y_true, y_pred):
    print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")

if __name__ == '__main__':
    load_data()