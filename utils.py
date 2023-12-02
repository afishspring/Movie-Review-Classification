import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score, recall_score, precision_score
from torchtext.data import get_tokenizer
from torchtext import transforms as T
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import TensorDataset

def load_data():
    reviews_train, labels_train = read_imdb(is_train=True)
    reviews_test, labels_test = read_imdb(is_train=False)
    vocab = build_vocab_from_iterator(reviews_train, min_freq=20, specials=['<pad>', '<unk>', '<cls>', '<sep>'])
    vocab.set_default_index(vocab['<unk>'])
    train_data = build_dataset(reviews_train, labels_train, vocab)
    test_data = build_dataset(reviews_test, labels_test, vocab)
    return train_data, test_data, vocab

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

def plot_training_curve(train_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks(range(len(classes)), classes)
    plt.yticks(range(len(classes)), classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def plot_roc_curve(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

def report(y_true, y_pred):
    print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")