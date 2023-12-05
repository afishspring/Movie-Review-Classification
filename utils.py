import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
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
    test_size = int(0.5 * len(test_data))
    valid_size = len(test_data) - test_size
    test_data, valid_data = torch.utils.data.random_split(test_data, [test_size, valid_size])

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


def plot_confusion_matrix(model_name, y_true, y_pred):
    plt.cla()
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 8))
    sns.set(style="darkgrid")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('figs/'+model_name+'_Confusion_matrix.jpg', dpi=500)
    plt.show()

def plot_roc_curve(model_name, y_true, y_scores):
    plt.cla()
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=fpr, y=tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig('figs/'+model_name+'_Roc_curve.jpg', dpi=500)
    plt.show()

def report(model_name, y_true, y_pred):
    report = classification_report(y_true, y_pred)
    print(model_name+"-Classification Report:")
    print(report)

if __name__ == '__main__':
    load_data()