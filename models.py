import torch
import torch.nn as nn
from torchtext.vocab import GloVe


class BiLSTM(nn.Module):
    def __init__(self, vocab, embed_size=100, hidden_size=256, num_layers=2, dropout=0.1, use_glove=False):
        super().__init__()
        self.embedding = nn.Embedding(len(vocab), embed_size, padding_idx=vocab['<pad>'])
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(2 * hidden_size, 2)

        # xavier初始化参数
        self._reset_parameters()

        # 该参数决定是否使用预训练的词向量，如果使用，则将其冻结，训练期间不再更新
        if use_glove:
            glove = GloVe(name="6B", dim=100)
            self.embedding = nn.Embedding.from_pretrained(glove.get_vecs_by_tokens(vocab.get_itos()),
                                                          padding_idx=vocab['<pad>'],
                                                          freeze=True)

    def forward(self, x):
        x = self.embedding(x).transpose(0, 1)
        _, (h_n, _) = self.rnn(x)
        output = self.fc(torch.cat((h_n[-1], h_n[-2]), dim=-1))
        return output

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
