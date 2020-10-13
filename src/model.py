import torch
import torch.nn as nn
from torch.autograd import Variable

class LabelWordCompatLayer(nn.Module):
    def __init__(self, lemma_size, embedding_dim, ngram, output_dim, pad_idx, batch_size=10):
        nn.Module.__init__(self)

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.word_embedding = nn.Embedding(lemma_size, embedding_dim, padding_idx=pad_idx)

        assert ngram % 2 == 1, "n-gram should be odd number {2r+1}"
        self.phrase_filter = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            padding=(ngram - 1) / 2,  # pad should be (filter - 1)/2
            kernel_size=(ngram, 1))
        self.phrase_extract = nn.MaxPool2d(kernel_size=(1, output_dim))
        self.dropout = nn.Dropout(0.3)

        self.init_c()

    def init_c(self):
        self.c = Variable(torch.rand(size=(self.output_dim, self.embedding_dim)), requires_grad=True).to(device)

    def batch_cosinesim(self, v, c):
        normalized_v = v / torch.norm(v, p=2, dim=2).unsqueeze(2).repeat(1, 1, self.embedding_dim)
        normalized_c = c / torch.norm(c, p=2, dim=1).unsqueeze(1).repeat(1, self.embedding_dim)

        # nan -> pad_idx(0) or not-aligned label part
        normalized_v[torch.isnan(normalized_v)] = 0  # [b, l, h]
        normalized_c[torch.isnan(normalized_c)] = 0  # [k, h]

        normalized_c = normalized_c.unsqueeze(0).repeat(normalized_v.shape[0], 1, 1).permute(0, 2, 1)  # [b,h,k]
        g = torch.bmm(normalized_v, normalized_c)
        return g

    def forward(self, text):
        v = self.dropout(self.word_embedding(text))  # [b, l, h]

        g = self.dropout(self.batch_cosinesim(v, self.c))  # [b, l, k]
        u = torch.relu(self.phrase_filter(g.unsqueeze(1)).squeeze())  # [b, l, k]
        m = self.dropout(self.phrase_extract(u))  # [b, l, 1]
        b = torch.softmax(m, dim=1)  # [b, l, 1]

        return b, v


class JointInputShapeError(Exception):
    pass


class LEAM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, dropout, ngram, pad_idx, batch_size, device):
        nn.Module.__init__(self)

        self.compat_model = LabelWordCompatLayer(
            lemma_size=vocab_size,
            embedding_dim=embedding_dim,
            ngram=ngram,
            output_dim=output_dim,
            pad_idx=pad_idx,
            batch_size=batch_size
        )

        self.fc = nn.Linear(embedding_dim, output_dim)
        self.batch_size = batch_size
        self.device = device
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        weight, embed = self.compat_model(text)

        container = torch.full(size=embed.shape, fill_value=np.nan, device=self.device)
        for idx in xrange(self.batch_size):
            tmp = weight[idx] * embed[idx]
            container[idx] = tmp
        weighted_embed = container.sum(1)

        z = self.dropout(self.fc(weighted_embed))

        return z