import torch
import torch.nn as nn


class SkipGram(nn.Module):
    def __init__(self, num_embeddings, embedding_dim=100):
        super(SkipGram, self).__init__()
        self.embed_v = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.embed_u = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

    def forward(self, center, context_negative):
        print(self.embed_v(center).shape)
        # print(torch.transpose(self.embed_u(context_negative), 1, 2).shape)
