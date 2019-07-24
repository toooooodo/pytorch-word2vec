import torch
import torch.nn as nn


class SkipGram(nn.Module):
    def __init__(self, num_embeddings, embedding_dim=100):
        super(SkipGram, self).__init__()
        self.embed_v = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.embed_u = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

    def forward(self, center, context_negative):
        """
        在前向计算中，跳字模型的输入包含中心词索引center以及连结的背景词与噪声词索引contexts_and_negatives。
        其中center变量的形状为(批量大小,)，而contexts_and_negatives变量的形状为(批量大小, max_len)。
        这两个变量先通过词嵌入层分别由词索引变换为词向量，再通过小批量乘法得到形状为(批量大小, 1, max_len)的输出。
        输出中的每个元素是中心词向量与背景词向量或噪声词向量的内积。
        :param center: [batch_size]
        :param context_negative: [batch_size, max_len]
        :return: [batch_size, max_len]
        """
        out = torch.bmm(torch.unsqueeze(self.embed_v(center), 1), torch.transpose(self.embed_u(context_negative), 1, 2))
        out = torch.squeeze(out)
        return out
