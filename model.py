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

    def get_topk_similar_tokens(self, query_token, index_to_token, token_to_index, device, k=5):
        x = self.embed_v(torch.LongTensor([token_to_index[query_token]]).to(device))
        w = torch.Tensor(self.embed_v.weight.cpu()).to(device)
        # cos [num_embeddings]
        # cos = torch.squeeze(w @ x) / torch.sqrt(torch.sum(w * w, dim=1) + torch.sum(x * x) + 1e-9)
        cos = torch.cosine_similarity(x, w, dim=-1)
        values, indices = torch.topk(cos, k + 1)
        for i in range(1, k + 1):
            print(f"{index_to_token[indices[i]]}: sim={values[i]}")
