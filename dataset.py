from torch.utils.data import Dataset, DataLoader
import collections
import random
import math
import numpy as np


class PTBDataSet(Dataset):
    def __init__(self):
        super(PTBDataSet, self).__init__()

        self.count_least = 5
        self.t = 1e-4
        self.max_window_size = 5
        self.negative_sampling_num = 5
        self.centers, self.contexts_negatives, self.masks, self.labels = self.load_text()
        print(self.centers.shape, self.contexts_negatives.shape, self.masks.shape, self.labels.shape)

    def __len__(self):
        return self.centers.shape[0]

    def __getitem__(self, index):
        return [self.centers[index], self.contexts_negatives[index], self.masks[index], self.labels[index]]

    def load_text(self):
        with open('./data/ptb.train.txt', 'r') as f:
            text = f.readlines()
            raw_dataset = [sentence.split() for sentence in text]
        counter = collections.Counter([token for sentence in raw_dataset for token in sentence])
        # counter: {token: number_of_token}
        counter = dict(filter(lambda x: x[1] >= self.count_least, counter.items()))
        # index_to_token: [token1, token2, ..., tokenN]
        index_to_token = list(counter.keys())
        # token_to_index: {token: token_index}
        token_to_index = {token: index for index, token in enumerate(counter)}
        """
        dataset: [[sentence1],
                  [sentence2],
                  ...
                  [sentenceN]]
        all tokens in sentences are numbers(indices)
        """
        dataset = [[token_to_index[token] for token in sentence if token in token_to_index] for sentence in raw_dataset]
        token_num = sum([len(sentence) for sentence in dataset])
        sub_dataset = [[token for token in sentence if not self.discard(token, counter, index_to_token, token_num)] for
                       sentence in
                       dataset]
        print('token in sub_dataset', sum([len(sentence) for sentence in sub_dataset]))
        """
        all_centers: [center1, center2, ..., centerN]
        all_contexts: [[contexts1],
                       [contexts2],
                       ...
                       [contextsN]]
        """
        all_centers, all_contexts = self.get_centers_contexts(sub_dataset, self.max_window_size)
        sampling_weights = [counter[token] ** 0.75 for token in token_to_index]
        """
        all_negatives: [[negative1],
                        [negative2],
                        ...
                        [negativeN]]
        """
        all_negatives = self.get_negatives(all_contexts, sampling_weights, self.negative_sampling_num)
        return self.padding(all_centers=all_centers, all_contexts=all_contexts, all_negatives=all_negatives)
        # return all_centers, all_contexts, all_negatives
        # return np.array(all_centers), np.array(all_contexts), np.array(all_negatives)

    def discard(self, token_index, counter, index_to_token, token_num):
        return random.uniform(0, 1) < max(
            [0, 1 - math.sqrt(self.t / (counter[index_to_token[token_index]] / token_num))])

    def get_centers_contexts(self, dataset, max_window_size):
        centers, contexts = [], []
        for sentence in dataset:
            if len(sentence) < 2:
                continue
            centers += sentence
            for center_word_index in range(len(sentence)):
                # [1, max_window_size]
                window_size = random.randint(1, max_window_size)
                # [center_word - window_size, center_word + window_size]
                indices = list(
                    range(max([0, center_word_index - window_size]),
                          min([len(sentence), center_word_index + window_size + 1])))
                indices.remove(center_word_index)
                contexts.append([sentence[i] for i in indices])
        return centers, contexts

    def get_negatives(self, all_contexts, sampling_weights, K):
        """
        我们使用负采样来进行近似训练。对于一对中心词和背景词，我们随机采样 K 个噪声词（实验中设 K=5 ）。
        根据word2vec论文的建议，噪声词采样概率 P(w) 设为 w 词频与总词频之比的0.75次方
        :param all_contexts: 正样本
        :param sampling_weights: 采样概率
        :param K: K个噪声词
        :return:
        """
        all_negatives, neg_candidates, i = [], [], 0
        population = list(range(len(sampling_weights)))
        for context in all_contexts:
            negatives = []
            while len(negatives) < K * len(context):
                if i == len(neg_candidates):
                    i = 0
                    neg_candidates = random.choices(population=population, weights=sampling_weights, k=int(1e5))
                neg, i = neg_candidates[i], i + 1
                if neg not in set(context):
                    negatives += [neg]
            all_negatives.append(negatives)
        return all_negatives

    def padding(self, all_centers, all_contexts, all_negatives):
        """
        填充每一个训练样本
        :param all_centers:
        :param all_contexts:
        :param all_negatives:
        :return: centers: 样本中心词，shape: (centers,)
                 contexts_negatives：样本背景词与噪声词，shape: (centers, max_len)
                 masks：1 => 不是填充项；0 => 填充项, shape: (centers, max_len)
                 labels：1 => 正样本（背景词）；0 => 负样本（噪声词），shape: (centers, max_len)
        """
        max_len = self.max_window_size * 2 + self.max_window_size * 2 * self.negative_sampling_num
        centers, contexts_negatives, masks, labels = [], [], [], []
        for center, context, negative in zip(all_centers, all_contexts, all_negatives):
            cur_len = len(context) + len(negative)
            centers.append(center)
            contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
            masks += [cur_len * [1] + (max_len - cur_len) * [0]]
            labels += [len(context) * [1] + (max_len - len(context)) * [0]]
        return np.array(centers), np.array(contexts_negatives), np.array(masks), np.array(labels)


if __name__ == '__main__':
    train_set = PTBDataSet()
    train_loader = DataLoader(train_set, batch_size=512, shuffle=True)
    for batch_idx, (center, context_negative, mask, label) in enumerate(train_loader):
        print('center', center.shape)
        print('context_nagative',context_negative.shape)
        print('mask', mask.shape)
        print('label', label.shape)
        break
