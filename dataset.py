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

        self.all_centers, self.all_contexts, self.all_negatives = self.load_text()
        print('center',self.all_centers.shape)
        print('contexts', self.all_contexts.shape)
        print('negatives', self.all_negatives.shape)

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

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
        # return all_centers, all_contexts, all_negatives
        return np.array(all_centers), np.array(all_contexts), np.array(all_negatives)

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
                indices = list(
                    range(max([0, center_word_index - window_size]),
                          min([len(sentence), center_word_index + window_size + 1])))
                indices.remove(center_word_index)
                contexts.append([sentence[i] for i in indices])
        return centers, contexts

    def get_negatives(self, all_contexts, sampling_weights, K):
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


if __name__ == '__main__':
    Data = PTBDataSet()
