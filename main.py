from dataset import PTBDataSet
from model import SkipGram
import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

embedding_dim = 100
lr = 5e-3


def main():
    train_set = PTBDataSet()
    train_loader = DataLoader(train_set, batch_size=512, shuffle=True)
    device = torch.device('cuda')
    model = SkipGram(train_set.get_token_num(), embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for batch_idx, (center, context_negative, mask, label) in enumerate(train_loader):
        center, context_negative, mask, label = center.to(device), context_negative.to(device), mask.to(
            device), label.to(device)
        # model.train()
        loss = nn.BCEWithLogitsLoss(weight=mask, reduction='none').to(device)
        pred = model(center, context_negative)
        break


if __name__ == '__main__':
    main()
