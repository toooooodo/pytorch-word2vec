from dataset import PTBDataSet
from model import SkipGram
import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

embedding_dim = 100
lr = 5e-3
batch_size = 512
epochs = 20
show_top_k = 10


def main():
    train_set = PTBDataSet()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    device = torch.device('cuda')
    model = SkipGram(train_set.get_token_num(), embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (center, context_negative, mask, label) in enumerate(train_loader):
            center, context_negative, mask, label = center.to(device), context_negative.to(device), mask.to(
                device), label.to(device)
            criteon = nn.BCEWithLogitsLoss(weight=mask.double(), reduction='none').to(device)
            # pred: [batch_size, max_len]
            pred = model(center, context_negative)
            loss = torch.sum(torch.sum(criteon(pred.double(), label.double()), dim=1) / torch.sum(mask.double(), dim=1))
            total_loss += loss.item()
            if batch_idx % 200 == 0:
                print(f'epoch {epoch+1} batch {batch_idx} loss {loss.item()/pred.shape[0]}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'-->epoch {epoch+1} average loss {total_loss/train_set.__len__()}')

    model.get_topk_similar_tokens('chip', train_set.index_to_token, train_set.token_to_index, device, show_top_k)


if __name__ == '__main__':
    main()
