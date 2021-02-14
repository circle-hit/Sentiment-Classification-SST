from os import pread
from numpy.lib.arraysetops import setxor1d
import torch
from torch import device, dtype, long
import torch.utils
import torch.utils.data
import torch.nn.functional as F
import tqdm

def getLen_in_batch(sequences):
    result = []
    for sequence in sequences:
        cnt = 0
        for num in sequence:
            if num != 0:
                cnt += 1
        result.append(cnt)
    return torch.Tensor(result)
        
def evaluate(model, eval_set, device):
    model.eval()
    with torch.no_grad():
        tot_loss = 0.0
        tot_accuracy = 0.0
        for x, y in tqdm.tqdm(eval_set):
            x = x.long().to(device)
            y = y.long().to(device)
            x_len = torch.sum(x != 0, dim=1).to(device)

            out = model(x, x_len)['pred']
            out = F.softmax(out, dim=-1)

            tot_loss += F.cross_entropy(out, y, reduction='sum').item()

            predictions = torch.max(out, 1)[1]
            tot_accuracy += torch.sum(predictions == y).item()

        loss = tot_loss / len(eval_set.dataset)
        acc = tot_accuracy / len(eval_set.dataset)
        return loss, acc


def train(model, train_set, optimizer, device):
    model.train()
    tot_loss = 0.0
    tot_accuracy = 0.0
    for x, y in tqdm.tqdm(train_set):
        x = x.long().to(device)
        y = y.long().to(device)
        x_len = torch.sum(x != 0, dim=1).to(device)
        
        out = model(x, x_len)['pred']
        out = F.softmax(out, dim=-1)
        
        loss = F.cross_entropy(out, y, reduction='sum')
        tot_loss += loss.item()

        predictions = torch.max(out, 1)[1]
        tot_accuracy += torch.sum(predictions == y).item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss = tot_loss / len(train_set.dataset)
    acc = tot_accuracy / len(train_set.dataset)
    return loss, acc
