from reg import Regularization
import torch
import torch.utils
import torch.utils.data
import torch.nn.functional as F
import tqdm

def evaluate(model, eval_set, device):
    model.eval()
    with torch.no_grad():
        tot_loss = 0.0
        tot_accuracy = 0.0
        for x, y in tqdm.tqdm(eval_set):
            x = x.long().to(device)
            y = y.long().to(device)

            out = model(x)

            tot_loss += F.cross_entropy(out, y, reduction='mean').item()

            predictions = torch.max(out, 1)[1]
            tot_accuracy += torch.sum(predictions == y).item()

        loss = tot_loss / len(eval_set.dataset)
        acc = tot_accuracy / len(eval_set.dataset)
        return loss, acc


def train(model, train_set, optimizer, device):
    model.train()
    tot_loss = 0.0
    tot_accuracy = 0.0
    for i, (x, y) in enumerate(tqdm.tqdm(train_set)):
        x = x.long().to(device)
        y = y.long().to(device)

        model.zero_grad()
        
        out = model(x)

        loss = F.cross_entropy(out, y, reduction='mean')
        tot_loss += loss.item()

        predictions = torch.max(out, 1)[1]
        tot_accuracy += torch.sum(predictions == y).item()

        loss.backward()
        optimizer.step()

    loss = tot_loss / len(train_set.dataset)
    acc = tot_accuracy / len(train_set.dataset)
    return loss, acc
