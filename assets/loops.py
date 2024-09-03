import numpy as np
import torchmetrics
import torch
from torch import nn

metric = torchmetrics.Accuracy(task="multiclass", num_classes=38)
batch_size = 32

def train_loop(device, dataloader, model, loss_fn, optimizer, batch_index: int, epochs: int, epoch: int, debug: bool=True):
    """
    Trains an epoch of the model

    Parameters:
        - `device`: destination device
        - `dataloader`: the dataloader of the dataset
        - `model`: the model used
        - `loss_fn`: the loss function of the model
        - `optimizer`: the optimizer
        - `batch_index`: the number of the currently processed batch
        - `epochs`: the number of epochs
        - `epoch`: the index of the epoch
        - `debug`: (default `True`): prints debug info
    """
    model.train()
    size = len(dataloader)
    total = 0
    for item, (x, y) in enumerate(dataloader):

        video = x.to(device)
        label = y.to(device)
        lbllen = label.shape[0]
        pred = model(video)
        optimizer.zero_grad()
        
        loss = loss_fn(
            pred,
            label,
            torch.full(size=(1, ), fill_value=75, dtype=torch.long),
            torch.full(size=(1, ), fill_value=lbllen, dtype=torch.long)
        )

        loss.backward()
        optimizer.step()
        total += loss.item()

    if debug: print(f"→ Loss: {total/size} [Batch {batch_index + 1}/{size}, Epoch {epoch + 1}/{epochs}]")
    print(f"===     The batch {batch_index + 1}/160 has finished training     ===")


def GNLAccuracy(preds, labels) -> float:
    alphabet = [x for x in "abcdefghijklmnopqrstuvwxyz0123456789 "]
    total = 0
    for index, video in enumerate(preds):
        correct = 0
        pred_label = []
        label = [i for i in labels[index] if i != " "]
        for frame in video:
            letter = alphabet[torch.argmax(frame)]
            if letter != " ": pred_label.append(letter)

        for i, c in enumerate(pred_label):
            if c == label[i]:
                correct += 1
        total += correct / len(pred_label)
    return total / batch_size


def test_loop(device, dataloader, model, loss_fn, debug=True):
    model.eval()
    size = len(dataloader)

    with torch.no_grad():
        for item, (x, y) in enumerate(dataloader):
            video = x.to(device)
            label = y.to(device)
            pred = model(video)