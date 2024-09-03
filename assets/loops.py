import numpy as np
import torchmetrics
import torch
from torch import nn

metric = torchmetrics.Accuracy(task="multiclass", num_classes=38)
batch_size = 32

def train_loop(device, dataloader, model, loss_fn, optimizer, batch_index: int, epochs: int, epoch: int, debug: bool=True):
    """Trains an epoch of the model

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
    """ predictions = torch.zeros((batch_size, 75, 38)).to(device)  #np.ndarray(shape=(batch_size, 75, 38))
    labels = None #np.ndarray(shape=(batch_size, 37)) """
    total =0
    # Get the item from the dataset
    for item, (x, y) in enumerate(dataloader):
        #print(f"{x} -> {x.shape}")
        #for index, video in enumerate(x):
            # Move data to the device used
        video = x.to(device)
        label = y.to(device)
        lbllen = label.shape[0]
        # Compute the prediction and the loss
        pred = model(video)
        """ predictions.permute(1, 0, 2), """
        optimizer.zero_grad()
        
        loss = loss_fn(
        pred,
        label,
        torch.full(size=(1, ), fill_value=75, dtype=torch.long), # torch.Size([32])
        torch.full(size=(1, ), fill_value=lbllen, dtype=torch.long)  # torch.Size([32])
        )

        loss.backward()
        optimizer.step()
        """ predictions[item] = pred """
        total += loss.item()

            # if debug: print(video, video.shape, pred, pred.shape, label, label.shape, sep="\n\n========================================================\n\n")
        # total_acc = metric(pred.permute(1, 0), label)
        # print(f"[DEBUG] Accuracy: {total_acc}")

        # if debug: print(f"[DEBUG] Preds: {pred.shape}\n[DEBUG] Label: {label.shape}")
    
    
    # Adjust the weights
    # mean_loss = total_loss//batch_size
    # avg_acc=total_acc//batch_size
    
    
    """ torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) """
    

    if debug: print(f"→ Loss: {total/size} [Batch {batch_index + 1}/{size}, Epoch {epoch + 1}/{epochs}]")

    """predictions = torch.stack(predictions)
    labels = torch.stack(labels)
    preds_shape = predictions.shape
    labels_shape = labels.shape
    predictions = torch.reshape(predictions, (preds_shape[1], preds_shape[0], preds_shape[2]))
    """

    """
    print(
    f"Predictions:\n{predictions}\n\nSize of predictions: {preds_shape}",
    f"Labels:\n{y}\n\nLabels shape: {y.shape}",
    f"Input size:\n{torch.full(size=(batch_size, ), fill_value=75, dtype=torch.long)}",
    f"Labels size:\n{torch.full(size=(batch_size, ), fill_value=37, dtype=torch.long)}",
    sep="\n\n===============================================\n\n"
    )
    """


    # Print some information

    # if debug: print(f"Accuracy of item {item}/{size}: {GNLAccuracy(predictions, y)}")

    #accuracy = metric.compute()
    print(f"===     The batch {batch_index + 1}/125 has finished training     ===")
    #if debug: print(f"→ Final accuracy of the epoch: {accuracy}")
    #metric.reset()


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

    # Disable the updating of the weights
    with torch.no_grad():
        for item, (x, y) in enumerate(dataloader):
            # Move data to the device used
            video = x.to(device)
            label = y.to(device)

            # Compute the prediction and the loss
            pred = model(video)

            # Get the accuracy score
            
    # if debug: print(f"→ Final testing accuracy of the model: {acc}")
    # metric.reset()
