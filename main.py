# Pytorch imports
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch import nn
import torchmetrics
import torchinfo

# Utils imports
from mega import Mega
import numpy as np
import os
import matplotlib.pyplot as plt

# Internal imports
from assets import gnldataloader
from assets.gnldataloader import *
from assets.cnn import *
from assets.loops import *
from assets.checkpoint import *

DEBUG = True

def main():
    path_data = "data/matching/fronts"
    path_labels = "data/matching/labels"

    dataset = GNLDataLoader(path_labels, path_data, transform=None, debug=false)

    if DEBUG: print(
        f"[DEBUG] Items in the data folder: {len(sorted(os.listdir(path_data)))}",
        f"[DEBUG] Items in the labels folder: {len(sorted(os.listdir(path_labels)))}",
        sep="\n"
    )

    # Definition of the Hyperparameters

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LabialCNN(debug=False).to(device)

    # Print the summary of the model
    if DEBUG: torchinfo.summary(model, (1,75, 100, 150), col_names = ("input_size", "output_size", "num_params", "kernel_size", "mult_adds"), verbose = 1)

    batch_size = 32
    epochs = 2
    folds = 5
    learning_rate = 10 ** (-4)
    dropout = 0.5

    loss_fn = nn.CTCLoss(reduction="mean", zero_infinity=True, blank=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Training + Testing

    # Determines whether to resume from a given epoch or not
    recover = True

    if recover:
        model_from_check, epoch_reached, state_dict_check, optimizer_check, loss_fn_check = checkpoint_load(epoch_to_down = 6)  # Insert last epoch!!!
        model.load_state_dict(state_dict_check)
        optimizer.load_state_dict(optimizer_check)
        loss_fn = loss_fn_check
        epochs = epochs - epoch_reached
        recover = False

    for epoch_ind in range(epochs): # Epochs
        index = 0
        for fold in range(folds):   # k-fold Cross Validation
            for batch_index in range(125 // folds):
                if DEBUG: print(f"[DEBUG] Loading of batch {index + 1} for training (Index: {index})")
                current_batch = dataset[batch_size*index : batch_size*(index + 1)]

                if DEBUG: print(f"[DEBUG] Starting training of batch {batch_index + 1} (Index: {batch_index})")
                train_loop(device, current_batch, model, loss_fn, optimizer, index, epochs, epoch_ind, debug=DEBUG)

                index += 1
            print("===          The training has finished          ===")
            for batch_index in range(35 // folds):
                if DEBUG: print(f"[DEBUG] Loading of batch {index + 1} for testing (Index: {index})")
                current_batch = dataset[batch_size*index : batch_size*(index + 1)]

                if DEBUG: print(f"[DEBUG] Starting testing of batch {index + 1} (Index: {index})")
                test_loop(device, current_batch, model, batch_size, loss_fn, debug=DEBUG)

                index += 1
            print("===          The testing has finished          ===")
            print(f"===              Finished fold {fold}/{folds}              ===")
        print("=== === ==> SAVING A CHECKPOINT <== === ===")
        
        # Save the model with a checkpoint
        checkpoint_save(model, epoch_ind, model.state_dict(), optimizer.state_dict(), loss_fn)
        print("=== === ==>  CHECKPOINT SAVED  <== === ===")

    torch.save(model, "models/gunileo.pt")
    print("=== === ==>   SAVED THE MODEL   <== === ===")

    print("Goodbye, and thank you for all the fish")

def output():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    alphabet = [x for x in "abcdefghijklmnopqrstuvwxyz0123456789 "]
    path_data = "data/matching/fronts"
    path_labels = "data/matching/labels"
    dataset = GNLDataLoader(path_labels, path_data, transform=None, debug=False)
    model = torch.load("gunileo_model_epoch_2.pt",map_location=torch.device("cpu"))
    randomvids = dataset[2000:2004]
    for randomvid in randomvids:
    
        
        vid =randomvid[0].to(device)
        label =randomvid[1]
        print(label.shape)
        output=model(vid)
        o =""
        for frame in output:
            
            val =torch.argmax(frame)
            
            if val ==37:
                o+=" "
            elif val ==36:
                continue
            else:
                o+=alphabet[val]
        print(o)
        l = ""
        
        for lbl in label:
            print(int(lbl))
            l +=alphabet[int(lbl)]
        print(l)

if __name__ == "__main__":
    main()