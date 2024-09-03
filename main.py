from assets import gnldataloader
from assets.gnldataloader import *
from assets.cnn import *
from assets.loops import *

def main():
    path_data = "data/matching/fronts"
    path_labels = "data/matching/labels"

    dataset = GNLDataLoader(path_labels, path_data, transform=None, debug=False)

    print(
        f"[DEBUG] Items in the data folder: {len(sorted(os.listdir(path_data)))}",
        f"[DEBUG] Items in the labels folder: {len(sorted(os.listdir(path_labels)))}",
        sep="\n"
    )

    # Hyperparameters

    batch_size = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LabialCNN(debug=False).to(device)

    # Print the summary of the model
    # torchinfo.summary(model, (1,75, 100, 150), col_names = ("input_size", "output_size", "num_params", "kernel_size", "mult_adds"), verbose = 1)

    epochs = 1
    folds = 5
    learning_rate = 10 ** (-4)
    dropout = 0.5

    metric = torchmetrics.Accuracy(task="multiclass", num_classes=37)

    loss_fn = nn.CTCLoss(reduction="mean", zero_infinity=True, blank=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training + Testing

    for epoch_ind in range(epochs): # Epochs
        index = 0
        for fold in range(folds):   # k-fold Cross Validation
            for batch_index in range(125 // folds):    # 125
                print(f"[DEBUG] Loading of batch {index + 1} for training (Index: {index})")
                current_batch = dataset[batch_size*index : batch_size*(index + 1)]
                #print(f"[DEBUG] {type(current_batch), len(current_batch)}\n-> {type(current_batch[0]), len(current_batch[0])}\n-> {current_batch[0][0].shape}")

                print(f"[DEBUG] Starting training of batch {batch_index + 1} (Index: {batch_index})")
                train_loop(device, current_batch, model, loss_fn, optimizer, index, epochs, epoch_ind, debug=True)
                index += 1
            print("===          The training has finished          ===")
            for batch_index in range(35 // folds):    # 35
                print(f"[DEBUG] Loading of batch {index + 1} for testing (Index: {index})")
                current_batch = dataset[batch_size*index : batch_size*(index + 1)]

                print(f"[DEBUG] Starting testing of batch {index + 1} (Index: {index})")
                test_loop(device, current_batch, model, loss_fn, debug=True)
                index += 1
            print("===          The testing has finished          ===")
            print(f"===              Finished fold {fold}/{folds}              ===")
    print("=== === ==> SAVING THE MODEL...<== === ===")
    torch.save(model, "gunileonew.pt")
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
def check():
        cp =torch.load("gunileo_checkpoint_epoch_0.pt", map_location=torch.device('cpu'))
        print(cp.keys())
        a=cp["loss_fn"].methods()
        print(a)
        print(type(a))
        print(a.keys())
def dataloader_scrap():
    for item in gnldataloader.naughty_boys:
        os.remove(item)
    for item in gnldataloader.naughty_labels:
        os.remove(item)

if __name__ == "__main__":
    
    main()
