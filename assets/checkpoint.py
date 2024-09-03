import torch
from mega import Mega

mega_username, mega_password = "YourUsername", "YourPassword"

mg = Mega()
m = mg.login(mega_username, mega_password)

def checkpoint_save(model, epoch, state_dict, optimizer, loss_fn) -> None:
    """
    Saves the trained model with some extra parameters in two files

    Parameters:
        - `model`: the model trained;
        - `epoch`: the epoch until which the model trained;
        - `state_dict`: a dictionary with the state of the model;
        - `optimizer`: the state of the optimizer of the model;
        - `loss_fn`: the state of the loss function of the model.
    """
    checkpoint = {
        "epoch": epoch,
        "state_dict": state_dict,
        "optimizer": optimizer,
        "loss_fn": loss_fn
    }
    print("=== === ==>   SAVING THE MODEL...  <== === ===")
    torch.save(model, f"/kaggle/working/gunileo_model_epoch_{epoch}.pt")
    print("=== === ==> SAVING A CHECKPOINT... <== === ===")
    torch.save(checkpoint, f"/kaggle/working/gunileo_checkpoint_epoch_{epoch}.pt")
    
    print("[CHECK] Uploading...")
    model_file = m.upload(f"/kaggle/working/gunileo_model_epoch_{epoch}.pt")
    check_file = m.upload(f"/kaggle/working/gunileo_checkpoint_epoch_{epoch}.pt")
    m.get_upload_link(model_file)
    m.get_upload_link(check_file)
    print("[CHECK]     Uploaded!")
    
    print("=== === ==>    CHECKPOINT DONE!    <== === ===")
    pass

def checkpoint_load(epoch_to_down):
    """
    Loads a checkpoint from a model.pt and a checkpoint.pt file

    Parameters:
        - `epoch_to_down`: the epoch to download;

    Returns:
        - `model`: the model;
        - `checkpoint`: a dictionary with the epoch, state, optimizer and loss function of the model.
    """
    print("[CHECK] Downloading...")
    model_to_download = m.find(f"gunileo_model_epoch_{epoch_to_down-1}.pt")
    m.download(model_to_download, f"/kaggle/working/downloads")
    check_to_download = m.find(f"gunileo_checkpoint_epoch_{epoch_to_down-1}.pt")
    m.download(check_to_download, f"/kaggle/working/downloads")
    print("[CHECK]    Downloaded!")
    
    model = torch.load(f"/kaggle/working/downloads/gunileo_model_epoch_{epoch_to_down-1}.pt")
    checkpoint = torch.load(f"/kaggle/working/downloads/gunileo_checkpoint_epoch_{epoch_to_down-1}.pt")
    epoch, state_dict, optimizer, loss_fn = checkpoint.values()
    return model, epoch, state_dict, optimizer, loss_fn