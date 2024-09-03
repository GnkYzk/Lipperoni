import torch
from torch import nn
import torchinfo

class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]
    
class LabialCNN(nn.Module):
    def __init__(self, debug: bool = False):
        super().__init__()

        self.debug = debug
        self.cnn = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(3, 5, 5), padding=(1, 2, 2), stride=(1, 2, 2)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(3, 5, 5), padding=(1, 2, 2), stride=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 5, 5), padding=(1, 2, 2), stride=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
             # Left as default, check later if it causes problems
            
        )    
        self.gru = nn.Sequential(
            nn.GRU(input_size=1728, hidden_size=256, num_layers=2, dropout=0.5, bidirectional=True),
            SelectItem(0),

            nn.Linear(in_features=512, out_features=38),
            nn.LogSoftmax()
        )

    # Remember to put FALSE
    def forward(self, x):
        x = self.cnn(x) # Run through the model
        
        sh = x.shape
        x = torch.reshape(x, (sh[1], sh[0], sh[2], sh[3])) # Reshape so that the channels are flattened, not frames
        x = nn.Flatten()(x)
        x = self.gru(x)
      
        
        if self.debug: print(f"Layer's shape: {sh}")
        #x = torch.flatten(x, 1)     # Flatten layer
        #if debug: print(f"  Layer's shape: {x.shape}")
        if self.debug: print(f"Summary of the layer: a")

        return x