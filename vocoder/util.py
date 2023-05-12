import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1dBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size = 3, padding = 1, bias = True)
    
    def forward(self, x):
        return F.relu(self.conv(x))