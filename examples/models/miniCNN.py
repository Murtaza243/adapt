import torch
import torch.nn as nn
import torch.nn.functional as F

from adapt.approx_layers import axx_layers as approxNN

class MiniCNN(nn.Module):
    def __init__(self, in_channels = 1, num_classes = 10, axx_mult = None):
        super(MiniCNN, self).__init__()

        if axx_mult is not None:
            self.conv1 = approxNN.AdaPT_Conv2d(in_channels, 16, kernel_size=3, padding=1, axx_mult=axx_mult)
            self.conv2 = approxNN.AdaPT_Conv2d(16, 32, kernel_size=3, padding=1, axx_mult=axx_mult)
        else:
            self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        #3. layer: the linear output
        self.fc = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x):
        #activate and cut the picture in 2 part
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        #activate and cut the picture in 2 part again
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        #to pound flot
        x = x.view(-1, 32 * 7 * 7)
        x = self.fc(x)
        return x
        

#call the function
def minicnn(**kwargs):
    return MiniCNN(**kwargs)