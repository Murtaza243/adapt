import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniCNN(nn.Module):
    def __init__(self, in_channels = 1, num_classes = 10):
        super(MiniCNN, self).__init__()

        #fist layer: 1 Input canal
        self.conv1 = nn.Conv2d(in_channels, 16,kernel_size=3, padding=1)

        #2. layer: 16 Input canal
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

        #choose key
        x = F.relu(self.fc(x))

#call the function
def minicnn(**kwargs):
    return MiniCNN(**kwargs)