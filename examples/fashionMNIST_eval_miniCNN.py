# %% [markdown]
# # Model evaluation and re=training with Adapt (miniCNN) on FashionMNIST dataset.

# %%
import os 
import sys
import timeit
import torch 
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# for minicnn!!!
sys.path.append(os.path.abspath('examples'))

from models.miniCNN import MiniCNN, minicnn
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from adapt.references.classification.train import train_one_epoch

# %% [markdown]
# ## Hardware and Thread setup.
# %%
threads = 40 
torch.set_num_threads(threads)
print("threads are ready")

#Setup multiplicator
axx_mult = "mul8s_1L2H"


# %% [markdown]
# ## load Datasets (FashionMNIST)
# %%
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# testdataset

testset = datasets.FashionMNIST(root='/.data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)

# load training data and subset for fine tuning
trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
subset_indices = list(range(0, len(trainset), 10))
trainset_subset = Subset(trainset, subset_indices)
data_t = DataLoader(trainset_subset, batch_size=128, shuffle=True, num_workers=0)
print("data is ready!")


# %% [markdown]
# ## train with minicnn to get the weights
# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time 

from models.miniCNN import minicnn, MiniCNN

# load FashionMNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

# Setup the model
model = minicnn(in_channels=1)
model = model.to("cuda")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 200 
print("Start Training!!")

#trainigloop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()

    for inputs, labels in trainloader:
        inputs, labels = inputs.to("cuda"), labels.to("cuda")

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_time = time.time() - start_time
    print(f"Epoch {epoch+1}/{epochs} | accuracy: {100. * correct /total:.2f}% | time: {epoch_time:.1f}s")

# save the weights
torch.save(model.state_dict(), "fashionmnist_minicnn_weights.pt")
print("weihts are ready in 'fashionmnist_minicnn_weights.pt'!!!")


# %% [markdown]
# load miniCNN for EHaluation 
# %%
model = minicnn(in_channels=1, axx_mult=axx_mult)
model = model.to("cpu")

# load weights
saved_weights = torch.load("fashionmnist_minicnn_weights.pt")
model.load_state_dict(saved_weights, strict=False)
model.eval()
print("Model loaded with trained weights")

# %% [markdown]
# run model calibration for quantization
# %%
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib

def collect_state(model, data_loader, num_batches):
    # enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if getattr(module, '_calibrator', None) is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    for i, (image, _) in tqdm(enumerate(data_loader), total=num_batches):
        model(image.cpu())
        if i >= num_batches:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if getattr(module, '_calibrator', None) is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()

def compute_amax(model, **kwargs):
    #load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if getattr(module, '_calibrator', None) is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
            print(F"{name:40}: {module}")
model.cpu()

print("Start calibration")
with torch.no_grad():
    stats = collect_state(model, data_t, num_batches=2)
    amax = compute_amax(model, method="percentile",percentile=99.99)
print("end calibration")

# %% [markdown]
# first Evaluation
# %%
correct = 0
total = 0
model.eval()

print("start first evaluation")
start_time = timeit.default_timer()

with torch.no_grad():
    for images, labels in tqdm(testloader, total=len(testloader)):
        images, labels = images.to("cpu"), labels.to("cpu")
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"time: {timeit.default_timer() - start_time:.2f}s")
print(f"Accuracy befor re-training: {100 * correct / total:.2f} %")

# %% [markdown]
#retraining for 20 epochs
# %%
from adapt.references.classification.train import train_one_epoch

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

epochs = 20
print("start re-training")

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    train_one_epoch(model, criterion, optimizer, data_t, "cpu", epoch, 1)

# %% [markdown]
# second evaluation
# %%
correct = 0
total = 0
model.eval()

print("start evaluation")
with torch.no_grad():
    for images, labels in tqdm(testloader, total=len(testloader)):
        images, labels = images.to("cpu"), labels.to("cpu")
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy after re-training: {100 * correct / total:.2f} %")
