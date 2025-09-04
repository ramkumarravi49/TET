from main_training_parallel import test
import argparse
import shutil
import os
import time
import torch
import warnings
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from torch.utils.tensorboard import SummaryWriter
import pynvml
import matplotlib.pyplot as plt
from models.VGG_models import *
from models.resnet_models import *
import data_loaders
from functions import TET_loss, seed_all, get_logger
from datetime import datetime

train_dataset, val_dataset = data_loaders.build_cifar(cutout=True, use_cifar10=False, download=False)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True,
                                               num_workers=16, pin_memory=True)
test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64,
                                              shuffle=False, num_workers=16, pin_memory=True)

model = vgg11(num_classes=100)
device = 'cuda'
model = torch.load('/home/hiran/Desktop/hiran/cifar100_TET_vgg11.pth', weights_only=False).to(device)

# l = [29.65, 63.2, 68.97, 70.53, 69.85, 66.99]
l = []
# Values of T you want to test
T_values = [2, 4, 6, 8, 16, 32]

for T_val in T_values:
    model.T = T_val
    parallel_model = torch.nn.DataParallel(model)
    parallel_model.to(device)

    facc, _ = test(parallel_model, test_loader, device)
    print(f'Test Accuracy of the model @ T = {model.T}: {facc:.2f}%')

    l.append(facc)

# Bar graph
plt.figure(figsize=(6, 4))
bars = plt.bar([str(t) for t in T_values], l, color='skyblue', edgecolor='black')

# Add labels above bars
for bar, acc in zip(bars, l):
    plt.text(
        bar.get_x() + bar.get_width() / 2,  # x position (center of bar)
        bar.get_height() + 0.05,            # y position slightly above bar
        f"{acc:.2f}",                       # text format
        ha='center', va='bottom', fontsize=10
    )

plt.ylim(28, 80)
plt.xlabel("T (Number of Time Steps)")
plt.ylabel("Test Accuracy (%)")
plt.title("Accuracy vs T")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save figure
plt.tight_layout()
plt.savefig("accuracy_vs_T(TET).png", dpi=300)
plt.show()