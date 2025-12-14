import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import torchvision.transforms as transforms

np.random.seed(42)
torch.manual_seed(42)

# -----------------------------------------------------------------------------------------------------

transform_train = transforms.Compose([
    # transforms.Resize(256),
    # transforms.CenterCrop(224),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])

transform_eval = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225))
])

# -----------------------------------------------------------------------------------------------------

food_training_data = datasets.Food101(
    root="./data", split="train", download=True, transform=transform_train)
food_val_data = datasets.Food101(
    root="./data", split="train", download=True, transform=transform_eval)
food_test_data = datasets.Food101(
    root="./data", split="test", download=True, transform=transform_eval)

# -----------------------------------------------------------------------------------------------------

food_val_split = 0.1
num_samples = len(food_training_data)
indices = np.random.permutation(num_samples)

val_size = int(num_samples * food_val_split)
val_indices = indices[:val_size]
train_indices = indices[val_size:]

food_train_ds = Subset(food_training_data, train_indices)
food_val_ds = Subset(food_val_data, val_indices)

# -----------------------------------------------------------------------------------------------------

batch_size = 32
num_workers = 2

food_train_loader = DataLoader(
    food_train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
food_val_loader = DataLoader(
    food_val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
food_test_loader = DataLoader(
    food_test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

food_data_loaders = (food_train_loader, food_test_loader, food_val_loader)
