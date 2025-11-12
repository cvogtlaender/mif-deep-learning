from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import torchvision.transforms as transforms

transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


training_data = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
test_data = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

batch_size = 32
val_split = 0.1
val_size = int(len(training_data) * val_split)
train_size = len(training_data) - val_size
train_ds, val_ds = random_split(training_data, [train_size, val_size])

# Create data loaders.
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)
