from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import torchvision.transforms as transforms

transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406),
                         std=(0.229,0.224,0.225))
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406),
                         std=(0.229,0.224,0.225))
])

#---------------------------------------------------------------------------------------------------------------------

food_training_data = datasets.Food101(root="./data", split="train", download=True, transform=transform_train)
food_test_data = datasets.Food101(root="./data", split="test", download=True, transform=transform_test)

food_batch_size = 32
food_val_split = 0.1
food_val_size = int(len(food_training_data) * food_val_split)
food_train_size = len(food_training_data) - food_val_size
food_train_ds, food_val_ds = random_split(food_training_data, [food_train_size, food_val_size])

food_train_loader = DataLoader(food_train_ds, batch_size=food_batch_size, shuffle=True, num_workers=2)
food_val_loader = DataLoader(food_val_ds, batch_size=food_batch_size, shuffle=False, num_workers=2)
food_test_loader = DataLoader(food_test_data, batch_size=food_batch_size, shuffle=False, num_workers=2)

food_data_loaders = (food_train_loader, food_test_loader, food_val_loader)
