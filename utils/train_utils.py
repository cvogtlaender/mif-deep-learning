import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import time

def accuracy(preds, labels):
    """Berechnet Accuracy in Prozent."""
    _, pred_classes = torch.max(preds, 1)
    return (pred_classes == labels).float().mean().item() * 100

def train_model(model, train_loader, val_loader, device, num_epochs=10, lr=1e-4, 
                optimizer_type="adamw", scheduler_type="cosine", save_path="checkpoints/best_model.pth", early_stopping_patience=5):
    
    Path(save_path).parent.mkdir(exist_ok=True)

    criterion = nn.CrossEntropyLoss()

    if optimizer_type.lower() == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    elif optimizer_type.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unbekannter Optimizer: {optimizer_type}")

    if scheduler_type.lower() == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_type.lower() == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    else:
        scheduler = None

    best_val_acc = 0.0
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(num_epochs):
        start_time = time.time()

        model.train()
        train_loss, train_acc = 0.0, 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_acc += accuracy(outputs, labels) * images.size(0)

        train_loss /= len(train_loader.dataset)
        train_acc /= len(train_loader.dataset)

        model.eval()
        val_loss, val_acc = 0.0, 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_acc += accuracy(outputs, labels) * images.size(0)

        val_loss /= len(val_loader.dataset)
        val_acc /= len(val_loader.dataset)

        if scheduler:
            scheduler.step()

        elapsed = time.time() - start_time
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | "
              f"Time: {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Neues bestes Modell gespeichert ({save_path})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            print("Early Stopping aktiviert.")
            break

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

    print(f"\nTraining abgeschlossen. Beste Val Accuracy: {best_val_acc:.2f}%")
    return model, history

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss, total_acc = 0.0, 0.0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            total_acc += accuracy(outputs, labels) * images.size(0)

    total_loss /= len(data_loader.dataset)
    total_acc /= len(data_loader.dataset)
    return total_acc, total_loss
