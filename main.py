import torch
from torch import nn
import matplotlib.pyplot as plt
from pathlib import Path
from models.vit_finetune import ViTFinetune
from models.resnet_finetune import ResNetFinetune
from utils.dataset_utils import food_data_loaders
from utils.train_utils import train_model, evaluate_model
from utils.confusion_matrix import show_confusion_matrix


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    Path("data").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)

    train_loader, test_loader, val_loader = food_data_loaders

    # model = ViTFinetune(num_classes=101, pretrained=True, freeze_backbone=False).to(device)
    model = ResNetFinetune(num_classes=101, pretrained=True,
                           freeze_backbone=False).to(device)

    criterion = nn.CrossEntropyLoss()
    model, history = train_model(
        model, train_loader, val_loader, device, criterion=criterion, num_epochs=20, lr=1e-4)

    print("Evaluierung auf Testdaten ...")
    model.load_state_dict(torch.load("checkpoints/best_model.pth"))
    test_acc, test_loss = evaluate_model(model, test_loader, criterion, device)

    print(f"Test Accuracy: {test_acc:.2f}% | Test Loss: {test_loss:.4f}")

    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.title("Loss Verlauf")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.title("Accuracy Verlauf")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("checkpoints/loss_acc", dpi=300)
    plt.show()

    show_confusion_matrix(model, test_loader, device)


if __name__ == "__main__":
    main()
