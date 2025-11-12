import torch
from models.vit_finetune import ViTFinetune
from models.resnet_finetune import ResNetFinetune
from utils.dataset_utils import train_loader, test_loader, val_loader
from utils.train_utils import train_model, evaluate_model
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available:
        torch.cuda.set_per_process_memory_fraction(0.8, device=0)
        
    print(f"Device: {device}")

    Path("data").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)

    model = ResNetFinetune(num_classes=10, pretrained=True, freeze_backbone=True).to(device)
    history = train_model(model, train_loader, val_loader, device, num_epochs=10, lr=1e-4)

    print("Evaluierung auf Testdaten ...")
    model.load_state_dict(torch.load("checkpoints/best_model.pth"))
    test_acc, test_loss = evaluate_model(model, test_loader, device)

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
    plt.show()

if __name__ == "__main__":
    main()