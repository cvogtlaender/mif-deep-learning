import torch
from torch import nn
from pathlib import Path
from models.vit_finetune import ViTFinetune
from models.resnet_finetune import ResNetFinetune
from utils.dataset_utils import food_data_loaders
from utils.train_utils import train_model, evaluate_model
from utils.plot_utils import show_confusion_matrix, show_loss_acc_plot


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

    show_loss_acc_plot(history=history)
    show_confusion_matrix(model, test_loader, device)


if __name__ == "__main__":
    main()
