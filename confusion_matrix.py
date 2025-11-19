import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from models.resnet_finetune import ResNetFinetune
from models.vit_finetune import ViTFinetune
from utils.dataset_utils import food_data_loaders


def get_predictions(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(cm, class_names, normalize=True, save_path="confusion_matrix.png"):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(18, 16))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix (Normalized)", fontsize=18)
    plt.colorbar(shrink=0.7)

    step = 10 if len(class_names) > 50 else 1
    ticks = np.arange(0, len(class_names), step)
    tick_labels = [class_names[i] for i in ticks]

    plt.xticks(ticks, tick_labels, rotation=90, fontsize=6)
    plt.yticks(ticks, tick_labels, fontsize=6)

    plt.tight_layout()
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(save_path, dpi=300)
    plt.show()


def main():
    checkpoint_path = "checkpoints/best_model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, test_loader, val_loader = food_data_loaders
    class_names = test_loader.dataset.classes
    num_classes = len(class_names)

    model = ResNetFinetune(
        num_classes=num_classes, pretrained=True, freeze_backbone=False
    )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)

    preds, labels = get_predictions(model, test_loader, device)

    cm = confusion_matrix(labels, preds)

    plot_confusion_matrix(
        cm,
        class_names=class_names,
        normalize=True,
        save_path=f"checkpoints/matrix.png",
    )


if __name__ == "__main__":
    main()