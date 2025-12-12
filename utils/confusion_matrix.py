import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


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


def show_confusion_matrix(model, data_loader, device, normalize=True, save_path=f"checkpoints/matrix.png"):
    preds, labels = get_predictions(model, data_loader, device)
    class_names = data_loader.dataset.classes
    cm = confusion_matrix(labels, preds)

    plot_confusion_matrix(cm, class_names=class_names,
                          normalize=normalize, save_path=save_path)
