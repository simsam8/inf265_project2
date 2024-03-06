import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime


def _total_loss(y_pred: torch.Tensor, y_true: torch.Tensor):
    loss_BCE = nn.BCEWithLogitsLoss()
    loss_mse = nn.MSELoss()
    loss_CE = nn.CrossEntropyLoss()

    detection = loss_BCE(y_pred[:, 0], y_true[:, 0])
    localization = (
        loss_mse(y_pred[:, 1], y_true[:, 1])
        + loss_mse(y_pred[:, 2], y_true[:, 2])
        + loss_mse(y_pred[:, 3], y_true[:, 3])
        + loss_mse(y_pred[:, 4], y_true[:, 4])
    )
    classification = loss_CE(y_pred[:, 5:], y_true[:, 5].to(torch.long))

    return detection + localization + classification


def _detection_loss(y_pred: torch.Tensor, y_true: torch.Tensor):
    loss_BCE = nn.BCEWithLogitsLoss()
    detection = loss_BCE(y_pred[:, 0], y_true[:, 0])
    return detection


def custom_loss(y_pred: torch.Tensor, y_true: torch.Tensor):
    no_detection_mask = y_true[:, 0] == 0

    no_object_labels, no_object_preds = (
        y_true[no_detection_mask],
        y_pred[no_detection_mask],
    )
    is_object_labels, is_object_preds = (
        y_true[~no_detection_mask],
        y_pred[~no_detection_mask],
    )

    detection_loss = _detection_loss(no_object_preds, no_object_labels)
    combined_loss = _total_loss(is_object_preds, is_object_labels)

    return detection_loss + combined_loss


def compute_performance(
    y_pred: torch.Tensor, y_true: torch.Tensor, device: torch.device
):
    # TODO: Implement accuracy calculation where p_c of y_true is 1
    # TODO: Intersection Over Union for bounding boxes
    # overall performance = mean(accuracy and IoU)
    pass


# TODO: Compute accuracy each epoch
def train(
    n_epochs: int,
    optimizer: optim.Optimizer,
    model: nn.Module,
    loss_fn,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
):

    n_batch_train = len(train_loader)
    n_batch_val = len(val_loader)
    losses_train = []
    losses_val = []
    model.train()
    optimizer.zero_grad()

    for epoch in range(1, n_epochs + 1):

        loss_train = 0.0
        loss_val = 0.0

        for imgs, labels in train_loader:

            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_train += loss.item()

        losses_train.append(loss_train / n_batch_train)

        model.eval()
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                outputs = model(imgs)
                loss = loss_fn(outputs, labels)
                loss_val += loss.item()
            losses_val.append(loss_val / n_batch_val)

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"{datetime.now().time()}, {epoch}, train_loss: {loss_train/n_batch_train}, val_loss: {loss_val/n_batch_val}"
            )

    return losses_train, losses_val
