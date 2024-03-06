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
    """
    Computes the accuracy of predictions. 
    When the Pc of y_true is 0, no object is present, and the prediction is correct when Pc of y_pred < 0.
    When the Pc of y_true is 1, there is an object present, and the prediction is correct when both the predicted class
    is correct and the IOU of the bounding boxes is > 0.5. 

    :param y_pred: Tensor of shape (batch_size, 15). Looks like
                   y_pred = [[Pc, bx, by, bh, bw, c1, .., c10], [...], ...]
                   Where c1 is the probability of class 1 in box

    :param y_true: Tensor of size (batch_size, 6). Looks like
                   y_true = [[Pc, bx, by, bh, bw, C]], [...], ...]
                   Where C is the class label (a float, but one of integers 0-9)
    """
    correct = 0

    for i, pred in enumerate(y_pred):
        # If Pc == 0 then only Pc is relevant. If predicted Pc is < 0 then confidence is < 0.5 and prediction is correct
        if y_true[i][0] == torch.tensor(0):
            if pred[0] < 0:
                correct += 1

        else: 
            if pred[0] >= 0: # Object presence predicted to be more than 50% likely
                # Get the predicted class
                _, max_index = torch.max(pred[5:])
                if max_index == y_true[i][5]:  # Class prediction is correct
                    # Get corner coords for prediction box
                    b_x_pred, b_y_pred, b_h_pred, b_w_pred = pred[1:5]
                    x_top_pred = b_x_pred + b_w_pred / 2
                    y_top_pred = b_y_pred + b_h_pred / 2
                    x_bottom_pred = b_x_pred - b_w_pred / 2
                    y_bottom_pred = b_y_pred - b_h_pred / 2
                    # Get corner coords for label box
                    b_x_label, b_y_label, b_h_label, b_w_label = y_true[i][1:5]
                    x_top_label = b_x_label + b_w_label / 2
                    y_top_label = b_y_label + b_h_label / 2
                    x_bottom_label = b_x_label - b_w_label / 2
                    y_bottom_label = b_y_label - b_h_label / 2
                    
                    # Calculating the intersection
                    intersection_w = max(0, min(x_top_pred, x_top_label) - max(x_bottom_label, x_bottom_pred))
                    intersection_h = max(0, min(y_top_label, y_top_pred) - max(y_bottom_label, y_bottom_pred))
                    intersection = intersection_w * intersection_h
                    # Calculating the union
                    union = (b_w_label * b_h_label) + (b_w_pred * b_h_pred) - intersection

                    # If IOU > 0.5, the bounding box is "correct"
                    if (intersection / union) > 0.5:
                        correct += 1  # Both bounding box location and predicted class are deemed correct

    return correct / len(y_pred)


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

<<<<<<< HEAD
        losses_train.append(loss_train / n_batch)
=======
        losses_train.append(loss_train / n_batch_train)

>>>>>>> 12df7a4fe2ec1ecd54c559305fff5c5463c0cb08
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
            print(f"y_pred shape: {outputs.shape}")
            print(f"y_true shape: {labels.shape}")

    return losses_train, losses_val
