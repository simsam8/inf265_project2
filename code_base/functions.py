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
    When the Pc of y_true is 0, no object is present,
    and the prediction is correct when Pc of y_pred < 0.
    When the Pc of y_true is 1, there is an object present,
    and the prediction is correct when both the predicted class
    is correct and the IOU of the bounding boxes is > 0.5.

    :param y_pred: Tensor of shape (batch_size, 15). Looks like
                   y_pred = [[Pc, bx, by, bh, bw, c1, .., c10], [...], ...]
                   Where c1 is the probability of class 1 in box

    :param y_true: Tensor of size (batch_size, 6). Looks like
                   y_true = [[Pc, bx, by, bh, bw, C]], [...], ...]
                   Where C is the class label (a float, but one of integers 0-9)
    :return: triple of (accuracy, number of correct predictions, length of y_pred).
            The last two are included so that it is possible to
            keep track of the running average across batches/epochs.
    """
    correct = 0

    with torch.no_grad():
        for i, pred in enumerate(y_pred):
            # If Pc == 0 then only Pc is relevant.
            # If predicted Pc is < 0 then confidence is < 0.5 and prediction is correct
            if y_true[i][0] == torch.tensor(0):
                if pred[0] < 0:
                    correct += 1

            else:
                if pred[0] >= 0:  # Object presence predicted to be more than 50% likely
                    # Get the predicted class
                    max_value, max_index = torch.max(pred[5:], 0)
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
                        intersection_w = max(
                            0,
                            min(x_top_pred, x_top_label)
                            - max(x_bottom_label, x_bottom_pred),
                        )
                        intersection_h = max(
                            0,
                            min(y_top_label, y_top_pred)
                            - max(y_bottom_label, y_bottom_pred),
                        )
                        intersection = intersection_w * intersection_h
                        # Calculating the union
                        union = (
                            (b_w_label * b_h_label)
                            + (b_w_pred * b_h_pred)
                            - intersection
                        )

                        # If IOU > 0.5, the bounding box is "correct"
                        # Both bounding box location and predicted class are deemed correct
                        if (intersection / union) > 0.5:
                            correct += 1

    return correct / len(y_pred), correct, len(y_pred)


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
    performance_train = []
    performance_val = []
    model.train()
    optimizer.zero_grad()

    for epoch in range(1, n_epochs + 1):

        loss_train = 0.0
        n_correct_train = 0.0
        total_predictions_train = 0
        loss_val = 0.0
        n_correct_val = 0.0
        total_predictions_val = 0

        for i, (imgs, labels) in enumerate(train_loader):

            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_train += loss.item()

            # Keep track of performance
            _, batch_n_correct, n_preds = compute_performance(outputs, labels, device)
            total_predictions_train += n_preds
            n_correct_train += batch_n_correct

        losses_train.append(loss_train / n_batch_train)
        performance_train.append(n_correct_train / total_predictions_train)

        model.eval()
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                outputs = model(imgs)
                loss = loss_fn(outputs, labels)
                loss_val += loss.item()
                _, batch_n_correct, n_preds = compute_performance(
                    outputs, labels, device
                )
                total_predictions_val += n_preds
                n_correct_val += batch_n_correct
            losses_val.append(loss_val / n_batch_val)
            performance_val.append(n_correct_val / total_predictions_val)

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"{datetime.now().time()}\n"
                f"Epoch: {epoch}\n"
                f"train_loss:         {loss_train/n_batch_train:>10.3f}\n"
                f"val_loss:           {loss_val/n_batch_val:>10.3f}\n"
                f"train_performance:  {((n_correct_train / total_predictions_train)*100):>10.3f}%\n"
                f"val_performance:    {((n_correct_val / total_predictions_val)*100):>10.3f}%\n"
            )

    return losses_train, losses_val, performance_train, performance_val


def train_models(
    networks: list[nn.Module],
    hyper_parameters: dict,
    batch_size: int,
    n_epochs: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    seed: int,
):
    """
    Trains and returns models with different hyper parameters and
    model architectures.

    Parameters:
    networks: list of model architectures
    hyper_parameters: hyper parameters to train on
    batch_size: batch size
    n_epochs: epochs
    train_loader: DataLoader for training set
    val_loader: DataLoader for validation set
    device: device to train on
    seed: seed for randomization
    """
    loss_fn = custom_loss  # using our implemented loss function

    print("\tGlobal parameters:")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {n_epochs}")
    print(f"Seed: {seed}")

    models = []
    train_performances = []
    val_performances = []
    train_losses = []
    val_losses = []

    # Hyperparameter testing on each defined model architecture
    for network in networks:
        for hparam in hyper_parameters:
            print("\n", "=" * 50)
            print(f"Model architecture: {network}\n")
            print("\tCurrent parameters: ")
            [print(f"{key}:{value}") for key, value in hparam.items()]

            model = network()
            model.to(device)
            # TODO?: Using only SGD at the moment, could possibly add optimizers
            # as parameters.
            optimizer = optim.SGD(model.parameters(), **hparam)

            print("Starting training using above parameters:\n")
            train_loss, val_loss, train_performance, val_performance = train(
                n_epochs, optimizer, model, loss_fn, train_loader, val_loader, device
            )

            models.append(model)
            train_performances.append(train_performance)
            val_performances.append(val_performance)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print("\n", "-" * 3, "Performance", "-" * 3)
            print(f"Training performance: {train_performance[-1]*100:.2f}%")
            print(f"Validation performance: {val_performance[-1]*100:.2f}%")

    return models, train_performances, val_performances, train_losses, val_losses


def select_best_model(
    models: list[nn.Module], val_performances: list[list[float]]
) -> tuple[nn.Module, int]:
    """
    Selects the model with highest validation performance.

    Parameters:
    models: list of trained models
    val_performances: list of validation performances of models

    return: selected_model, index of model
    """
    # TODO?: Might change the input list of val performance from list of lists
    # to only the final performance value
    last_val_performances = [val_perf[-1] for val_perf in val_performances]
    selected_idx = last_val_performances.index(max(last_val_performances))
    selected_model = models[selected_idx]

    return selected_model, selected_idx
