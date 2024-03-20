import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision
from datetime import datetime
import time
from .object_detection import MAP_preprocess


def time_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time-start_time} seconds to complete.")
        return result

    return wrapper


def localization_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Calculates the localization loss.
    If there is no object (y_true[0] == 0), then only detection loss is calculated.
    Otherwise the sum of detection, bounding box and classification loss is calculated.
    """
    detection_loss = F.binary_cross_entropy_with_logits(y_pred[:, 0], y_true[:, 0])
    bbox_loss = F.mse_loss(y_pred[:, 1:5], y_true[:, 1:5])
    classification_loss = F.cross_entropy(y_pred[:, 5:], y_true[:, 5].to(torch.long))

    loss = torch.mean(
        torch.where(
            y_true[:, 0] == 0,
            detection_loss,
            detection_loss + bbox_loss + classification_loss,
        ),
        dim=0,
    )
    return loss


def _grid_cell_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Intermediate function to calculate the loss sum of grid cells.
    """
    detection_loss = F.binary_cross_entropy_with_logits(y_pred[:, 0], y_true[:, 0])
    bbox_loss = F.mse_loss(y_pred[:, 1:5], y_true[:, 1:5])
    classification_loss = F.binary_cross_entropy_with_logits(
        y_pred[:, 5], y_true[:, 5].to(torch.long)
    )

    loss = torch.sum(
        torch.where(
            y_true[:, 0] == 0,
            detection_loss,
            detection_loss + bbox_loss + classification_loss,
        ),
        dim=0,
    )
    return loss


def detection_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Calculate the detection loss
    """
    y_true = y_true.permute(0, 2, 3, 1).flatten(1, 2)
    y_pred = y_pred.permute(0, 2, 3, 1).flatten(1, 2)

    vectorized_grid_cell_loss = torch.vmap(_grid_cell_loss)
    loss = torch.mean(vectorized_grid_cell_loss(y_pred, y_true))
    return loss


def compute_performance(
    y_pred: torch.Tensor, y_true: torch.Tensor, iou_threshold: float = 0.5
) -> dict:
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
    :return: Dict with performance metrics.
    """

    results = {
        "detection_correct": 0,
        "strict_correct": 0,
        "box_correct": 0,
        "total_box_predictions": 0,
        "n_predictions": len(y_pred),
        "detection_accuracy": 0,
        "mean_accuracy": 0,
        "box_accuracy": 0,
        "strict_accuracy": 0,
        "n_boxes_true": (y_true[:, 0] == 1).sum().item(),
    }

    with torch.no_grad():
        for i, pred in enumerate(y_pred):
            # If Pc == 0 then only Pc is relevant.
            # If predicted Pc is < 0 then confidence is < 0.5 and prediction is correct
            if y_true[i][0] == torch.tensor(0):
                if pred[0] < 0:
                    results["strict_correct"] += 1
                    results["detection_correct"] += 1

            else:
                if pred[0] >= 0:  # Object presence predicted to be more than 50% likely
                    results["detection_correct"] += 1
                    results["total_box_predictions"] += 1
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
                        if (intersection / union) > iou_threshold:
                            results["strict_correct"] += 1
                            results["box_correct"] += 1
    # Compute metrics
    if results["total_box_predictions"] == 0:
        pass
    else:
        results["box_accuracy"] = (
            results["box_correct"] / results["total_box_predictions"]
        )
    results["detection_accuracy"] = (
        results["detection_correct"] / results["n_predictions"]
    )
    results["mean_accuracy"] = (
        results["box_accuracy"] + results["detection_accuracy"]
    ) / 2
    if results["n_boxes_true"] == 0:
        pass
    else:
        results["strict_accuracy"] = results["box_correct"] / results["n_boxes_true"]
    return results


def train(
    task: str,
    n_epochs: int,
    optimizer: optim.Optimizer,
    model: nn.Module,
    loss_fn,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> dict[str, list[float]]:
    """
    Trains a given model with optimizer. Keeps track of training and
    validation performance and loss.

    Returns a dictionary containing lists of training and validation
    loss and performance. The lists contains values for each epoch.

    keys:
    - loss_train
    - loss_val
    - detection_train
    - detection_val
    - box_train
    - box_val
    - mean_perf_train
    - mean_perf_val
    - strict_train
    - strict_val
    """

    n_batch_train = len(train_loader)
    n_batch_val = len(val_loader)
    losses_train = []
    losses_val = []
    strict_performance_train = []
    strict_performance_val = []
    mean_performance_train = []
    mean_performance_val = []
    detection_performance_train = []
    detection_performance_val = []
    box_performance_train = []
    box_performance_val = []
    model.train()
    optimizer.zero_grad()

    for epoch in range(1, n_epochs + 1):
        if task == "detection":
            train_metric = MeanAveragePrecision(
                box_format="cxcywh", iou_type="bbox", extended_summary=True
            )
            val_metric = MeanAveragePrecision(
                box_format="cxcywh", iou_type="bbox", extended_summary=True
            )
        else:
            # Initialize performance values
            train_total_strict_correct = 0
            train_total_box_correct = 0
            train_total_box_preds = 0
            train_total_predictions = 0
            train_total_detection_correct = 0
            val_total_strict_correct = 0
            val_total_box_correct = 0
            val_total_predictions = 0
            val_total_detection_correct = 0
            val_total_box_preds = 0
        loss_train = 0.0
        loss_val = 0.0

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
            if task == "localization":
                results = compute_performance(outputs, labels)
                train_total_predictions += results["n_predictions"]
                train_total_detection_correct += results["detection_correct"]
                train_total_box_correct += results["box_correct"]
                train_total_box_preds += results["total_box_predictions"]
                train_total_strict_correct += results["strict_correct"]
            elif task == "detection":
                outputs_prep = MAP_preprocess(outputs)
                labels_prep = MAP_preprocess(labels)
                train_metric.update(outputs_prep, labels_prep)

        losses_train.append(loss_train / n_batch_train)
        if task == "localization":
            box_performance_train.append(
                train_total_box_correct / train_total_box_preds
            )
            detection_performance_train.append(
                train_total_detection_correct / train_total_predictions
            )
            strict_performance_train.append(
                train_total_strict_correct / train_total_predictions
            )
            mean_performance_train.append(
                (detection_performance_train[-1] + box_performance_train[-1]) / 2
            )
        else:
            map_metrics = train_metric.compute()
            strict_performance_train.append(map_metrics["map"])

        model.eval()
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                outputs = model(imgs)
                loss = loss_fn(outputs, labels)
                loss_val += loss.item()
                # Keep track of performance
                if task == "localization":
                    results = compute_performance(outputs, labels)
                    val_total_detection_correct += results["detection_correct"]
                    val_total_box_correct += results["box_correct"]
                    val_total_predictions += results["n_predictions"]
                    val_total_box_preds += results["total_box_predictions"]
                    val_total_strict_correct += results["strict_correct"]

                elif task == "detection":
                    outputs_prep = MAP_preprocess(outputs)
                    labels_prep = MAP_preprocess(labels)
                    val_metric.update(outputs_prep, labels_prep)

        losses_val.append(loss_val / n_batch_val)
        if task == "localization":
            box_performance_val.append(val_total_box_correct / val_total_box_preds)
            detection_performance_val.append(
                val_total_detection_correct / val_total_predictions
            )
            mean_performance_val.append(
                (detection_performance_val[-1] + box_performance_val[-1]) / 2
            )
            strict_performance_val.append(
                val_total_strict_correct / val_total_predictions
            )
        elif task == "detection":
            map_metrics = val_metric.compute()
            strict_performance_val.append(map_metrics["map"])

        if epoch == 1 or epoch % 5 == 0:
            if task == "localization":
                print(
                    f"{datetime.now().time()}\n"
                    f"Epoch: {epoch}\n"
                    f"train_loss:         {losses_train[-1]:>10.3f}\n"
                    f"val_loss:           {losses_val[-1]:>10.3f}\n"
                    f"train_performance:  \n\n    \
                        Box accuracy: {((box_performance_train[-1])*100):>10.3f}% \
                        Detection accuracy: {((detection_performance_train[-1])*100):>10.3f}% \n\
                        Mean accuracy: {((mean_performance_train[-1])*100):>10.3f}% \
                        Strict accuracy: {((strict_performance_train[-1])*100):>10.3f}%\n\n"
                    f"val_performance:    \n    \
                        Box accuracy: {((box_performance_val[-1])*100):>10.3f}% \
                        Detection accuracy: {((detection_performance_val[-1])*100):>10.3f}% \n\
                        Mean accuracy: {((mean_performance_val[-1])*100):>10.3f}% \
                        Strict accuracy: {((strict_performance_val[-1])*100):>10.3f}%\n\n\n"
                )
            elif task == "detection":
                print(
                    f"{datetime.now().time()}\n"
                    f"Epoch: {epoch}\n"
                    f"train_loss:         {losses_train[-1]:>10.3f}\n"
                    f"val_loss:           {losses_val[-1]:>10.3f}\n"
                    f"train_performance:\nStrict accuracy: {((strict_performance_train[-1])*100):>10.3f}%\n\n"
                    f"val_performance:\nStrict accuracy: {((strict_performance_val[-1])*100):>10.3f}%\n\n\n"
                )

    training_result = {
        "loss_train": losses_train,
        "loss_val": losses_val,
        "detection_train": detection_performance_train,
        "detection_val": detection_performance_val,
        "box_train": box_performance_train,
        "box_val": box_performance_val,
        "mean_perf_train": mean_performance_train,
        "mean_perf_val": mean_performance_val,
        "strict_train": strict_performance_train,
        "strict_val": strict_performance_val,
    }
    return training_result


def train_models(
    task: str,
    networks: list[nn.Module],
    hyper_parameters: dict,
    batch_size: int,
    n_epochs: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    seed: int,
) -> dict[str, list[nn.Module | float]]:
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
    if task == "localization":
        loss_fn = localization_loss  # using our implemented loss function
    elif task == "detection":
        loss_fn = detection_loss
    else:
        raise RuntimeError("set task to 'localization' or 'detection'")
    print("\tGlobal parameters:")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {n_epochs}")
    print(f"Seed: {seed}")

    grid_search_result = {
        "models": [],
        "loss_train": [],
        "loss_val": [],
        "detection_train": [],
        "detection_val": [],
        "box_train": [],
        "box_val": [],
        "mean_perf_train": [],
        "mean_perf_val": [],
        "strict_train": [],
        "strict_val": [],
    }

    # Hyperparameter testing on each defined model architecture
    for network in networks:
        for hparam in hyper_parameters:
            print("\n", "=" * 50)
            print(f"Model architecture: {network}\n")
            print("\tCurrent parameters: ")
            [print(f"{key}:{value}") for key, value in hparam.items()]

            model = network()
            model.to(device)
            optimizer = optim.Adam(model.parameters(), **hparam)

            print(f"Starting training for {task} using above parameters:\n")
            train_results = train(
                task,
                n_epochs,
                optimizer,
                model,
                loss_fn,
                train_loader,
                val_loader,
                device,
            )

            grid_search_result["models"].append(model)
            grid_search_result["loss_train"].append(train_results["loss_train"])
            grid_search_result["loss_val"].append(train_results["loss_val"])
            grid_search_result["detection_train"].append(
                train_results["detection_train"]
            )
            grid_search_result["detection_val"].append(train_results["detection_val"])
            grid_search_result["box_train"].append(train_results["box_train"])
            grid_search_result["box_val"].append(train_results["box_val"])
            grid_search_result["mean_perf_train"].append(
                train_results["mean_perf_train"]
            )
            grid_search_result["mean_perf_val"].append(train_results["mean_perf_val"])
            grid_search_result["strict_train"].append(train_results["strict_train"])
            grid_search_result["strict_val"].append(train_results["strict_val"])

            # print("\n", "-" * 3, "Performance", "-" * 3)
            # print(f"Training performance: {train_performance[-1]*100:.2f}%")
            # print(f"Validation performance: {val_performance[-1]*100:.2f}%")

    return grid_search_result


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
    last_val_performances = [val_perf[-1] for val_perf in val_performances]
    selected_idx = last_val_performances.index(max(last_val_performances))
    selected_model = models[selected_idx]

    return selected_model, selected_idx


def evaluate_performance(
    task: str, model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[dict[str, float | int], torch.Tensor]:
    """
    Evaluate the performance of a model.
    """
    strict_performance = 0
    mean_performance = 0
    detection_performance = 0
    box_performance = 0
    model_outputs = []

    if task == "localization":
        total_strict_correct = 0
        total_box_correct = 0
        total_box_preds = 0
        total_predictions = 0
        total_detection_correct = 0
    elif task == "detection":
        metric = MeanAveragePrecision(box_format="cxcywh", iou_type="bbox")

    model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            model_outputs.append(outputs)
            if task == "localization":
                results = compute_performance(outputs, labels)
                total_predictions += results["n_predictions"]
                total_detection_correct += results["detection_correct"]
                total_box_correct += results["box_correct"]
                total_box_preds += results["total_box_predictions"]
                total_strict_correct += results["strict_correct"]
            elif task == "detection":
                outputs_prep = MAP_preprocess(outputs)
                labels_prep = MAP_preprocess(labels)
                metric.update(outputs_prep, labels_prep)

        if task == "localization":
            box_performance = total_box_correct / total_box_preds
            detection_performance = total_detection_correct / total_predictions
            strict_performance = total_strict_correct / total_predictions
            mean_performance = (detection_performance + box_performance) / 2
        elif task == "detection":
            strict_performance = metric.compute()["map"]

    model_output = torch.concat(model_outputs)

    evaluation_result = {
        "box": box_performance,
        "detection": detection_performance,
        "strict": strict_performance,
        "mean": mean_performance,
    }
    return evaluation_result, model_output
