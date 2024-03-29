import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision
from datetime import datetime
from .object_detection import MAP_preprocess, compute_loc_performance


class EarlyStop:
    """
    Class implementation of early stopping during training.
    """
    def __init__(self, patience=1, min_delta=0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if (
            validation_loss > (self.min_validation_loss + self.min_delta)
            or abs(validation_loss - self.min_validation_loss) <= 1e-4
        ):
            self.counter += 1
            print(f"__Stopping training in {self.patience-self.counter} epochs")
            if self.counter >= self.patience:
                return True
            return False
        elif validation_loss < self.min_validation_loss:
            print("__Patience reset") if self.counter != 0 else None
            self.min_validation_loss = validation_loss
            self.counter = 0


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
    Calculates the detection loss
    """
    y_true = y_true.permute(0, 2, 3, 1).flatten(1, 2)
    y_pred = y_pred.permute(0, 2, 3, 1).flatten(1, 2)

    vectorized_grid_cell_loss = torch.vmap(_grid_cell_loss)
    loss = torch.mean(vectorized_grid_cell_loss(y_pred, y_true))
    return loss


def train(
    task: str,
    n_epochs: int,
    optimizer: optim.Optimizer,
    model: nn.Module,
    loss_fn,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    early_stop_patience: int | None = None,
    early_stop_min_delta: float = 0.0,
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
    model.train()
    optimizer.zero_grad()

    # Loss and performance metrics
    running_metrics = {metric: list() for metric in [
        "detection_performance_train", "detection_performance_val", 
        "box_performance_train", "box_performance_val", 
        "mean_performance_train", "mean_performance_val", 
        "strict_performance_train", "strict_performance_val", 
        "losses_train", "losses_val"
    ]
    }

    if early_stop_patience is not None:
        early_stopper = EarlyStop(early_stop_patience, early_stop_min_delta)

    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        loss_val = 0.0
        if task == "detection":
            train_metric = MeanAveragePrecision(
                box_format="cxcywh", iou_type="bbox", extended_summary=True
            )
            val_metric = MeanAveragePrecision(
                box_format="cxcywh", iou_type="bbox", extended_summary=True
            )
        elif task=="localization" or task=="localisation":
            # For tracking intermediate performance values
            epoch_loc_preds = {i: list() for i in [
                "train_strict_correct", "train_box_correct", 
                "train_box_preds", "train_predictions", 
                "train_detection_correct", "val_total_strict_correct", 
                "val_box_correct", "val_total_predictions", 
                "val_detection_correct", "val_box_preds"
                ]
            }
        else: 
            raise RuntimeError("Set task='localization' or task='detection'")

        for imgs, labels in train_loader:

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
                results = compute_loc_performance(outputs, labels)
                epoch_loc_preds["train_predictions"] += results["n_predictions"]
                epoch_loc_preds["train_detection_correct"] += results["detection_correct"]
                epoch_loc_preds["train_box_correct"] += results["box_correct"]
                epoch_loc_preds["train_box_preds"] += results["total_box_predictions"]
                epoch_loc_preds["train_strict_correct"] += results["strict_correct"]
            elif task == "detection":
                outputs_prep = MAP_preprocess(outputs)
                labels_prep = MAP_preprocess(labels)
                train_metric.update(outputs_prep, labels_prep)

        loss_train /= n_batch_train

        running_metrics["losses_train"].append(loss_train)

        # Calculating train performance
        if task == "localization":
            running_metrics["box_performance_train"].append(
                epoch_loc_preds["train_box_correct"] / epoch_loc_preds["train_box_preds"]
            )
            running_metrics["detection_performance_train"].append(
                epoch_loc_preds["train_detection_correct"] / epoch_loc_preds["train_predictions"]
            )
            running_metrics["strict_performance_train"].append(
                epoch_loc_preds["train_strict_correct"] / epoch_loc_preds["train_predictions"]
            )
            running_metrics["mean_performance_train"].append(
                (running_metrics["detection_performance_train"][-1] + running_metrics["box_performance_train"][-1]) / 2
            )
        else:
            map_metrics = train_metric.compute()
            running_metrics["strict_performance_train"].append(map_metrics["map"])

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
                    results = compute_loc_performance(outputs, labels)
                    epoch_loc_preds["val_detection_correct"] += results["detection_correct"]
                    epoch_loc_preds["val_box_correct"] += results["box_correct"]
                    epoch_loc_preds["val_total_predictions"] += results["n_predictions"]
                    epoch_loc_preds["val_box_preds"] += results["total_box_predictions"]
                    epoch_loc_preds["val_total_strict_correct"] += results["strict_correct"]

                elif task == "detection":
                    outputs_prep = MAP_preprocess(outputs)
                    labels_prep = MAP_preprocess(labels)
                    val_metric.update(outputs_prep, labels_prep)

        loss_val /= n_batch_val
        running_metrics["losses_val"].append(loss_val)

        # Calculating validation performance
        if task == "localization":
            running_metrics["box_performance_val"].append(epoch_loc_preds["val_box_correct"] / epoch_loc_preds["val_box_preds"])
            running_metrics["detection_performance_val"].append(
                epoch_loc_preds["val_detection_correct"] / epoch_loc_preds["val_total_predictions"]
            )
            running_metrics["mean_performance_val"].append(
                (running_metrics["detection_performance_val"][-1] + running_metrics["box_performance_val"][-1]) / 2
            )
            running_metrics["strict_performance_val"].append(
                epoch_loc_preds["val_total_strict_correct"] / epoch_loc_preds["val_total_predictions"]
            )
        elif task == "detection":
            map_metrics = val_metric.compute()
            running_metrics["strict_performance_val"].append(map_metrics["map"])

        if epoch == 1 or epoch % 10 == 0:
            if task == "localization":
                log_output = (
                    f"\n| {datetime.now().time()} | "
                    + f"Epoch: {epoch} | "
                    + f"train_loss: {running_metrics["losses_train"][-1]:.3f} | "
                    + f"val_loss: {running_metrics["losses_val"][-1]:.3f} |\n"
                    + "training: "
                    + f"| Box accuracy: {running_metrics["box_performance_train"][-1]*100:.3f}% | "
                    + f"Detection accuracy: {running_metrics["detection_performance_train"][-1]*100:.3f}% | "
                    + f"Mean accuracy: {running_metrics["mean_performance_train"][-1]*100:.3f}% | "
                    + f"Strict accuracy: {running_metrics["strict_performance_train"][-1]*100:.3f}% |\n"
                    + "validation: "
                    + f"| Box accuracy: {running_metrics["box_performance_val"][-1]*100:.3f}% | "
                    + f"Detection accuracy: {running_metrics["detection_performance_val"][-1]*100:.3f}% | "
                    + f"Mean accuracy: {running_metrics["mean_performance_val"][-1]*100:.3f}% | "
                    + f"Strict accuracy: {running_metrics["strict_performance_val"][-1]*100:.3f}% |"
                )
                print(log_output)
            elif task == "detection":
                log_output = (
                    f"\n| {datetime.now().time()} | "
                    + f"Epoch: {epoch} | "
                    + f"train_loss: {running_metrics["losses_train"][-1]:.3f} | "
                    + f"val_loss: {running_metrics["losses_val"][-1]:.3f} | "
                    + f"train strict accuracy: {running_metrics["strict_performance_train"][-1]*100:.3f}% | "
                    + f"val Strict accuracy: {running_metrics["strict_performance_val"][-1]*100:.3f}% |"
                )
                print(log_output)

        if early_stop_patience is not None:
            if early_stopper.early_stop(loss_val):
                print(f"--- Stopping early at epoch {epoch} ---")
                break

    training_result = {
        "loss_train": running_metrics["losses_train"],
        "loss_val": running_metrics["losses_val"],
        "detection_train": running_metrics["detection_performance_train"],
        "detection_val": running_metrics["detection_performance_val"],
        "box_train": running_metrics["box_performance_train"],
        "box_val": running_metrics["box_performance_val"],
        "mean_perf_train": running_metrics["mean_performance_train"],
        "mean_perf_val": running_metrics["mean_performance_val"],
        "strict_train": running_metrics["strict_performance_train"],
        "strict_val": running_metrics["strict_performance_val"],
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
    early_stop_patience: int | None = None,
    early_stop_min_delta: float = 0.0,
) -> dict[str, list[nn.Module | float | dict]]:
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
        "hyper_params": [],
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

            torch.manual_seed(seed)
            model = network()
            model.to(device)
            optimizer = optim.SGD(model.parameters(), **hparam)

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
                early_stop_patience,
                early_stop_min_delta,
            )

            grid_search_result["models"].append(model)
            grid_search_result["hyper_params"].append(hparam)
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

            print("\n", "-" * 3, "Final Performance", "-" * 3)
            for key, item in train_results.items():
                try:
                    if key in ["loss_train", "loss_val"]:
                        print(f"{key}:\t\t{item[-1]:.3f}")
                    elif key in ["detection_train", "mean_perf_train"]:
                        print(f"{key}:\t{item[-1]*100:.3f}%")
                    else:
                        print(f"{key}:\t\t{item[-1]*100:.3f}%")
                except IndexError:
                    pass

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
                results = compute_loc_performance(outputs, labels)
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
