import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset


def _global_to_local(
    in_tensor: torch.Tensor, grid_dimensions: tuple[int, int]
) -> tuple[int, int, torch.Tensor]:
    """
    Convert global bounding box values to local.

    Parameters:
    in_tensor: tensor containing bounding box in the form of ->
                [p_c, x, y, w, h, c ...]
    grid_dimensions: tuple of n_rows and n_cols

    return: row_index, col_index, converted tensor
    """
    x, y, w, h = in_tensor[1:5]
    rows, cols = grid_dimensions

    cell_w = 1 / cols
    cell_h = 1 / rows

    row_index = int(y / cell_h)
    col_index = int(x / cell_w)

    row_index = max(0, min(row_index, rows - 1))
    col_index = max(0, min(col_index, cols - 1))

    x_local = (x - col_index * cell_w) * cols
    y_local = (y - row_index * cell_h) * rows
    w_local = w * cols
    h_local = h * rows

    local_tensor = torch.tensor(
        [in_tensor[0], x_local, y_local, w_local, h_local] + list(in_tensor[5:])
    )

    return row_index, col_index, local_tensor


def _local_to_global(
    in_tensor: torch.Tensor, grid_dimensions: tuple[int, int], pos: int
) -> torch.Tensor:
    """
    Converts local bounding box values to global.

    Parameters:
    in_tensor: tensor containing bounding box in the form of ->
                [p_c, x, y, w, h, c ...]
    grid_dimensions: tuple of n_rows and n_cols
    pos: index of the grid cell in flattened list

    return: converted tensor
    """
    x_local, y_local, w_local, h_local = in_tensor[1:5]
    rows, cols = grid_dimensions
    cell_w = 1 / cols
    cell_h = 1 / rows

    row_index = pos // cols
    col_index = pos % cols

    x_global = x_local / cols + cell_w * col_index
    y_global = y_local / rows + cell_h * row_index
    w_global = w_local / cols
    h_global = h_local / rows

    global_tensor = torch.tensor(
        [in_tensor[0], x_global, y_global, w_global, h_global] + list(in_tensor[5:])
    )

    return global_tensor


def _create_grid_boxes(
    label_list: list[torch.Tensor], grid_dimensions: tuple[int, int]
) -> torch.Tensor:
    """
    Converts a list of tensors into a nxm grid tensor.

    Parameters:
    label_list: list of tensors containing bounding boxes in the form of ->
                [p_c, x, y, w, h, c ...]
    grid_dimensions: tuple of n_rows and n_cols

    return: tensor with shape n_classes x rows x cols
    """
    h, w = grid_dimensions
    output_box = torch.zeros((h, w, 6))
    for label in label_list:
        row, col, coords = _global_to_local(label, grid_dimensions)
        output_box[row][col] = coords
    output_box = output_box.permute(2, 0, 1)  # CxHxW
    return output_box



def compute_loc_performance(
    y_pred: torch.Tensor, y_true: torch.Tensor, iou_threshold: float = 0.5
) -> dict:
    """
    Computes the accuracy of localization predictions.
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


def MAP_preprocess(input_tensor: torch.Tensor, threshold: float = 0.5) -> list[dict]:
    """Preprocesses the label or prediction tensor for torchmetrics.MAP calculation.
    Box format: cx cy w h
    Return: A dictionary of Tensor values as specified in torchmetrics docs."""
    dict_for_MAP = []
    # New dimensions: (batch size, grid height, grid width, channels)
    input_tensor = input_tensor.permute(0, 2, 3, 1)
    grid_dimensions = (input_tensor.shape[1], input_tensor.shape[2])
    # Loop over all images in the batch
    for batch in range(input_tensor.shape[0]):
        # Store prediction or target values in a dict
        img_detect = {"boxes": [], "scores": [], "labels": []}

        # Extracts data from tensor
        cell_index = 0
        for h in range(grid_dimensions[0]):
            for w in range(grid_dimensions[1]):
                cell = input_tensor[batch, h, w]
                # If detection confidence above the detection threshold
                if cell[0] >= threshold:
                    # Extract local box coords and convert to global coords
                    box = _local_to_global(cell, grid_dimensions, cell_index)[1:5]
                    label = F.sigmoid(cell[5]).long()
                    img_detect["boxes"].append(box.tolist())
                    img_detect["scores"].append(cell[0].item())
                    img_detect["labels"].append(label.item())
                cell_index += 1

        # Converts all dict values to Tensors and stores image detection data in dict
        img_detect["boxes"] = torch.tensor(img_detect["boxes"])
        img_detect["scores"] = torch.tensor(img_detect["scores"])
        img_detect["labels"] = torch.tensor(img_detect["labels"])
        dict_for_MAP.append(img_detect)

    return dict_for_MAP


def get_converted_data(
    grid_dimensions: tuple[int, int]
) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    """
    Get training, validation, and test datasets with labels
    converted into to a given grid size.

    Parameters:
    grid_dimensions: dimensions of the grid

    return: converted train, val, test datasets
    """

    list_train = torch.load("data/list_y_true_train.pt")
    list_val = torch.load("data/list_y_true_val.pt")
    list_test = torch.load("data/list_y_true_test.pt")
    label_sets = [list_train, list_val, list_test]

    imgs_train = torch.load("data/detection_train.pt")
    imgs_val = torch.load("data/detection_val.pt")
    imgs_test = torch.load("data/detection_test.pt")
    img_sets = [imgs_train, imgs_val, imgs_test]

    output_datasets = []
    for img_set, label_set in zip(img_sets, label_sets):
        labels = [_create_grid_boxes(label, grid_dimensions) for label in label_set]
        imgs = [img for img, _ in img_set]
        labels_tensor = torch.stack(labels, dim=0)
        imgs_tensor = torch.stack(imgs, dim=0)
        tensor_data = TensorDataset(imgs_tensor, labels_tensor)
        output_datasets.append(tensor_data)

    return tuple(output_datasets)
