import torch
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


# TODO: not tested properly
def _local_to_global(in_tensor, grid_dimensions):
    x_local, y_local, w_local, h_local = in_tensor[1:5]
    rows, cols = grid_dimensions

    cell_w = 1 / cols
    cell_h = 1 / rows

    x_global = x_local / cols + cell_w
    y_global = y_local / rows + cell_h
    w_global = w_local / cols
    h_global = h_local / rows

    global_tensor = torch.tensor(
        [in_tensor[0], x_global, y_global, w_global, h_global] + list(in_tensor[5:])
    )

    return global_tensor


def _create_grid_boxes(
    label_list: list[torch.Tensor], grid_dimensions: tuple[int, int]
) -> torch.Tensor:
    h, w = grid_dimensions
    output_box = torch.zeros((h, w, 6))
    for label in label_list:
        row, col, coords = _global_to_local(label, grid_dimensions)
        output_box[row][col] = coords
    return output_box


def get_converted_data(
    grid_dimensions: tuple[int, int]
) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    """
    Get training, validation, and test datasets with labels
    converted into to a given grid size.
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
