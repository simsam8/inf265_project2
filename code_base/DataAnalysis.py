import matplotlib.pyplot as plt
import torch
from torchvision.utils import draw_bounding_boxes
from torchvision.ops import box_convert
from torchvision.transforms.functional import convert_image_dtype


class DataAnalysis:

    @staticmethod
    def get_summary(dataset):
        # Convert data to PyTorch Tensors
        imgs = torch.stack([img for img, _ in dataset])
        labels = torch.stack([label for _, label in dataset])

        # Display useful information about the data
        print(f"Type of images in dataset: {type(imgs[0])}")
        print(f"First image in dataset: {imgs[0]}")
        print(
            "First data label (a vector of P_c, box_x_pos, box_y_pos, box_height, box_width, class label): ",
            labels[0],
        )
        print(f"Dataset size: {len(imgs)}")
        print(f"Dataset shape: {imgs.shape}")
        print(f"Image shape: {imgs[0].shape}")

        class_counts = {}
        for label_vector in labels:
            label = int(label_vector[-1])
            is_object = int(label_vector[0]) == 1
            if is_object and label in class_counts:
                class_counts[label] += 1
            elif is_object:
                class_counts[label] = 1
            else:
                if -1 in class_counts:
                    class_counts[-1] += 1
                else:
                    class_counts[-1] = 1

        print("\n----- Dataset class distribution (-1 for no object) -----")
        for label, count in sorted(class_counts.items()):
            print(f"Class {label}: {count} examples")

    @staticmethod
    def plot_performance_over_time(
        train_loss, val_loss, title, label1="Train Loss", label2="Val Loss"
    ):
        _, ax = plt.subplots()
        ax.set_title(title)
        ax.plot(train_loss, label=label1)
        ax.plot(val_loss, label=label2)
        ax.legend()
        plt.show()

    @staticmethod
    def plot_instances_with_bounding_box(
        dataset, class_n=None, n_instances=4, predictions=None
    ):
        """
        Plot instances and their bounding box from the given dataset.
        Optionally plot the bounding box of predictions.
        """
        fig, axes = plt.subplots(nrows=1, ncols=n_instances, tight_layout=False)

        imgs = []
        bboxes_true = []
        bboxes_pred = []
        object_true = []
        object_pred = []
        class_true = []
        class_pred = []
        img_index = None
        for i, (img, label) in enumerate(dataset):
            if len(imgs) == n_instances:
                break
            # Images with no object
            if class_n is None:
                if int(label[0]) == 0:
                    img_index = i
                    object_true.append(None)
                    class_true.append(None)
                    imgs.append(img.clone())
            else:
                if int(label[-1]) == class_n and int(label[0]) == 1:
                    img_index = i
                    object_true.append(label[0].bool())
                    class_true.append(label[-1].int())
                    imgs.append(img.clone())
                    bboxes_true.append(label[1:5].clone())

            # Get prediction to the corresponding image
            if predictions is not None and i == img_index:
                is_object = predictions[i][0] > 0
                object_pred.append(is_object)
                predicted_class = (
                    torch.argmax(predictions[i][5:]) if is_object else None
                )
                class_pred.append(predicted_class)
                bboxes_pred.append(predictions[i][1:5].clone())

        for j, ax in enumerate(axes.flat):
            if predictions is not None:
                ax.set_title(
                    f"Pred object: {object_pred[j]}\nPred class: {class_pred[j]}"
                )
            img_out = imgs[j]

            # Don't plot bbox when there is no object
            if class_n is None:
                img_out = imgs[j]
            else:
                # Scale bbox cx, cy, width, height with dimensions of image
                bboxes_true[j][0::2] *= img_out.shape[2]
                bboxes_true[j][1::2] *= img_out.shape[1]

                # Apply bbox to image
                img_out = draw_bounding_boxes(
                    convert_image_dtype(img_out, torch.uint8),
                    box_convert(bboxes_true[j].view(1, 4), "cxcywh", "xyxy"),
                    width=1,
                    colors="green",
                )
            if predictions is not None:
                # clip box values less than 0
                bboxes_pred[j] = torch.clamp_min(bboxes_pred[j], 0)
                # Scale bbox cx, cy, width, height with dimensions of image
                bboxes_pred[j][0::2] *= img_out.shape[2]
                bboxes_pred[j][1::2] *= img_out.shape[1]

                # Apply bbox to image
                img_out = draw_bounding_boxes(
                    convert_image_dtype(img_out, torch.uint8),
                    box_convert(bboxes_pred[j].view(1, 4), "cxcywh", "xyxy"),
                    width=1,
                    colors="red",
                )

            ax.imshow(img_out.permute(1, 2, 0))
            ax.axis("off")
        fig.suptitle(f"Label: {class_n}", y=0.7)
        plt.show()
