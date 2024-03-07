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
        print(
            "First data label (a vector of P_c, box_x_pos, box_y_pos, box_height, box_width, class label): ",
            labels[0],
        )
        print(f"Dataset size: {len(imgs)}")
        print(f"Dataset shape: {imgs.shape}")
        print(f"Image shape: {imgs[0].shape}")

        # TODO: Images with no object are labeled with class 1
        #       Add count for images with no objects
        class_counts = {}
        for label_vector in labels:
            label = int(label_vector[-1])
            if label in class_counts:
                class_counts[label] += 1
            else:
                class_counts[label] = 1

        print("Dataset class distribution")
        for label, count in sorted(class_counts.items()):
            print(f"Class {label}: {count} examples")

    @staticmethod
    def plot_performance_over_time(train_loss, val_loss, title, label1="Train Loss", label2="Val Loss"):
        _, ax = plt.subplots()
        ax.set_title(title)
        ax.plot(train_loss, label=label1)
        ax.plot(val_loss, label=label2)
        ax.legend()
        plt.show()

    # TODO: For both plot_instances methods:
    #       Images with no object and label 1 are plotted together. Separate them.

    @staticmethod
    def plot_instances(dataset, class_n, n_instances=4):
        fig, axes = plt.subplots(nrows=1, ncols=n_instances, tight_layout=True)
        imgs = [img for (img, label) in dataset if int(label[-1]) == class_n]

        for j, ax in enumerate(axes.flat):
            # Plot image
            ax.imshow(imgs[j].permute(1, 2, 0), cmap="gray")
            # Remove axis
            ax.axis("off")

        fig.suptitle(f"Label: {class_n}", y=0.7)
        plt.show()

    # TODO: Plot bounding box for predictions.
    #       Fix bug that removes bbox when rerunning code block in notebook.
    @staticmethod
    def plot_instances_with_bounding_box(
        dataset, class_n, n_instances=4, predictions=None
    ):
        fig, axes = plt.subplots(nrows=1, ncols=n_instances, tight_layout=True)
        imgs = [img for (img, label) in dataset if int(label[-1]) == class_n]
        b_boxes = [label[1:5] for (_, label) in dataset if int(label[-1]) == class_n]
        objects = [label[0] for (_, label) in dataset if int(label[-1]) == class_n]

        for j, ax in enumerate(axes.flat):
            # Don't plot bbox when there is no object
            if objects[j] == 0:
                img_out = imgs[j]
            # Plot when there is
            else:
                # Scale bbox cx, cy, width, height with dimensions of image
                b_boxes[j][0::2] *= imgs[j].shape[2]
                b_boxes[j][1::2] *= imgs[j].shape[1]

                # Apply bbox to image
                img_out = draw_bounding_boxes(
                    convert_image_dtype(imgs[j], torch.uint8),
                    box_convert(b_boxes[j].view(1, 4), "cxcywh", "xyxy"),
                    width=1,
                    colors="green",
                )
            ax.imshow(img_out.permute(1, 2, 0))
            ax.axis("off")
        fig.suptitle(f"Label: {class_n}", y=0.7)
        plt.show()
