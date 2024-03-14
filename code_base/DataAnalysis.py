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
        fig, axes = plt.subplots(nrows=1, ncols=n_instances, tight_layout=True)

        if class_n is None:
            imgs = [img.clone() for (img, label) in dataset if int(label[0]) == 0]
        else:
            imgs = [
                img.clone()
                for (img, label) in dataset
                if int(label[-1]) == class_n and int(label[0]) == 1
            ]
            b_boxes = [
                label[1:5].clone()
                for (_, label) in dataset
                if int(label[-1]) == class_n and int(label[0] == 1)
            ]

        if predictions is not None:
            b_boxes_prediction = [
                pred[1:5].clone()
                for pred in predictions
                if torch.argmax(pred[5:]) == class_n and pred[0] >= 0
            ]

        for j, ax in enumerate(axes.flat):
            img_out = imgs[j]

            # Don't plot bbox when there is no object
            if class_n is None:
                img_out = imgs[j]
            else:
                # Scale bbox cx, cy, width, height with dimensions of image
                b_boxes[j][0::2] *= img_out.shape[2]
                b_boxes[j][1::2] *= img_out.shape[1]

                # Apply bbox to image
                img_out = draw_bounding_boxes(
                    convert_image_dtype(img_out, torch.uint8),
                    box_convert(b_boxes[j].view(1, 4), "cxcywh", "xyxy"),
                    width=1,
                    colors="green",
                )
            if predictions is not None:
                # skip if there is no predicted bounding box
                if j < len(b_boxes_prediction):
                    # clip box values less than 0
                    b_boxes_prediction[j] = torch.clamp_min(b_boxes_prediction[j], 0)
                    # Scale bbox cx, cy, width, height with dimensions of image
                    b_boxes_prediction[j][0::2] *= img_out.shape[2]
                    b_boxes_prediction[j][1::2] *= img_out.shape[1]
                    img_out = draw_bounding_boxes(
                        img_out,
                        box_convert(b_boxes_prediction[j].view(1, 4), "cxcywh", "xyxy"),
                        width=1,
                        colors="red",
                    )

            ax.imshow(img_out.permute(1, 2, 0))
            ax.axis("off")
        fig.suptitle(f"Label: {class_n}", y=0.7)
        plt.show()
