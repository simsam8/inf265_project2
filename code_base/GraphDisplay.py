import matplotlib.pyplot as plt


class GraphDisplay:

    @staticmethod
    def plot_performance_over_time(train_loss, val_loss, title):
        _, ax = plt.subplots()
        ax.set_title(title)
        ax.plot(train_loss, label="train loss")
        ax.plot(val_loss, label="val loss")
        ax.legend()
        plt.show()

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
