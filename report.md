---
title: 'Project 2: Object detection'
author:
- Simon Vedaa
- Sebastian Røkholt
numbersections: true
---

# Introduction
This report aims to explain our approach and design choices for defining, training and evaluating CNNs for the tasks of object localization and object detection. Additionally, we will discuss the performance of our models and evaluate our implementations.

For general information and setup guidance, please refer to the [README](README.md).

## Contributions
- **Simon Vedaa**:
- **Sebastion Røkholt**:

# Approach and Design choices


## Object localization
**Task**: Train a CNN that can predict
 - Whether there is a single handwritten digit in a black and white image (0 = no, 1 = yes)
 - The bounding box of this digit (x-coordinate, y-coordinate, height and width)
 - Which digit it is (out of two possibilities: 0 or 1)

### Loss function

The loss function for localization calculates three different loss functions. 
There is no sigmoid activation on the model output, so cross entropy with logits 
is used.

- Detection loss: Binary cross entropy on whether there is a class or not in the image.
- Bounding Box loss: Mean square error on bounding box coordinates and size.
- Classification loss: Cross entropy loss on predicted classes.

If there is no object in the image, only the detection loss is calculated.
Otherwise the loss is the sum of the three losses.
The final is then the mean of losses in the batch.


- performance functions
- Models and hyperparameters

### Performance function
TODO


### Model architectures
TODO


## Object detection

### Loss function

The loss function for object detection is similar to localization.
The loss becomes the sum of localization loss for each grid cell.
We vectorized the localization on grid cells to speed up loss calculation.
The final loss is the the batch mean of the summed losses.


### Performance function
TODO

### Model architectures
TODO

## Training function

The same training function is used for both object localization and 
object detection. The parameter ```task```, specifies the task, and affects
the performance calculation while training.

The function ```train_models``` is used for training with different model 
architectures and hyperparameters.



## Model selection and evaluation



- Models and hyperparameters
