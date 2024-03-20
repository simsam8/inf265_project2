---
title: 'Project 2: Object detection'
author:
- Simon Vedaa
- Sebastian Røkholt
numbersections: true
---

# Introduction

## Contributions


- **Simon Vedaa**:
- **Sebastion Røkholt**:

# Approach and Design choices


## Object localization

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
