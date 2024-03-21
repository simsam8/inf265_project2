---
title: 'Project 2: Object detection'
author:
- Simon Vedaa
- Sebastian Røkholt
numbersections: true
---

## Introduction
This report aims to explain our approach and design choices for defining, training and evaluating CNNs for the tasks of object localization and object detection. Additionally, we will discuss the performance of our models and evaluate our implementations.

For general information and setup guidance, please refer to the [README](README.md).

#### Contributions
- **Simon Vedaa**:
- **Sebastion Røkholt**:

## Approach and Design choices


### Task 1: Object localization
**Goal**: Train a CNN that can predict
 - Whether there is a single handwritten digit in a black and white image (0 = no, 1 = yes)
 - The bounding box of this digit (x-coordinate, y-coordinate, height and width)
 - Which digit it is (out of two possibilities: 0 or 1)

#### Loss function

To calculate the total cost of a localization prediction, we use an ensemble of three different loss metrics:
- **Detection loss**: Binary cross entropy on whether there is a class or not in the image.
- **Bounding box loss**: Mean square error on bounding box coordinates and size.
- **Classification loss**: Cross entropy loss on the class predictions.

Details:
 - We use cross entropy with logits, as the model's output isn't run through a sigmoid activation.
 - If there is no object in the image, only the detection loss is used. Otherwise the total loss is the sum of the three losses. 
 - Weights are updated after each batch, so the loss used for backpropagation is the mean loss for the current batch.


#### Performance function
TODO


#### Model architecture
TODO

### Task 2: Object detection

**Goal**: Train a CNN that can predict
 - Whether there are any handwritten digits in a black and white image (0 = no, 1 = yes)
 - The bounding box of each digit (x-coordinate, y-coordinate, height and width)
 - Which digits have been detected (out of two possibilities: 0 or 1)

#### Detection grid
TODO

#### Loss function

The loss function for object detection is similar to localization.
The loss becomes the sum of localization loss for each grid cell.
We vectorized the localization on grid cells to speed up loss calculation.
The final loss is the the batch mean of the summed losses.


#### Performance function
TODO

#### Model architectures
TODO

## Model training, selection and evaluation

### Training function

The same training function is used for both object localization and 
object detection. The parameter ```task```, specifies the task, and affects
the performance calculation while training.

The function ```train_models``` is used for training with different model 
architectures and hyperparameters.



### Model selection
TODO

- Grid search over hyperparameters
- Best model

### Evaluation
- Evaluation of the best model
- What worked well, what didn't. Challenges. Overall process
- Further improvements that could be made
