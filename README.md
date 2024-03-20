# INF265 Project 2: Object Detection

This repository contains the code, documentation and report for a deep learning project completed as part of the course INF265: Deep Learning at the University of Bergen. 
The project assignment was completed as a collaboration between [Simon Vedaa](https://github.com/simsam8) and [Sebastian Røkholt](https://github.com/SebastianRokholt). It was handed in on the 22nd of March 2024. 

The aim of the project was to define, train, select and evaluate convolutional neural networks (CNNs) for solving two different tasks: </br> 
1. **Object localization** for images containing a single digit.
2. **Object detection** for images containing multiple digits.

We successfully implemented two different CNNs and evaluated their performance. As a result, we gained two things: 
  1. **Valuable experience with PyTorch**. We had to perform image processing, build convolutional model architectures and do plenty of tensor wrangling/manipulation/operations. 
  2. **A deeper understanding of object detection**, including how to predict and plot bounding boxes for detections, how to deal with images with no objects, and how to evaluate models for object detection tasks.

Feel free to clone or fork the repository, and don't hesitate to contact us or leave an issue if you have any questions. 
If you found the repository helpful, please leave us a star! 

## Setup

The project was created with Python 3.11. To run and replicate our results, make sure to install the project dependencies. 
For Windows, run `pip install -r requirements.txt`.
For Linux, run `pip install -r requirements-linux.txt`.

To view and run the notebook, launch Jupyter Notebook with the `jupyter notebook` command in the terminal and simply select the .ipynb file from the directory to open it.

To reproduce our work with identical results, you should set the seed for the randoom state to 265 and train on an Nvidia GPU (CUDA).

## File Structure

```
Project 2
├── docs    # General information about the assignment
│   ├── inf265_v24_project2.pdf    # The assignment text written by the professor
│   ├── project_checklist.pdf    # Some evaluation requirements to look through before handing in the project
├── src    # Contains the Python code for modules used in the notebooks
│   ├── DataAnalysis.py    # Code for plotting, exploration etc. 
│   ├── functions.py    # For training, performance metrics, ++
│   ├── models.py    # Model architectures
│   └── object_detection.py    # Helper functions for object detection
├── data    # Contains the Parquet files used in the project
│   ├── detection_test.pt
│   ├── detection_train.pt
│   ├── detection_val.pt
│   ├── list_y_true_test.pt
│   ├── list_y_true_train.pt
│   ├── list_y_true_val.pt
│   ├── localization_train.pt
│   ├── localization_test.pt
│   └── localization_val.pt
├── notebooks    # Jupyter Notebooks for running the project code and viewing results
|   ├── object_localization.ipynb    # Solves the object localization task
│   └── object_detection.ipynb    # Solves the object detection task
├── README.md    # You are reading this now
├── requirements.txt    # Windows-specific list of the Python dependencies for this project. 
├── requirements-linux.txt    # Linux-specific list of the Python dependencies for this project. 
├── report.pdf    # Details the design choices and general approach for solving the assignment.
└── report.md    # Markdown for generating the project report.
```
