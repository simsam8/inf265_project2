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

**Step 1: Install Git LFS**</br>
Download Git LFS from [the official website](https://git-lfs.com/) or use a package manager to install it on your system.
    Windows: Download and run the installer from the website .
    macOS: Use Homebrew with `brew install git-lfs`.
    Linux: Use your distribution’s package manager, for example, `sudo apt-get install git-lfs` for Debian/Ubuntu.

After installation, open your terminal or Git Bash (on Windows) and run the following command to set up Git LFS:
```bash 
git lfs install 
```

**Step 2: Clone the repository**</br>
If you haven't done so already, clone this repo with
```bash
git clone https://github.com/simsam8/inf265_project2.git
```
and navigate to the project repository: 
```bash
cd inf265_project2
```

**Step 3: Pull data folder with Git LFS**</br>
To ensure that all LFS-tracked files are correctly downloaded, use the following command:
```bash
git lfs pull
```

**Step 4: Install project dependencies**</br>
The project was created with Python 3.11. To run and replicate our results, make sure to install the project's Python dependencies:
For Windows, run `pip install -r requirements.txt`.
For Linux, run `pip install -r requirements-linux.txt`.


**Step 5: Launch Jupyter Notebook**</br>
To view and run the notebook, launch Jupyter Notebook with the `jupyter notebook` command in the terminal and simply select the .ipynb file from the directory to open it.

To reproduce our work with identical results, you should set the seed for the random state to 265 and train on an Nvidia GPU (CUDA).

## File Structure

```
inf265_project2
├── docs    # General information about the assignment
│   ├── inf265_v24_project2.pdf    # The assignment text written by the professor
│   ├── project_checklist.pdf    # Some evaluation requirements to look through before handing in the project
├── src    # Contains the Python code for modules used in the notebooks
│   ├── DataAnalysis.py    # Code for plotting, exploration etc. 
│   ├── train.py    # Code for model training and selection, including loss functions
│   ├── models.py    # Model architectures
│   └── object_detection.py    # Helper functions for object detection
├── data    # Contains the Parquet files used in the project. NB! Download requires Git LFS
│   ├── localization_train.pt
│   ├── localization_test.pt
│   ├── localization_val.pt
│   ├── detection_test.pt
│   ├── detection_train.pt
│   ├── detection_val.pt
│   ├── list_y_true_test.pt  # Used to create grid labels
│   ├── list_y_true_train.pt # Used to create grid labels
│   └── list_y_true_val.pt  # Used to create grid labels
├── notebooks    # Jupyter Notebooks for running the project code and viewing results
|   ├── object_localization.ipynb    # Solves the object localization task
│   └── object_detection.ipynb    # Solves the object detection task
├── imgs    # Plots generated by the notebooks
├── README.md 
├── requirements.txt    # Windows-specific list of the Python dependencies for this project. 
├── requirements-linux.txt    # Linux-specific list of the Python dependencies for this project. 
├── report.pdf    # Details the design choices and general approach for solving the assignment.
└── report.md    # Markdown for generating the project report.
```
