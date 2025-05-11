# Pet Classification Model Comparison with Varying Training Epochs

## Overview

This repository contains a Python script (Jupyter Notebook) that trains and compares the performance of three Convolutional Neural Network (CNN) models for classifying pet images. The models share the same architecture but are trained for a different number of epochs: 100, 200, and 300. The primary goal is to visually and quantitatively analyze the impact of training duration on the model's accuracy and generalization ability.

## Project Structure

* `pet_classification_comparison.ipynb`: Jupyter Notebook containing the Python code for data loading, preprocessing, model definition, training, evaluation, and comparison.
* `data/PetImages/`: (Expected) Directory containing the pet image dataset. This directory should have subdirectories for each pet class (e.g., `cats`, `dogs`). **You need to place your dataset here or update the `data_dir` variable in the notebook.**
* `models/`: (Optional) Directory where trained model weights can be saved (currently commented out in the code).
* `README.md`: This file.
* `requirements.txt`: (Optional) A file listing the necessary Python libraries.

## Model Architecture

The CNN model implemented in this notebook consists of the following layers:

1.  Convolutional layer with 32 filters, a 3x3 kernel, and ReLU activation.
2.  Max-pooling layer with a 2x2 pool size.
3.  Convolutional layer with 64 filters, a 3x3 kernel, and ReLU activation.
4.  Max-pooling layer with a 2x2 pool size.
5.  Convolutional layer with 128 filters, a 3x3 kernel, and ReLU activation.
6.  Max-pooling layer with a 2x2 pool size.
7.  Flatten layer to convert the 2D feature maps to a 1D vector.
8.  Dense layer with 128 units and ReLU activation.
9.  Output dense layer with a sigmoid activation function for binary classification (or softmax for multi-class).

## Training and Evaluation

The `pet_classification_comparison.ipynb` notebook performs the following steps:

1.  **Data Loading and Preprocessing:**
    * Loads image data from the specified directory using `ImageDataGenerator`.
    * Applies rescaling to normalize pixel values.
    * Implements data augmentation techniques (rotation, shifts, shear, zoom, horizontal flip) to enhance model robustness.
    * Splits the data into training and validation sets.

2.  **Model Definition:**
    * Defines the CNN model architecture as described above.

3.  **Model Training:**
    * Trains three separate instances of the same model architecture for 100, 200, and 300 epochs, respectively.
    * Uses the Adam optimizer and binary cross-entropy loss (or categorical cross-entropy for multi-class).
    * Monitors training and validation accuracy during the training process.

4.  **Accuracy Comparison:**
    * Generates plots comparing the training and validation accuracy curves for the three models across their respective training epochs. This allows for a visual assessment of how training duration impacts performance and potential overfitting.
    * Prints the final validation loss and accuracy for each of the three models for a quantitative comparison.

## Getting Started

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd Pet-Classification-Model-Comparison
    ```

2.  **Prepare the Dataset:**
    * Download a pet image dataset (e.g., Cats vs. Dogs dataset).
    * Create the `data/PetImages/` directory within the repository.
    * Organize the images into subdirectories based on their class names (e.g., `cats`, `dogs`).

3.  **Install Dependencies:**
    ```bash
    pip install tensorflow matplotlib scikit-learn numpy
    ```
    (You might want to create a `requirements.txt` file with these dependencies for easier installation.)

4.  **Run the Notebook:**
    ```bash
    jupyter notebook pet_classification_comparison.ipynb
    ```
    Execute the cells in the notebook sequentially to train and compare the models.

5.  **Analyze the Results:**
    * Observe the generated plots to understand how training accuracy and validation accuracy evolve with more epochs for each model.
    * Examine the final validation metrics printed in the notebook to compare the quantitative performance of the models.

## Expected Outcome

By running this notebook, you should be able to:

* Visualize the training and validation accuracy curves for models trained with 100, 200, and 300 epochs.
* Compare the final validation accuracy achieved by each model.
* Gain insights into the impact of the number of training epochs on model performance, including potential benefits of longer training and the risk of overfitting.

## Potential Improvements

* Experiment with different CNN architectures.
* Explore more advanced data augmentation techniques.
* Implement early stopping to prevent overfitting.
* Fine-tune hyperparameters (learning rate, batch size, etc.).
* Evaluate the models on a separate test set for a more robust assessment of generalization.
* Save the trained models for future use.
