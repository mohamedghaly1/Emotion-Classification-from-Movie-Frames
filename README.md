# Emotion Classification from Movie Frames

This project focuses on building an end-to-end deep learning system to classify human emotions from images extracted from Egyptian movies. It involves custom dataset creation, model implementation from scratch, library-based CNN design, and result evaluation.

## 🎯 Project Objectives

Create a labeled dataset of facial expressions.

Implement a convolutional neural network (CNN) from scratch.

Train a second CNN model using deep learning libraries.

Evaluate model performance on unseen test data.

## 🗂️ Milestone 1: Dataset Creation

Goal: Build a dataset of facial images labeled with emotions: Happy, Sad, Angry, Surprised, Neutral.

Steps:

Extract frames from Egyptian movies.

Collect 100 images per class (1 class per team member).

Ensure 640x480 resolution and clear, centered faces.

Remove duplicates and low-quality samples.

File naming format: C_T_N.png

Save as PNG, organized in class-specific folders.

### 🧠 Milestone 2: Model Development

Goal: Develop two models for emotion classification.

### 🔧 Data Preparation (5%)

Resize all images to 512x512x3.

Split into 70% train, 20% validation, 10% test.

Ensure balanced class distribution.

### 🧱 Model 1: CNN from Scratch (35%)

ConvLayer: custom 3x3x3 filters.

PoolingLayer: Max/Average pooling.

Build a 3-layer convolutional pipeline.

Apply activation and flattening.

Reduce to 1x128 vector.

Use K-Means clustering on feature vectors.

### 🧪 Model 2: CNN with DL Library (30%)

3 convolutional layers + ReLU + MaxPooling.

Filter sizes: 3x3 to 7x7, depths: 32, 64, 32, 16.

Fully connected layer (Sigmoid), output layer (Softmax).

Train and validate on prepared dataset.

## 📊 Milestone 3: Evaluation & Reporting (20%)

Goal: Evaluate model performance and summarize results.

Tasks:

Plot accuracy vs iterations curve.

Perform 4-fold cross-validation.

Compute and analyze the confusion matrix.

Create architecture diagrams with hyperparameter details.

Document preprocessing/postprocessing steps.

Submit final report or presentation with findings.

## 🧰 Tools & Libraries

Python, NumPy, OpenCV

Scikit-learn (for K-means, evaluation)

TensorFlow / PyTorch (Model 2)

Matplotlib (visualizations)

## Output
### Accuracy vs Number of Iterations
A curve for accuracy versus number of epochs (iterations) was plotted using validation accuracy across each epoch for the 4-fold cross-validation. The plot titled "Validation Accuracy per Fold" shows the improvement in validation accuracy with training epochs for each fold, demonstrating the model's learning progression. This allows monitoring of potential overfitting or underfitting.

![image](https://github.com/user-attachments/assets/bd73ded4-e0c6-4cd3-a6d5-ea07af3b70b8)

### K-Fold Cross-Validation Results
A 4-fold cross-validation approach was used to evaluate the model's performance. The model was trained and validated on 4 different folds of the dataset. Below are the results:
•	Fold 1 Accuracy: 56.77%
•	Fold 2 Accuracy: 61.96%
•	Fold 3 Accuracy: 59.89%
•	Fold 4 Accuracy: 58.19%
Average Accuracy: 59.20%
This demonstrates consistent model performance across the folds.

![image](https://github.com/user-attachments/assets/1f6c63bc-42ba-4a8a-a707-749518f1f819)

### Confusion Matrix and Analysis
The trained model was evaluated on a separate 10% test set. The confusion matrix below shows the performance across the four emotion classes:

![image](https://github.com/user-attachments/assets/173c1360-1b7c-4734-9935-1bdf6d9c8752)

 
Class Accuracy:
•	Angry: 26/26+3+18+18= 34.21%
•	Happy: 43/5+43+12+2= 71.67%
•	Sad: 36/12+3+36+5= 63.16
•	Neutral: 12/7+1+10+12 = 40.00%
Overall Accuracy: 26+43+36+12/213 = 55.40%
Error Rate: 1 – 0.5540 = 44.60
The confusion matrix highlights stronger classification for the "Happy" and "Sad" classes, with more misclassifications observed in the "Angry" and "Neutral" categories.


### CNN Network Architecture and Hyperparameters
![image](https://github.com/user-attachments/assets/b0b6aca3-bef3-4040-90b2-53ab757c9992)

The CNN was built using TensorFlow/Keras and consists of the following layers:
#### Model Summary:
•	Input Shape: (224, 224, 3)
•	Conv2D: 5 convolutional blocks
•	Pooling: MaxPooling2D after each Conv layer
•	Normalization: BatchNormalization after each pool layer
•	Dropout: 30% after final convolution
•	Dense Layers: 1 hidden (64 units), 1 output (4 classes)
#### Hyperparameters:
•	Optimizer: Adam
•	Loss: Sparse Categorical Crossentropy
•	Activation Functions: ReLU (Conv layers), Sigmoid (Dense), Softmax (Output)
•	Regularization: L2 (1e-4)
•	Epochs: 20
•	Batch Size: 8
•	K-Folds: 4
•	Seed: 42

### Data Preprocessing and Post-Processing
#### Preprocessing Steps:
•	All images resized to (224, 224)
•	Pixel values normalized to [0, 1] by dividing by 255
•	Labels encoded as integers (0–3)
•	Data loaded using load_img and converted using img_to_array
•	Used stratified split for maintaining class balance in test and validation sets
#### Post-processing:
•	Predictions were collected as softmax probabilities and converted to class predictions using argmax
•	Confusion matrix created using sklearn utilities


