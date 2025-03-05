# Machine Learning Classifiers for Diabetes Prediction

## Project Overview
This project involves the implementation and evaluation of two machine learning classifiers—**K-Nearest Neighbour (KNN)** and **Naive Bayes (NB)**—on the **Pima Indian Diabetes dataset**. The goal is to predict whether a patient shows signs of diabetes (class: "yes" or "no") based on their personal characteristics and test measurements. The project also includes the evaluation of other classifiers using **Weka**, a popular machine learning tool, and investigates the effect of **Correlation-based Feature Selection (CFS)** on classifier performance.

## Key Features
- **Classifier Implementation**: Implement the K-Nearest Neighbour and Naive Bayes algorithms from scratch in Python.
- **Data Preprocessing**: Normalize the dataset using Weka's normalization filter and prepare it for classification tasks.
- **Stratified Cross-Validation**: Perform 10-fold stratified cross-validation to evaluate the performance of the classifiers.
- **Feature Selection**: Apply Correlation-based Feature Selection (CFS) using Weka to reduce the number of features and analyze its impact on classifier performance.
- **Comparison with Weka**: Compare the performance of the implemented classifiers with other classifiers available in Weka, such as ZeroR, 1R, Decision Tree (J48), Multi-Layer Perceptron (MLP), Support Vector Machine (SMO), and Random Forest (RF).

## Technical Details
- **Programming Language**: Python (for implementing KNN and Naive Bayes classifiers)
- **Tools**: Weka (for data preprocessing, feature selection, and comparison with other classifiers)
- **Dataset**: Pima Indian Diabetes dataset (768 instances, 8 numeric attributes, 2 classes: "yes" or "no")
- **Evaluation Method**: 10-fold stratified cross-validation
- **Feature Selection**: Correlation-based Feature Selection (CFS) with Best-First Search in Weka

## Project Tasks
1. **Data Preprocessing**:
   - Normalize the dataset using Weka's normalization filter.
   - Add and remove headers to ensure compatibility with Weka.
   - Save the preprocessed file as `pima.csv`.

2. **Classifier Implementation**:
   - Implement the K-Nearest Neighbour (KNN) algorithm using Euclidean distance.
   - Implement the Naive Bayes (NB) algorithm using a probability density function for numeric attributes.
   - Handle ties by choosing the "yes" class.

3. **Stratified Cross-Validation**:
   - Generate `pima-folds.csv` containing 10 stratified folds for cross-validation.
   - Evaluate the classifiers using 10-fold stratified cross-validation and report average accuracy.

4. **Feature Selection**:
   - Apply Correlation-based Feature Selection (CFS) using Weka to reduce the number of features.
   - Save the reduced dataset as `pima-CFS.csv`.

5. **Comparison with Weka**:
   - Run multiple classifiers (ZeroR, 1R, IBk, NB, J48, MLP, SMO, RF) in Weka using 10-fold cross-validation.
   - Compare the performance of the implemented KNN and NB classifiers with Weka's classifiers, both with and without feature selection.

## How It Works
1. **Data Preprocessing**: The dataset is normalized, and headers are added/removed to ensure compatibility with Weka.
2. **Classifier Implementation**: The KNN and Naive Bayes classifiers are implemented in Python, with KNN using Euclidean distance and Naive Bayes assuming a normal distribution for numeric attributes.
3. **Cross-Validation**: The dataset is divided into 10 stratified folds, and the classifiers are evaluated using 10-fold cross-validation.
4. **Feature Selection**: CFS is applied to reduce the number of features, and the impact on classifier performance is analyzed.
5. **Comparison**: The performance of the implemented classifiers is compared with Weka's classifiers, both with and without feature selection.

## Acknowledgments
This project was developed as part of a machine learning assignment for COMP3308, focusing on the implementation and evaluation of classifiers for a real-world dataset. The goal is to gain hands-on experience with machine learning algorithms, data preprocessing, and feature selection techniques.
